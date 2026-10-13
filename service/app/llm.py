"""The one place that talks to the LLM provider.

Everything else calls complete_json(), so swapping Groq for another provider
means rewriting this file only.
"""

import json
import time

from groq import APIError, BadRequestError, Groq, RateLimitError

from app.config import (
    DAILY_TOKEN_BUDGET,
    GROQ_API_KEY,
    LLM_MAX_RETRIES,
    LLM_MODEL,
    LLM_TIMEOUT_SECONDS,
)
from app.metrics import tokens_today

RATE_LIMIT_WAITS = 3
MAX_RATE_LIMIT_WAIT_SECONDS = 60

_client = None


class LLMUnavailable(RuntimeError):
    """No API key, the daily token budget is spent, or the provider still failed."""


def _get_client() -> Groq:
    global _client
    if not GROQ_API_KEY:
        raise LLMUnavailable("GROQ_API_KEY is not set")
    if _client is None:
        _client = Groq(
            api_key=GROQ_API_KEY,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=LLM_MAX_RETRIES,
        )
    return _client


def _create_with_retries(system: str, user: str):
    """Call the model, retrying the two transient failures the free tier shows.

    On a per-minute rate limit it waits as long as Groq asks, then retries.

    The SDK's own retries back off for seconds only, but the free tier's limit is
    per minute (8000 tokens): a burst of cards needs up to a minute to clear.
    """
    for attempt in range(RATE_LIMIT_WAITS + 1):
        try:
            return _get_client().chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,  # factual extraction, not creativity
            )
        except BadRequestError as error:
            # In JSON mode Groq answers 400 "json_validate_failed" when the model's
            # output was not valid JSON. That is random, not a bad request: retry.
            if "json_validate_failed" not in str(error) or attempt == RATE_LIMIT_WAITS:
                raise
        except RateLimitError as error:
            wait = float(error.response.headers.get("retry-after", 20))
            # A wait beyond a minute means the daily limit, not the minute one: give up.
            if attempt == RATE_LIMIT_WAITS or wait > MAX_RATE_LIMIT_WAIT_SECONDS:
                raise
            time.sleep(wait + 1)


def complete_json(system: str, user: str) -> tuple[dict | None, dict]:
    """Ask the model for a JSON object.

    Returns (parsed object or None if the reply is not valid JSON, token usage).
    """
    spent = tokens_today()
    if spent >= DAILY_TOKEN_BUDGET:
        raise LLMUnavailable(
            f"daily token budget spent ({spent} of {DAILY_TOKEN_BUDGET} tokens)"
        )
    try:
        response = _create_with_retries(system, user)
    except APIError as error:
        # Rate limit or provider outage left after the SDK's retries: report it
        # as "unavailable" so the caller (n8n) can retry later, not crash with 500.
        raise LLMUnavailable(f"LLM call failed: {type(error).__name__}") from error
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }
    try:
        parsed = json.loads(response.choices[0].message.content or "")
    except json.JSONDecodeError:
        parsed = None
    return (parsed if isinstance(parsed, dict) else None), usage
