"""The one place that talks to the LLM provider.

Everything else calls complete_json(), so swapping Groq for another provider
means rewriting this file only.
"""

import json

from groq import APIError, Groq

from app.config import (
    DAILY_TOKEN_BUDGET,
    GROQ_API_KEY,
    LLM_MAX_RETRIES,
    LLM_MODEL,
    LLM_TIMEOUT_SECONDS,
)
from app.metrics import tokens_today

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
        response = _get_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,  # factual extraction, not creativity
        )
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
