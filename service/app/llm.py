"""The one place that talks to the LLM provider.

Everything else calls complete_json(), so swapping Groq for another provider
means rewriting this file only.
"""

import json

from groq import Groq

from app.config import GROQ_API_KEY, LLM_MAX_RETRIES, LLM_MODEL, LLM_TIMEOUT_SECONDS

_client = None


class LLMUnavailable(RuntimeError):
    """No API key configured."""


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
    response = _get_client().chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,  # factual extraction, not creativity
    )
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }
    try:
        parsed = json.loads(response.choices[0].message.content or "")
    except json.JSONDecodeError:
        parsed = None
    return (parsed if isinstance(parsed, dict) else None), usage
