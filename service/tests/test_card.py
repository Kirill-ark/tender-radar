"""Card guardrails: schema check, review flags, and the code-computed fit score."""

import pytest

from app import card as card_module
from app.card import Card, fit_score, generate_card, review_reasons

GOOD_REPLY = {
    "what": "Interior doors for a school",
    "location": "Bückeburg, Germany",
    "deadline": "2026-11-02",
    "estimated_value": None,
    "requirements": ["Three references"],
    "trade_match": "core",
    "size_match": "unknown",
    "fit_reason": "Interior doors are a core trade; no value stated.",
    "confidence": "high",
    "sources": [1, 2],
}
CHUNKS = [
    {"text": "a", "file": "t.pdf", "page": 1},
    {"text": "b", "file": "t.pdf", "page": 3},
]


@pytest.fixture
def fake_card_inputs(monkeypatch):
    def use(reply):
        monkeypatch.setattr(card_module, "retrieve_card_chunks", lambda tender_id: CHUNKS)
        monkeypatch.setattr(card_module, "complete_json", lambda system, user: (reply, {}))

    return use


@pytest.mark.parametrize(
    "trade, size, expected",
    [("core", "fits", 10), ("core", "unknown", 9), ("adjacent", "unknown", 6), ("none", "too_large", 0)],
)
def test_fit_score_is_computed_from_the_classes(trade, size, expected):
    card = Card.model_validate({**GOOD_REPLY, "trade_match": trade, "size_match": size})
    assert fit_score(card) == expected


def test_confident_card_needs_no_review(fake_card_inputs):
    fake_card_inputs(GOOD_REPLY)
    result = generate_card("1-2026")
    assert result["needs_review"] is False
    assert result["card"]["fit_score"] == 9
    assert result["citations"] == [{"file": "t.pdf", "page": 1}, {"file": "t.pdf", "page": 3}]


def test_reply_off_schema_is_flagged_not_sent_as_confident(fake_card_inputs):
    fake_card_inputs({"what": "doors", "fit_score": 11})
    result = generate_card("1-2026")
    assert result["card"] is None
    assert result["needs_review"] is True


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"deadline": None}, "no deadline found"),
        ({"deadline": "next Friday"}, "deadline is not a date"),
        ({"confidence": "low"}, "model reported low confidence"),
        ({"sources": []}, "no sources cited"),
        ({"sources": [9]}, "cites excerpts that do not exist"),
    ],
)
def test_review_reasons(change, reason):
    card = Card.model_validate({**GOOD_REPLY, **change})
    assert reason in review_reasons(card, chunk_count=len(CHUNKS))
