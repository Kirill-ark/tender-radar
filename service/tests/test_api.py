"""API input validation and the daily token budget."""

import pytest
from fastapi.testclient import TestClient

from app import llm
from app.main import app

client = TestClient(app)


@pytest.mark.parametrize("tender_id", ["abc", "674261-26", "../../etc", ""])
def test_malformed_tender_id_is_rejected(tender_id):
    response = client.post("/card", json={"tender_id": tender_id})
    assert response.status_code == 422


def test_overlong_question_is_rejected():
    response = client.post("/ask", json={"tender_id": "674261-2026", "question": "x" * 501})
    assert response.status_code == 422


def test_llm_is_not_called_once_the_daily_budget_is_spent(monkeypatch):
    monkeypatch.setattr(llm, "tokens_today", lambda: llm.DAILY_TOKEN_BUDGET)
    monkeypatch.setattr(llm, "_get_client", lambda: pytest.fail("LLM called despite the budget"))
    with pytest.raises(llm.LLMUnavailable, match="budget"):
        llm.complete_json("system", "user")
