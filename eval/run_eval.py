"""Evaluate retrieval and answers on hand-written questions with known answers.

Runs inside the service container (it imports the service code):
    docker compose exec service python -m eval.run_eval

For each answerable question:
- retrieval hit@1 / hit@3: is the page holding the answer among the pages of
  the top 1 / top 3 retrieved chunks;
- answer correct: does the answer contain one of the accepted strings.
For each question whose answer is not in the document: the guardrail must
reply "not found" instead of guessing.
Results are written to eval/results.json so the README numbers are reproducible.
"""

import json
import time
from pathlib import Path

from app.ask import answer_question, retrieve

HERE = Path(__file__).parent
PAUSE_SECONDS = 8  # stay under the free tier's tokens-per-minute limit


def normalize(text: str) -> str:
    """Lowercase, and turn the model's narrow / no-break spaces into plain ones."""
    return text.replace("\u202f", " ").replace("\u00a0", " ").lower()


def main() -> None:
    questions = json.loads((HERE / "questions.json").read_text())
    rows = []
    for q in questions:
        result = answer_question(q["tender_id"], q["question"])
        row = {"question": q["question"], "tender_id": q["tender_id"], "answer": result["answer"]}
        if q.get("not_in_document"):
            row["correct"] = not result["found"]
        else:
            pages = [chunk["page"] for chunk in retrieve(q["tender_id"], q["question"])]
            row["hit@1"] = pages[0] in q["pages"]
            row["hit@3"] = any(page in q["pages"] for page in pages[:3])
            row["correct"] = result["found"] and any(
                normalize(accepted) in normalize(result["answer"]) for accepted in q["answer"]
            )
        rows.append(row)
        print(f"{'OK ' if row['correct'] else 'BAD'} {q['question']} -> {result['answer']}")
        time.sleep(PAUSE_SECONDS)

    answerable = [r for r in rows if "hit@1" in r]
    unanswerable = [r for r in rows if "hit@1" not in r]
    summary = {
        "answerable": len(answerable),
        "hit@1": sum(r["hit@1"] for r in answerable),
        "hit@3": sum(r["hit@3"] for r in answerable),
        "answers_correct": sum(r["correct"] for r in answerable),
        "not_in_document": len(unanswerable),
        "refused_correctly": sum(r["correct"] for r in unanswerable),
    }
    print(json.dumps(summary, indent=2))
    (HERE / "results.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
