"""Request log in SQLite: the source of the README metrics and the daily token budget."""

import sqlite3
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from app.config import METRICS_DB

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    ts TEXT NOT NULL,              -- UTC, ISO 8601
    endpoint TEXT NOT NULL,        -- ingest | card | ask
    tender_id TEXT,
    status INTEGER NOT NULL,       -- HTTP status the endpoint answered
    duration_ms INTEGER NOT NULL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0
)
"""


def _connect() -> sqlite3.Connection:
    Path(METRICS_DB).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(METRICS_DB)
    connection.execute(SCHEMA)
    return connection


def record(
    endpoint: str,
    tender_id: str | None,
    status: int,
    started: float,
    usage: dict | None = None,
) -> None:
    """Log one request; `started` is the time.monotonic() taken when it began."""
    usage = usage or {}
    with closing(_connect()) as connection, connection:
        connection.execute(
            "INSERT INTO requests VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                endpoint,
                tender_id,
                status,
                round((time.monotonic() - started) * 1000),
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
            ),
        )


def tokens_today() -> int:
    """Prompt + completion tokens spent since UTC midnight."""
    today = datetime.now(timezone.utc).date().isoformat()
    with closing(_connect()) as connection:
        (total,) = connection.execute(
            "SELECT COALESCE(SUM(prompt_tokens + completion_tokens), 0)"
            " FROM requests WHERE ts >= ?",
            (today,),
        ).fetchone()
    return total


def summary() -> dict:
    """Aggregates for the README: per endpoint, per tender, failure rate, budget."""
    with closing(_connect()) as connection:
        endpoints = {
            endpoint: {
                "requests": count,
                "failed": failed,
                "failure_rate": round(failed / count, 3),
                "avg_duration_ms": round(avg_ms),
                "avg_tokens": round(avg_tokens),
            }
            for endpoint, count, failed, avg_ms, avg_tokens in connection.execute(
                "SELECT endpoint, COUNT(*), SUM(status >= 400), AVG(duration_ms),"
                " AVG(prompt_tokens + completion_tokens)"
                " FROM requests GROUP BY endpoint"
            )
        }
        # End to end per tender = its successful ingest + card requests.
        per_tender = connection.execute(
            "SELECT COUNT(*), AVG(ms), AVG(tokens) FROM ("
            "  SELECT SUM(duration_ms) AS ms, SUM(prompt_tokens + completion_tokens) AS tokens"
            "  FROM requests WHERE endpoint IN ('ingest', 'card') AND status < 400"
            "  GROUP BY tender_id HAVING COUNT(DISTINCT endpoint) = 2)"
        ).fetchone()
    tenders, avg_ms, avg_tokens = per_tender
    return {
        "endpoints": endpoints,
        "per_tender": {
            "tenders": tenders,
            "avg_duration_ms": round(avg_ms or 0),
            "avg_tokens": round(avg_tokens or 0),
        },
        "tokens_today": tokens_today(),
    }
