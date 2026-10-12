"""Export the project's n8n workflows into workflows/*.json for the repository.

    python3 scripts/export_workflows.py

Reads them from the running n8n container and makes them safe to publish:
the personal Telegram chat id becomes a placeholder, and run-time noise
(pinned test data, version ids, timestamps) is dropped so diffs stay readable.
Credentials are never in an export: nodes only name the credential to use.
"""

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ["fetch-tenders", "process-tender", "answer-question", "error-handler"]
CHAT_ID_PLACEHOLDER = "YOUR_TELEGRAM_CHAT_ID"
DROPPED_KEYS = ["pinData", "versionId", "activeVersionId", "createdAt", "updatedAt", "shared"]


def export_all() -> list[dict]:
    command = (
        "n8n export:workflow --all --output=/tmp/workflows.json >/dev/null 2>&1"
        " && cat /tmp/workflows.json"
    )
    output = subprocess.run(
        ["docker", "compose", "exec", "-T", "n8n", "sh", "-c", command],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout
    return json.loads(output)


def sanitize(workflow: dict) -> dict:
    for key in DROPPED_KEYS:
        workflow.pop(key, None)
    for node in workflow["nodes"]:
        chat_id = node.get("parameters", {}).get("chatId")
        # A literal numeric id is personal; expressions ("={{ ... }}") stay as they are.
        if isinstance(chat_id, str) and re.fullmatch(r"-?\d+", chat_id):
            node["parameters"]["chatId"] = CHAT_ID_PLACEHOLDER
    return workflow


def main() -> None:
    by_name = {workflow["name"]: workflow for workflow in export_all()}
    out_dir = ROOT / "workflows"
    out_dir.mkdir(exist_ok=True)
    for name in WORKFLOWS:
        workflow = sanitize(by_name[name])
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(workflow, indent=2, ensure_ascii=False) + "\n")
        print(f"{path.relative_to(ROOT)}: {len(workflow['nodes'])} nodes")


if __name__ == "__main__":
    main()
