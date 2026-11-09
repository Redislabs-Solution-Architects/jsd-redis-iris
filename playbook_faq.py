#!/usr/bin/env python3
"""
Load the FAQ knowledge base (knowledge-base.json) into Redis Iris Agent Playbooks.

knowledge-base.json is the source of truth. On every startup its prompt/response
pairs are converted to Playbook FAQ files in a temporary directory and reconciled
with `iris agent-playbooks faqs sync --publish changed`, so:

- new or edited entries are created/versioned and published,
- unchanged entries are left alone (the sync is idempotent and cheap),
- entries removed from the file are only removed from the Playbook when
  PLAYBOOK_FAQ_PRUNE=true (deletion is irreversible, and the Playbook may hold
  FAQs that were not authored here).

The credentials used here must be allowed to read, write and publish FAQs.
"""

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MAX_KEY_LENGTH = 64


def faq_key(prompt: str, taken: set[str]) -> str:
    """Stable entry key derived from the question text, unique within `taken`."""
    base = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")[:_MAX_KEY_LENGTH].strip("-") or "faq"
    key, n = base, 2
    while key in taken:
        suffix = f"-{n}"
        key = f"{base[: _MAX_KEY_LENGTH - len(suffix)]}{suffix}"
        n += 1
    taken.add(key)
    return key


def knowledge_base_to_faqs(entries: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Convert knowledge-base.json entries to Playbook FAQ documents."""
    taken: set[str] = set()
    faqs = []
    for entry in entries:
        prompt, response = entry["prompt"].strip(), entry["response"].strip()
        faqs.append({"entryKey": faq_key(prompt, taken), "question": prompt, "answer": response})
    return faqs


def _is_unreachable(output: str) -> bool:
    """True when the CLI reported the service as unreachable (DNS / connection error)."""
    try:
        return json.loads(output).get("error", {}).get("code") == "unavailable"
    except (ValueError, AttributeError):
        return False


def sync_knowledge_base(
    knowledge_base_file: Path,
    environ: dict[str, str],
    prune: bool = False,
    timeout: float = 300.0,
    attempts: int = 6,
    retry_delay: float = 5.0,
) -> None:
    """Reconcile knowledge_base_file with the Playbook; raise RuntimeError if that fails.

    `environ` holds the IRIS_AGENT_PLAYBOOKS_* settings for the CLI. A sync is idempotent, so when
    the service is unreachable (transient DNS or connection errors) it is retried up to `attempts`
    times, `retry_delay` seconds apart. Any other error (bad key, forbidden, invalid FAQ) fails at once.
    """
    entries = json.loads(Path(knowledge_base_file).read_text(encoding="utf-8"))
    faqs = knowledge_base_to_faqs(entries)

    with tempfile.TemporaryDirectory(prefix="playbook-faqs-") as tmp:
        for faq in faqs:
            (Path(tmp) / f"{faq['entryKey']}.json").write_text(
                json.dumps(faq, indent=2) + "\n", encoding="utf-8"
            )

        command = [
            sys.executable, "-m", "redis_iris", "--json", "--no-input",
            "agent-playbooks", "faqs", "sync", tmp, "--publish", "changed",
        ]
        if prune:
            command += ["--prune", "--yes"]

        for attempt in range(1, attempts + 1):
            result = subprocess.run(
                command,
                env={**os.environ, **environ},
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0:
                break
            detail = (result.stderr or result.stdout).strip()
            if attempt < attempts and _is_unreachable(detail):
                logger.warning(
                    "Agent Playbooks unreachable (attempt %d/%d); retrying in %.0fs",
                    attempt, attempts, retry_delay,
                )
                time.sleep(retry_delay)
                continue
            raise RuntimeError(
                f"Loading the FAQ knowledge base into Agent Playbooks failed after "
                f"{attempt} attempt(s): {detail}"
            )
    logger.info("FAQ knowledge base synced to Agent Playbooks (%d entries)", len(faqs))
