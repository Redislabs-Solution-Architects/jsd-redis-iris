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

A knowledge base can hold entries for several subjects (for example "finance" and "insurance"),
each entry naming its own in attributes.subject_id. Every entry is loaded with that attribute,
and its entry key starts with a short prefix derived from the subject, so two subjects can hold
the same question. The application looks up only the subject it is configured for (SUBJECT_ID)
and pruning only touches the subjects named in the file. An entry without a subject_id is
loaded unlabelled and is never found by a subject lookup.
"""

import hashlib
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
SUBJECT_ID_ATTRIBUTE = "subject_id"  # the FAQ attribute that says which subject an entry belongs to


def key_prefix(subject_id: str) -> str:
    """Start of every entry key of `subject_id` ("" without a subject).

    Entry keys may only hold lowercase letters, digits and single hyphens, so a subject id cannot
    be told apart from the question that follows it (subject "iris" would also match the keys of
    subject "iris-support"). A fixed-length hash of the id has no such ambiguity, which keeps one
    subject's prefix, and so its pruning, from ever matching another's entries. The readable id
    is kept in the FAQ's subject_id attribute."""
    return hashlib.sha256(subject_id.encode("utf-8")).hexdigest()[:8] + "-" if subject_id else ""


def faq_key(prompt: str, taken: set[str], prefix: str = "") -> str:
    """Stable entry key derived from the question text, unique within `taken`, starting with `prefix`."""
    room = _MAX_KEY_LENGTH - len(prefix)
    base = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")[:room].strip("-") or "faq"
    key, n = base, 2
    while prefix + key in taken:
        suffix = f"-{n}"
        key = f"{base[: room - len(suffix)]}{suffix}"
        n += 1
    taken.add(prefix + key)
    return prefix + key


def entry_subject(entry: dict[str, Any]) -> str:
    """The subject a knowledge-base entry belongs to (its attributes.subject_id), or ""."""
    return str((entry.get("attributes") or {}).get(SUBJECT_ID_ATTRIBUTE) or "").strip()


def knowledge_base_to_faqs(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert knowledge-base.json entries to Playbook FAQ documents.

    An entry's attributes (including its subject_id) travel with the FAQ, and its entry key is
    prefixed from its subject_id."""
    taken: set[str] = set()
    faqs = []
    for entry in entries:
        prompt, response = entry["prompt"].strip(), entry["response"].strip()
        attributes = dict(entry.get("attributes") or {})
        subject = entry_subject(entry)
        if subject:
            attributes[SUBJECT_ID_ATTRIBUTE] = subject
        faq: dict[str, Any] = {
            "entryKey": faq_key(prompt, taken, key_prefix(subject)),
            "question": prompt,
            "answer": response,
        }
        if attributes:
            faq["attributes"] = attributes
        faqs.append(faq)
    return faqs


def _is_unreachable(output: str) -> bool:
    """True when the CLI reported the service as unreachable (DNS / connection error)."""
    try:
        return json.loads(output).get("error", {}).get("code") == "unavailable"
    except (ValueError, AttributeError):
        return False


def _cli(
    arguments: list[str], environ: dict[str, str], timeout: float, attempts: int, retry_delay: float
) -> str:
    """Run `iris agent-playbooks ...` and return its stdout; raise RuntimeError if it fails.

    When the service is unreachable (transient DNS or connection errors) the command is retried up
    to `attempts` times, `retry_delay` seconds apart. Any other error (bad key, forbidden, invalid
    FAQ) fails at once."""
    command = [sys.executable, "-m", "redis_iris", "--json", "--no-input", "agent-playbooks", *arguments]
    for attempt in range(1, attempts + 1):
        result = subprocess.run(
            command, env={**os.environ, **environ}, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode == 0:
            return result.stdout
        detail = (result.stderr or result.stdout).strip()
        if attempt < attempts and _is_unreachable(detail):
            logger.warning(
                "Agent Playbooks unreachable (attempt %d/%d); retrying in %.0fs", attempt, attempts, retry_delay
            )
            time.sleep(retry_delay)
            continue
        raise RuntimeError(
            f"Loading the FAQ knowledge base into Agent Playbooks failed after {attempt} attempt(s): {detail}"
        )
    raise RuntimeError("unreachable")  # pragma: no cover


def sync_knowledge_base(
    knowledge_base_file: Path,
    environ: dict[str, str],
    prune: bool = False,
    timeout: float = 300.0,
    attempts: int = 6,
    retry_delay: float = 5.0,
    subject_id: str = "",
) -> None:
    """Reconcile knowledge_base_file with the Playbook; raise RuntimeError if that fails.

    `environ` holds the IRIS_AGENT_PLAYBOOKS_* settings for the CLI. A sync is idempotent, so a
    failure to reach the service is retried (see _cli). Every entry is loaded, each with its own
    subject_id; `subject_id` (SUBJECT_ID) only widens the pruning scope.

    Pruning deletes entries that are no longer in the file. When subjects are in play it is limited
    to the subjects named in the file (and `subject_id`), so another subject's or application's
    entries in a shared Playbook are never touched; it is done here instead of with the CLI's
    --prune, which deletes every remote-only entry in the whole Playbook. With no subjects at all
    the CLI's --prune is used, as before."""
    entries = json.loads(Path(knowledge_base_file).read_text(encoding="utf-8"))
    faqs = knowledge_base_to_faqs(entries)
    subjects = {entry_subject(e) for e in entries} - {""}
    if subject_id.strip():
        subjects.add(subject_id.strip())
    prefixes = {key_prefix(sub) for sub in subjects}
    unlabelled = sum(1 for e in entries if not entry_subject(e))
    if unlabelled and subjects:
        logger.warning(
            "%d knowledge base entr%s no attributes.subject_id: loaded unlabelled, never found by a subject lookup",
            unlabelled, "y has" if unlabelled == 1 else "ies have",
        )

    # With subjects, an empty file only prunes (the CLI rejects syncing an empty folder).
    with tempfile.TemporaryDirectory(prefix="playbook-faqs-") as tmp:
        for faq in faqs:
            (Path(tmp) / f"{faq['entryKey']}.json").write_text(
                json.dumps(faq, indent=2) + chr(10), encoding="utf-8"
            )
        if faqs or not prefixes:
            arguments = ["faqs", "sync", tmp, "--publish", "changed"]
            if prune and not prefixes:
                arguments += ["--prune", "--yes"]
            _cli(arguments, environ, timeout, attempts, retry_delay)

    if prune and prefixes:
        wanted = {faq["entryKey"] for faq in faqs}
        inventory = json.loads(_cli(["faqs", "list"], environ, timeout, attempts, retry_delay))
        stale = [
            item["entryKey"]
            for item in inventory
            if item.get("entryKey", "").startswith(tuple(prefixes)) and item["entryKey"] not in wanted
        ]
        if stale:
            _cli(["faqs", "delete", *stale, "--yes"], environ, timeout, attempts, retry_delay)
            logger.info(
                "Removed %d FAQ(s) of subject(s) %s that are no longer in the file", len(stale), ", ".join(sorted(subjects))
            )
    logger.info(
        "FAQ knowledge base synced to Agent Playbooks (%d entries%s)",
        len(faqs), f", subjects: {', '.join(sorted(subjects))}" if subjects else "",
    )
