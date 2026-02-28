"""Persistence for passed transcription lines and QA pairs."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from whisper_stt.pipeline import QAPair

logger = logging.getLogger(__name__)

_HISTORY_FILE = Path.home() / ".config" / "whisper-stt" / "history.json"


def load_history() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Load saved history. Returns (all_lines, qa_dicts).

    Each entry in all_lines is {"text": ..., "kind": "passed"|"filtered"}.
    Backward-compatible with old format that had separate lists.
    """
    try:
        data = json.loads(_HISTORY_FILE.read_text())
        all_lines: list[dict[str, str]] | None = data.get("all_lines")
        if all_lines is None:
            # Backward compat: reconstruct from old separate lists
            all_lines = [
                {"text": t, "kind": "filtered"}
                for t in data.get("filtered_lines", [])
            ] + [
                {"text": t, "kind": "passed"}
                for t in data.get("passed_lines", [])
            ]
        return all_lines, data.get("qa_pairs", [])
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return [], []


def save_history(
    all_lines: list[dict[str, str]],
    qa_pairs: list[QAPair],
) -> None:
    """Persist all lines and QA pairs to disk."""
    data = {
        "all_lines": all_lines,
        "passed_lines": [e["text"] for e in all_lines if e["kind"] == "passed"],
        "qa_pairs": [asdict(p) for p in qa_pairs],
    }
    try:
        _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        _HISTORY_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception:
        logger.exception("Failed to save history")
