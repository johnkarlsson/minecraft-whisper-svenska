"""Minecraft Q&A pipeline orchestrator.

Filters finalized transcript lines for Minecraft questions (Haiku),
generates answers (Opus with web search), displays them, and speaks
answers aloud via macOS TTS while pausing the recorder.

All stages use priority queues ordered by sentence count (fewer = higher
priority) and question-mark count (more = higher priority).
"""

from __future__ import annotations

import itertools
import logging
import queue as _queue
import re
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass

from whisper_stt.claude_sdk import (
    filter_minecraft_question,
    generate_answer,
    summarize_question,
)
from whisper_stt.history import save_history

logger = logging.getLogger(__name__)

_SENTENCE_END_RE = re.compile(r"[.!?…]+")


def compute_filter_priority(text: str) -> tuple[int, int]:
    """Compute priority for pipeline queues. Lower tuple = higher priority.

    Fewer sentences and more question marks give higher priority.
    """
    parts = _SENTENCE_END_RE.split(text.strip())
    sentence_count = sum(1 for p in parts if p.strip())
    question_marks = text.count("?")
    return (sentence_count, -question_marks)


@dataclass
class QAPair:
    question: str
    headline: str
    answer: str
    source_texts: list[str] | None = None


class MinecraftPipeline:
    """Orchestrates Minecraft question detection, answering, and TTS."""

    def __init__(
        self,
        display: object,
        pause_recorder: Callable[[], None],
        resume_recorder: Callable[[], None],
        *,
        speak: bool = False,
    ) -> None:
        # Display must have mark_line_green, add_qa_pair, set_pipeline_status
        self._display = display
        self._pause_recorder = pause_recorder
        self._resume_recorder = resume_recorder
        self._speak_enabled = speak

        # State (protected by _lock)
        self._lock = threading.Lock()
        self._passed_lines: list[str] = []
        self._all_lines: list[dict[str, str]] = []
        self._qa_pairs: list[QAPair] = []
        self._opus_session_id: str | None = None
        self._shutdown_flag = False

        # Filter priority queue: (priority, seq, text, line_index)
        self._filter_queue: _queue.PriorityQueue[
            tuple[tuple[int, int], int, str, int]
        ] = _queue.PriorityQueue()
        self._filter_seq = itertools.count()
        self._filter_threads: list[threading.Thread] = []
        for i in range(2):
            t = threading.Thread(
                target=self._filter_worker, daemon=True, name=f"filter-{i}"
            )
            t.start()
            self._filter_threads.append(t)

        # Answer priority queue: (priority, seq, text, line_index)
        self._answer_queue: _queue.PriorityQueue[
            tuple[tuple[int, int], int, str, int]
        ] = _queue.PriorityQueue()
        self._answer_seq = itertools.count()

        # Signal for answer thread
        self._new_question_event = threading.Event()

        # Answer thread: single daemon thread for serial Opus calls + TTS
        self._answer_thread = threading.Thread(
            target=self._answer_loop, daemon=True, name="minecraft-answer"
        )
        self._answer_thread.start()

    def submit(self, text: str, line_index: int) -> None:
        """Non-blocking: submit finalized text for Minecraft filtering."""
        priority = compute_filter_priority(text)
        self._filter_queue.put((priority, next(self._filter_seq), text, line_index))

    def force_submit(self, text: str, line_index: int) -> None:
        """Non-blocking: submit text directly, bypassing the filter."""
        threading.Thread(
            target=self._force_enqueue,
            args=(text, line_index),
            daemon=True,
            name="force-enqueue",
        ).start()

    def _filter_worker(self) -> None:
        """Pull items from the filter priority queue and run the filter."""
        while not self._shutdown_flag:
            try:
                _priority, _seq, text, line_index = self._filter_queue.get(
                    timeout=1.0
                )
            except _queue.Empty:
                continue
            try:
                self._filter_and_enqueue(text, line_index)
            finally:
                self._filter_queue.task_done()

    def _filter_and_enqueue(self, text: str, line_index: int) -> None:
        """Run Haiku filter; if Minecraft question, enqueue for Opus."""
        try:
            self._set_status("Filtrerar...", "yellow")
            is_minecraft = filter_minecraft_question(text)
            if not is_minecraft:
                self._display.mark_line_grey(line_index)  # type: ignore[attr-defined]
                with self._lock:
                    self._all_lines.append({"text": text, "kind": "filtered"})
                    save_history(self._all_lines, self._qa_pairs)
                self._set_status("", "dim")
                return

            # Mark the transcript line green
            self._display.mark_line_green(line_index)  # type: ignore[attr-defined]

            with self._lock:
                self._passed_lines.append(text)
                self._all_lines.append({"text": text, "kind": "passed"})
                save_history(self._all_lines, self._qa_pairs)

            # Generate headline
            headline = summarize_question(text)

            # Enqueue in answer priority queue
            priority = compute_filter_priority(text)
            self._answer_queue.put(
                (priority, next(self._answer_seq), text, line_index)
            )

            self._set_status(f"Minecraft-fråga: {headline}", "green")
            self._new_question_event.set()

        except Exception:
            logger.exception("Filter/enqueue failed for: %s", text[:60])
            self._set_status("Filterfel", "red")

    def _force_enqueue(self, text: str, line_index: int) -> None:
        """Enqueue for Opus without filtering."""
        try:
            self._display.mark_line_green(line_index)  # type: ignore[attr-defined]

            with self._lock:
                self._passed_lines.append(text)
                self._all_lines.append({"text": text, "kind": "passed"})
                save_history(self._all_lines, self._qa_pairs)

            headline = summarize_question(text)

            priority = compute_filter_priority(text)
            self._answer_queue.put(
                (priority, next(self._answer_seq), text, line_index)
            )

            self._set_status(f"Minecraft-fråga: {headline}", "green")
            self._new_question_event.set()
        except Exception:
            logger.exception("Force enqueue failed for: %s", text[:60])
            self._set_status("Fel", "red")

    def _answer_loop(self) -> None:
        """Wait for new questions, batch them in priority order, call Opus."""
        while not self._shutdown_flag:
            self._new_question_event.wait(timeout=1.0)
            if self._shutdown_flag:
                break
            self._new_question_event.clear()

            # Drain queue in priority order
            items: list[tuple[str, int]] = []
            while True:
                try:
                    _priority, _seq, text, line_index = (
                        self._answer_queue.get_nowait()
                    )
                    items.append((text, line_index))
                except _queue.Empty:
                    break

            if not items:
                continue

            new_questions = [t for t, _ in items]
            line_indices = [li for _, li in items]

            try:
                self._set_status("Genererar svar...", "yellow")
                self._show_answering()
                answer_text, session_id = generate_answer(
                    new_questions, self._opus_session_id
                )
                self._opus_session_id = session_id

                if not answer_text:
                    self._set_status("Tomt svar", "red")
                    continue

                # Build headline from first question in batch
                headline = summarize_question(new_questions[0])
                combined_q = " / ".join(new_questions)
                pair = QAPair(
                    question=combined_q,
                    headline=headline,
                    answer=answer_text,
                    source_texts=new_questions,
                )
                with self._lock:
                    self._qa_pairs.append(pair)
                    save_history(self._all_lines, self._qa_pairs)

                self._display.add_qa_pair(pair, line_indices)  # type: ignore[attr-defined]
                if self._speak_enabled:
                    self._set_status("Läser svar...", "cyan")
                    self._speak(answer_text)
                self._set_status("", "dim")

            except Exception:
                logger.exception("Answer generation failed")
                self._set_status("Svarsfel", "red")

    def _speak(self, answer: str) -> None:
        """Speak answer via macOS TTS, pausing the recorder meanwhile."""
        try:
            self._pause_recorder()
            subprocess.run(
                ["say", "-v", "Alva", answer],
                check=False,
                timeout=120,
            )
        except Exception:
            logger.exception("TTS failed")
        finally:
            self._resume_recorder()

    def _show_answering(self) -> None:
        try:
            self._display.show_answering()  # type: ignore[attr-defined]
        except Exception:
            pass

    def _set_status(self, msg: str, style: str) -> None:
        try:
            self._display.set_pipeline_status(msg, style)  # type: ignore[attr-defined]
        except Exception:
            pass

    def remove_qa_pair(self, pair: QAPair) -> None:
        """Remove a QA pair and its source texts from state, then save."""
        with self._lock:
            try:
                self._qa_pairs.remove(pair)
            except ValueError:
                pass
            if pair.source_texts:
                for st in pair.source_texts:
                    try:
                        self._passed_lines.remove(st)
                    except ValueError:
                        pass
                    # Also remove from ordered list
                    for i, entry in enumerate(self._all_lines):
                        if entry["text"] == st and entry["kind"] == "passed":
                            self._all_lines.pop(i)
                            break
            save_history(self._all_lines, self._qa_pairs)

    def restore_history(
        self,
        all_lines: list[dict[str, str]],
        qa_pairs: list[QAPair],
    ) -> None:
        """Initialize pipeline state from previously saved history."""
        with self._lock:
            self._all_lines = list(all_lines)
            self._passed_lines = [
                e["text"] for e in all_lines if e["kind"] == "passed"
            ]
            self._qa_pairs = list(qa_pairs)

    def shutdown(self) -> None:
        """Stop threads."""
        self._shutdown_flag = True
        self._new_question_event.set()  # Wake answer thread
        self._answer_thread.join(timeout=5.0)
