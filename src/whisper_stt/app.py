from __future__ import annotations

import argparse
import itertools
import logging
import queue as _queue
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.events import Click
from textual.widgets import Footer, Header, LoadingIndicator, Markdown, Static
from textual import work

if TYPE_CHECKING:
    from textual.theme import Theme
    from whisper_stt.config import STTConfig
    from whisper_stt.pipeline import QAPair

logger = logging.getLogger(__name__)

_THEME_FILE = Path.home() / ".config" / "whisper-stt" / "theme"


class TranscriptionLog(VerticalScroll):
    """Transcription scroll area that tracks last scroll time for auto-scroll."""

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        app = self.app
        if isinstance(app, WhisperApp):
            app._transcription_last_scroll_t = time.monotonic()


class QALog(VerticalScroll):
    """QA scroll area that dismisses the new-answers notice on scroll-to-bottom."""

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        app = self.app
        if isinstance(app, WhisperApp):
            app._qa_last_scroll_t = time.monotonic()
            if app._qa_has_unseen and self.is_vertical_scroll_end:
                app._dismiss_qa_notice()


class WhisperApp(App[None]):
    CSS_PATH = "app.tcss"
    TITLE = "Whisper STT"
    BINDINGS = [("q", "quit", "Quit"), ("ctrl+c", "quit", "Quit")]

    def __init__(self, config: STTConfig, args: argparse.Namespace) -> None:
        super().__init__()
        self._config = config
        self._args = args
        self._line_counter = 0
        self._interim_line_id: int | None = None
        self._line_texts: dict[int, str] = {}
        self._qa_has_unseen = False
        self._last_interim_text = ""
        self._transcriptions_in_flight = 0
        self._transcription_queue: _queue.PriorityQueue[
            tuple[tuple[int, int], int, object, int | None]
        ] = _queue.PriorityQueue()
        self._transcription_seq = itertools.count()
        self._qa_counter = 0
        self._qa_line_ids: dict[str, list[int]] = {}
        self._qa_pairs_by_widget: dict[str, QAPair] = {}
        self._loaded_all_lines: list[dict[str, str]] = []
        self._loaded_qa_pairs: list[QAPair] = []
        self._old_filtered_ids: list[int] = []
        self._old_visible = False
        self._transcription_last_scroll_t = time.monotonic()
        self._qa_last_scroll_t = time.monotonic()

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-container"):
            with Vertical(id="transcription-panel") as v:
                v.border_title = "Transkription"
                yield Static("Visa gammalt skräp", id="old-toggle-btn")
                yield TranscriptionLog(id="transcription-log")
                yield Static("Initializing...", id="transcription-status")
            with Vertical(id="qa-panel") as v:
                v.border_title = "Minecraft Q&A"
                yield QALog(id="qa-log")
                yield Static(
                    "Nya svar finns längre ner", id="qa-new-notice"
                )
                yield Static("", id="pipeline-status")
        yield Footer()

    def on_mount(self) -> None:
        try:
            saved = _THEME_FILE.read_text().strip()
            if saved:
                self.theme = saved
        except FileNotFoundError:
            pass
        self.theme_changed_signal.subscribe(self, self._on_theme_changed)
        self._load_history()
        self._run_audio_loop()
        self.set_interval(10, self._check_auto_scroll)

    def _load_history(self) -> None:
        from whisper_stt.history import load_history
        from whisper_stt.pipeline import QAPair

        all_lines, qa_dicts = load_history()

        # Restore old lines in transcription panel in original order
        container = self.query_one("#transcription-log", VerticalScroll)
        text_to_line_ids: dict[str, list[int]] = {}
        for entry in all_lines:
            text = entry["text"]
            kind = entry["kind"]
            self._line_counter += 1
            line_id = self._line_counter
            if kind == "filtered":
                container.mount(
                    Static(text, id=f"line-{line_id}", classes="line-filtered line-old hidden")
                )
                self._old_filtered_ids.append(line_id)
            else:
                container.mount(
                    Static(text, id=f"line-{line_id}", classes="line-passed")
                )
                text_to_line_ids.setdefault(text, []).append(line_id)
            self._line_texts[line_id] = text
        if all_lines:
            container.scroll_end(animate=False)

        # Show toggle button if there are old filtered lines
        if self._old_filtered_ids:
            try:
                self.query_one("#old-toggle-btn").add_class("visible")
            except NoMatches:
                pass

        # Restore QA pairs in QA panel
        qa_container = self.query_one("#qa-log", QALog)
        loaded_qa_pairs: list[QAPair] = []
        for d in qa_dicts:
            raw_source = d.get("source_texts")
            source_texts: list[str] = (
                list(raw_source) if isinstance(raw_source, list) else [d["question"]]
            )
            pair = QAPair(
                question=d["question"],
                headline=d["headline"],
                answer=d["answer"],
                source_texts=source_texts,
            )
            loaded_qa_pairs.append(pair)

            # Resolve line IDs for this pair's source texts
            line_ids: list[int] = []
            for st in source_texts:
                ids = text_to_line_ids.get(st, [])
                if ids:
                    line_ids.append(ids.pop(0))

            self._qa_counter += 1
            qa_id = f"qa-{self._qa_counter}"
            md_content = f"### Q: {pair.headline}\n\n{pair.answer}"
            wrapper = Vertical(
                Markdown(md_content),
                Static("X", classes="qa-delete-btn"),
                id=qa_id,
                classes="qa-entry",
            )
            qa_container.mount(wrapper)
            self._qa_line_ids[qa_id] = line_ids
            self._qa_pairs_by_widget[qa_id] = pair
        if qa_dicts:
            qa_container.scroll_end(animate=False)

        self._loaded_all_lines = all_lines
        self._loaded_qa_pairs = loaded_qa_pairs

    def _on_theme_changed(self, theme: Theme) -> None:
        _THEME_FILE.parent.mkdir(parents=True, exist_ok=True)
        _THEME_FILE.write_text(theme.name)

    def on_click(self, event: Click) -> None:
        widget = event.widget
        if not isinstance(widget, Static):
            return

        # Handle old-lines toggle button
        if widget.id == "old-toggle-btn":
            self._toggle_old_lines()
            return

        # Handle QA delete button
        if widget.has_class("qa-delete-btn"):
            self._handle_qa_delete(widget)
            return

        # Handle clicking filtered lines to resubmit
        if not widget.has_class("line-filtered"):
            return
        widget_id = widget.id
        if not widget_id or not widget_id.startswith("line-"):
            return
        line_index = int(widget_id.removeprefix("line-"))
        text = self._line_texts.get(line_index)
        if not text:
            return
        # Only trigger if clicking on actual text, not empty space
        if event.x >= len(text):
            return
        widget.set_classes("line-passed")
        from whisper_stt.pipeline import MinecraftPipeline

        pipeline = self._pipeline
        if isinstance(pipeline, MinecraftPipeline):
            pipeline.force_submit(text, line_index)

    def _handle_qa_delete(self, btn: Static) -> None:
        """Remove a QA entry, grey out its transcription lines, update history."""
        from textual.widget import Widget

        parent = btn.parent
        if not isinstance(parent, Widget):
            return
        qa_id = parent.id
        if qa_id is None:
            return

        # Grey out associated transcription lines
        line_ids = self._qa_line_ids.pop(qa_id, [])
        for lid in line_ids:
            try:
                self.query_one(f"#line-{lid}", Static).set_classes("line-deleted")
            except NoMatches:
                pass

        # Remove from pipeline state
        pair = self._qa_pairs_by_widget.pop(qa_id, None)
        if pair is not None:
            from whisper_stt.pipeline import MinecraftPipeline

            pipeline = self._pipeline
            if isinstance(pipeline, MinecraftPipeline):
                pipeline.remove_qa_pair(pair)

        # Remove the widget
        parent.remove()

    def _toggle_old_lines(self) -> None:
        """Toggle visibility of old filtered (grey) lines."""
        self._old_visible = not self._old_visible
        for lid in self._old_filtered_ids:
            try:
                w = self.query_one(f"#line-{lid}", Static)
                if self._old_visible:
                    w.remove_class("hidden")
                else:
                    w.add_class("hidden")
            except NoMatches:
                pass
        try:
            btn = self.query_one("#old-toggle-btn", Static)
            btn.update("Dölj gammalt skräp" if self._old_visible else "Visa gammalt skräp")
        except NoMatches:
            pass

    def _check_auto_scroll(self) -> None:
        """Auto-scroll to bottom if user hasn't scrolled in 60 seconds."""
        now = time.monotonic()
        t_log = self.query_one("#transcription-log", VerticalScroll)
        if not t_log.is_vertical_scroll_end and now - self._transcription_last_scroll_t >= 60:
            t_log.scroll_end(animate=False)
        qa_log = self.query_one("#qa-log", QALog)
        if not qa_log.is_vertical_scroll_end and now - self._qa_last_scroll_t >= 60:
            qa_log.scroll_end(animate=False)
            self._dismiss_qa_notice()

    @work(thread=True)
    def _run_audio_loop(self) -> None:
        config = self._config
        args = self._args

        self.call_from_thread(self.set_status, "Loading models... (this may take a moment)", "yellow")

        from faster_whisper import WhisperModel  # type: ignore[import-untyped]
        from RealtimeSTT import AudioToTextRecorder  # type: ignore[import-untyped]

        from whisper_stt.pipeline import MinecraftPipeline, compute_filter_priority

        self.call_from_thread(self.set_status, "Loading main model...", "yellow")
        whisper_model = WhisperModel(
            config.main_model,
            device=config.device,
            compute_type=config.compute_type,
        )

        pipeline = MinecraftPipeline(
            self, self._pause_recorder, self._resume_recorder, speak=args.speak
        )
        self._pipeline = pipeline
        pipeline.restore_history(self._loaded_all_lines, self._loaded_qa_pairs)

        def transcription_worker() -> None:
            import numpy as np

            while not self._shutting_down:
                try:
                    item = self._transcription_queue.get(timeout=1.0)
                except _queue.Empty:
                    continue
                _priority, _seq, audio, line_id = item
                try:
                    audio_array = np.asarray(audio, dtype=np.float32)
                    segments, _ = whisper_model.transcribe(
                        audio_array,
                        language=config.language,
                        initial_prompt=config.initial_prompt,
                    )
                    text = " ".join(s.text for s in segments).strip()
                    line_index: int = self.call_from_thread(
                        self.finalize_line, text, line_id
                    )
                    if text:
                        pipeline.submit(text, line_index)
                except Exception:
                    logger.exception("Transcription failed")
                finally:
                    self._transcription_queue.task_done()

        transcription_thread = threading.Thread(
            target=transcription_worker, daemon=True, name="transcription"
        )
        transcription_thread.start()

        def on_transcription_start(audio: object) -> bool:
            # Read interim text BEFORE snapshot_pending clears it
            interim_text = self._last_interim_text
            line_id: int | None = self.call_from_thread(self.snapshot_pending)
            self.call_from_thread(self.set_status, "Transcribing...", "yellow")
            priority = (
                compute_filter_priority(interim_text)
                if interim_text
                else (0, 0)
            )
            self._transcription_queue.put(
                (priority, next(self._transcription_seq), audio, line_id)
            )
            return True

        recorder = AudioToTextRecorder(
            model=config.realtime_model,
            language=config.language,
            device=config.device,
            compute_type=config.compute_type,
            silero_sensitivity=config.silero_sensitivity,
            silero_deactivity_detection=config.silero_deactivity_detection,
            post_speech_silence_duration=config.post_speech_silence_duration,
            min_length_of_recording=config.min_length_of_recording,
            spinner=config.spinner,
            initial_prompt=config.initial_prompt,
            initial_prompt_realtime=config.initial_prompt,
            enable_realtime_transcription=config.enable_realtime,
            realtime_model_type=config.realtime_model if config.enable_realtime else "tiny",
            on_recording_start=lambda: self.call_from_thread(
                self.set_status, "Recording...", "red"
            ),
            on_recording_stop=lambda: self.call_from_thread(
                self.set_status, "Processing...", "yellow"
            ),
            on_transcription_start=on_transcription_start,
            on_realtime_transcription_stabilized=(
                (lambda text: self.call_from_thread(self.update_interim, text))
                if config.enable_realtime
                else None
            ),
        )
        self._recorder = recorder

        self.call_from_thread(self.set_status, "Listening...", "green")

        while not self._shutting_down:
            recorder.text()

    _pipeline: object = None
    _recorder: object = None
    _shutting_down: bool = False

    def _pause_recorder(self) -> None:
        recorder = self._recorder
        if recorder is not None:
            recorder.set_microphone(False)  # type: ignore[union-attr]

    def _resume_recorder(self) -> None:
        recorder = self._recorder
        if recorder is not None:
            recorder.set_microphone(True)  # type: ignore[union-attr]

    # --- Thread-safe display update methods (called via call_from_thread) ---

    def update_interim(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self._last_interim_text = text
        container = self.query_one("#transcription-log", VerticalScroll)
        at_bottom = container.is_vertical_scroll_end
        if self._interim_line_id is not None:
            self.query_one(f"#line-{self._interim_line_id}", Static).update(text)
        else:
            self._line_counter += 1
            self._interim_line_id = self._line_counter
            container.mount(
                Static(text, id=f"line-{self._interim_line_id}", classes="line-interim")
            )
        if at_bottom:
            container.scroll_end(animate=False)

    def snapshot_pending(self) -> int | None:
        line_id = self._interim_line_id
        if line_id is not None:
            try:
                self.query_one(f"#line-{line_id}", Static).set_classes(
                    "line-finalized"
                )
            except NoMatches:
                pass
        self._interim_line_id = None
        self._last_interim_text = ""
        self._transcriptions_in_flight += 1
        return line_id

    def finalize_line(self, text: str, line_id: int | None) -> int:
        text = text.strip()
        if text:
            if line_id is not None:
                try:
                    line = self.query_one(f"#line-{line_id}", Static)
                    line.update(text)
                    line.set_classes("line-transcribed")
                except NoMatches:
                    pass
                result_id = line_id
            else:
                self._line_counter += 1
                result_id = self._line_counter
                container = self.query_one("#transcription-log", VerticalScroll)
                container.mount(
                    Static(text, id=f"line-{result_id}", classes="line-transcribed")
                )
            self._line_texts[result_id] = text
        else:
            if line_id is not None:
                try:
                    self.query_one(f"#line-{line_id}", Static).remove()
                except NoMatches:
                    pass
            result_id = -1

        container = self.query_one("#transcription-log", VerticalScroll)
        if container.is_vertical_scroll_end:
            container.scroll_end(animate=False)

        self._transcriptions_in_flight -= 1
        if self._transcriptions_in_flight <= 0:
            self._transcriptions_in_flight = 0
            self.set_status("Listening...", "green")

        return result_id

    def mark_line_green(self, line_index: int) -> None:
        self.call_from_thread(self._mark_line_green_impl, line_index)

    def _mark_line_green_impl(self, line_index: int) -> None:
        try:
            self.query_one(f"#line-{line_index}", Static).set_classes("line-passed")
        except NoMatches:
            pass

    def mark_line_grey(self, line_index: int) -> None:
        self.call_from_thread(self._mark_line_grey_impl, line_index)

    def _mark_line_grey_impl(self, line_index: int) -> None:
        try:
            self.query_one(f"#line-{line_index}", Static).set_classes("line-filtered")
        except NoMatches:
            pass

    def show_answering(self) -> None:
        self.call_from_thread(self._show_answering_impl)

    def _show_answering_impl(self) -> None:
        container = self.query_one("#qa-log", QALog)
        try:
            self.query_one("#answer-throbber")
        except NoMatches:
            at_bottom = container.is_vertical_scroll_end
            container.mount(LoadingIndicator(id="answer-throbber"))
            if at_bottom:
                container.scroll_end(animate=False)

    def _hide_answering(self) -> None:
        try:
            self.query_one("#answer-throbber").remove()
        except NoMatches:
            pass

    def add_qa_pair(self, pair: QAPair, line_indices: list[int] | None = None) -> None:
        self.call_from_thread(self._add_qa_pair_impl, pair, line_indices or [])

    def _add_qa_pair_impl(self, pair: QAPair, line_indices: list[int]) -> None:
        self._hide_answering()
        container = self.query_one("#qa-log", QALog)
        at_bottom = container.is_vertical_scroll_end

        self._qa_counter += 1
        qa_id = f"qa-{self._qa_counter}"
        md_content = f"### Q: {pair.headline}\n\n{pair.answer}"
        wrapper = Vertical(
            Markdown(md_content),
            Static("X", classes="qa-delete-btn"),
            id=qa_id,
            classes="qa-entry",
        )
        container.mount(wrapper)
        self._qa_line_ids[qa_id] = line_indices
        self._qa_pairs_by_widget[qa_id] = pair

        if at_bottom:
            container.scroll_end(animate=False)
        else:
            self._qa_has_unseen = True
            self._show_qa_notice()

    def _show_qa_notice(self) -> None:
        try:
            self.query_one("#qa-new-notice").add_class("visible")
        except NoMatches:
            pass

    def _dismiss_qa_notice(self) -> None:
        self._qa_has_unseen = False
        try:
            self.query_one("#qa-new-notice").remove_class("visible")
        except NoMatches:
            pass

    def set_pipeline_status(self, msg: str, style: str = "dim") -> None:
        self.call_from_thread(self._set_pipeline_status_impl, msg, style)

    def _set_pipeline_status_impl(self, msg: str, style: str) -> None:
        status = self.query_one("#pipeline-status", Static)
        if msg:
            status.update(Text(msg, style=style))
        else:
            status.update("")

    def set_status(self, msg: str, style: str = "green") -> None:
        status = self.query_one("#transcription-status", Static)
        status.update(Text(msg, style=style))

    async def action_quit(self) -> None:
        self._shutting_down = True
        from whisper_stt.pipeline import MinecraftPipeline

        if isinstance(self._pipeline, MinecraftPipeline):
            self._pipeline.shutdown()
        recorder = self._recorder
        if recorder is not None:
            recorder.shutdown()  # type: ignore[union-attr]
        self.exit()
