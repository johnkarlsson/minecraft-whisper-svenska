from __future__ import annotations

import argparse

from whisper_stt.config import STTConfig


def parse_args() -> tuple[STTConfig, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="Real-time Swedish speech-to-text")
    parser.add_argument(
        "--model",
        default="small",
        help="Main transcription model (default: small)",
    )
    parser.add_argument(
        "--realtime-model",
        default="base",
        help="Realtime transcription model (default: base)",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="Compute type for CTranslate2 (default: int8)",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Disable realtime interim transcription",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Initial prompt to bias transcription (overrides default Minecraft prompt)",
    )
    parser.add_argument(
        "--speak",
        action="store_true",
        help="Speak Minecraft answers aloud via TTS (pauses mic during speech)",
    )
    args = parser.parse_args()
    config = STTConfig(
        main_model=args.model,
        realtime_model=args.realtime_model,
        compute_type=args.compute_type,
        enable_realtime=not args.no_realtime,
    )
    if args.prompt is not None:
        config.initial_prompt = args.prompt
    return config, args


def main() -> None:
    config, args = parse_args()

    # Pre-initialize the multiprocessing resource tracker while sys.stderr is
    # still a real fd.  Textual redirects stderr once the app starts, making
    # stderr.fileno() return -1.  tqdm (pulled in by huggingface_hub during
    # model download) creates a multiprocessing lock whose resource-tracker
    # subprocess spawn fails with "bad value(s) in fds_to_keep" when -1 is
    # in the fds-to-pass list.
    from multiprocessing import RLock as _MpRLock

    _MpRLock()

    from whisper_stt.app import WhisperApp

    app = WhisperApp(config, args)
    app.run()
