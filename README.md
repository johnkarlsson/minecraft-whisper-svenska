# Whisper STT

Real-time Swedish speech-to-text terminal application for answering Minecraft questions. Captures microphone audio, transcribes it with Whisper, filters for Minecraft-related questions using Claude Haiku, and generates answers using Claude Opus with web search.

Built for Swedish-speaking kids playing Minecraft on PS5 (Bedrock Edition).

## Features

- **Real-time transcription** — interim text appears as you speak, powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT)
- **Minecraft question filtering** — Claude Haiku classifies whether transcriptions are Minecraft-related
- **AI-powered answers** — Claude Opus generates contextual answers in Swedish, with optional web search
- **Text-to-speech** — speaks answers aloud using macOS TTS
- **Interactive TUI** — two-panel Textual interface with transcription log and Q&A log
- **Persistent history** — saves transcriptions and Q&A pairs across sessions

## Installation

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

```bash
uv run whisper-stt
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `small` | Main Whisper transcription model |
| `--realtime-model` | `base` | Interim transcription model |
| `--compute-type` | `int8` | CTranslate2 compute type |
| `--no-realtime` | | Disable interim transcription |
| `--prompt TEXT` | | Override default transcription prompt |
| `--speak` | | Enable text-to-speech for answers |

### Examples

```bash
# Use a smaller model with TTS enabled
uv run whisper-stt --model base --speak

# Disable real-time interim transcription
uv run whisper-stt --no-realtime
```

## How It Works

1. Microphone audio is captured and transcribed in real-time
2. Completed transcriptions are sent to Claude Haiku for Minecraft relevance filtering
3. Relevant questions get a short headline summary (via Haiku)
4. Claude Opus generates a detailed answer in Swedish, optionally using web search
5. Answers appear in the Q&A panel and can be spoken aloud

## Project Structure

```
src/whisper_stt/
├── main.py          # CLI argument parsing and app startup
├── app.py           # Textual TUI application
├── app.tcss         # TUI styling
├── pipeline.py      # Multi-threaded question processing pipeline
├── claude_sdk.py    # Claude API wrappers (filter, summarize, answer)
├── config.py        # STTConfig dataclass and system prompts
└── history.py       # Persistent history (~/.config/whisper-stt/)
```

## Dependencies

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — speech-to-text
- [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) — real-time audio capture
- [claude-agent-sdk](https://github.com/anthropics/claude-agent-sdk) — Claude API integration
- [Textual](https://github.com/Textualize/textual) — terminal UI framework
