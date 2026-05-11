# transcript

Local voice-memo transcription with speaker diarization on Apple Silicon Macs.
Pairs **whisper.cpp** (CoreML + Metal + ANE) with **NeMo Streaming Sortformer 4spk-v2.1** (CPU)
and merges their output into a markdown transcript. Audio never leaves your machine.

```
$ transcript interview.m4a
# interview.m4a

> Transcribed with whisper.cpp large-v3 (fr) + NeMo Streaming Sortformer 4spk-v2.1 · 2 speakers · 12m34s

## Speaker 1 [00:00]
Bonjour, nous allons parler de…

## Speaker 2 [00:14]
Oui, exactement, et je pense que…
```

## Quick install

```bash
curl -fsSL https://raw.githubusercontent.com/thbttrd/transcript-cli/main/scripts/install.sh | bash
```

After install: `transcript --doctor` to verify everything is in place.

<details>
<summary>What does the install script actually do?</summary>

1. Checks you're on macOS / Apple Silicon and have Homebrew.
2. Installs `ffmpeg`, `cmake`, `uv` if missing (via brew).
3. Clones `whisper.cpp` to `~/.local/share/transcript/whisper.cpp` and builds it with `WHISPER_COREML=1`.
4. Downloads `ggml-large-v3.bin` (~3 GB) and generates the CoreML encoder.
5. Installs the `transcript` CLI globally via `uv tool install`. NeMo Sortformer weights download lazily on first run (~120 MB, cached in `~/.cache/huggingface/`).
6. Runs `transcript --doctor` to confirm.

Everything lives under `~/.local/share/transcript/` — to uninstall, delete that directory and run `uv tool uninstall transcript-app`.

</details>

## Manual install

If you'd rather do it by hand:

```bash
brew install ffmpeg cmake uv

# whisper.cpp with CoreML
git clone https://github.com/ggerganov/whisper.cpp ~/.local/share/transcript/whisper.cpp
cd ~/.local/share/transcript/whisper.cpp
WHISPER_COREML=1 WHISPER_COREML_ALLOW_FALLBACK=1 make -j

# Models
bash ./models/download-ggml-model.sh large-v3

# CoreML conversion needs torch + coremltools + openai-whisper + ane_transformers,
# pinned to versions that disagree with the runtime's torch — keep them isolated.
uv venv --python 3.11 ~/.local/share/transcript/coreml-venv
uv pip install --python ~/.local/share/transcript/coreml-venv/bin/python \
  -r ./models/requirements-coreml.txt
PATH="$HOME/.local/share/transcript/coreml-venv/bin:$PATH" \
  bash ./models/generate-coreml-model.sh large-v3

mkdir -p ~/.local/share/transcript/models
mv models/ggml-large-v3.bin ~/.local/share/transcript/models/
mv models/ggml-large-v3-encoder.mlmodelc ~/.local/share/transcript/models/

# Install the CLI — pin Python 3.11 per pyproject's requires-python.
# NeMo Streaming Sortformer weights (~120 MB) download lazily on first run.
git clone https://github.com/<you>/transcript-app
cd transcript-app
uv tool install --python 3.11 --force --from "$(pwd)" transcript-app
```

## Usage

```bash
# Personal note, solo, no diarization
transcript note.m4a --no-diarize

# Interview, French, save to file
transcript interview.m4a -l fr --speakers 2 -o interview.md

# Meeting, JSON output
transcript meeting.m4a -f json -o meeting.json

# What Claude Code typically calls
transcript voice-memo.m4a    # → markdown to stdout
```

Full flag reference: `transcript --help`.

## Troubleshooting

Run `transcript --doctor` first. It checks every prerequisite and tells you what's missing.

Common issues:

| Symptom | Cause | Fix |
|---|---|---|
| `whisper.cpp binary not found` | Install incomplete | `bash scripts/install.sh` |
| `ffmpeg not found` | Missing system dep | `brew install ffmpeg` |
| Hangs at "transcribing" | First diarize run downloading model | Wait — Sortformer weights (~120 MB) cache to `~/.cache/huggingface/`, only downloaded once |
| `--doctor` says CoreML encoder missing | Built without `WHISPER_COREML=1` | Re-run install script |
| `ModuleNotFoundError: No module named 'torch'` during install | CoreML conversion deps not installed | Re-run install script (newer versions stand up an isolated venv at `~/.local/share/transcript/coreml-venv`) |
| `torch>=…,==2.6.0 has no wheels with a matching Python ABI tag` | `uv` picked a Python version outside pyproject's `requires-python = ">=3.11,<3.12"` | Re-run install script — it now passes `--python 3.11` explicitly. If you used a custom command, add `--python 3.11` to `uv tool install` |
| Integration tests skip "tiny.wav not generated" | Fixture not built yet | After install.sh, run `bash scripts/generate_tiny_wav.sh` to create the test fixture |

## For Claude Code users

Add this single capability to your Claude Code session and Claude can transcribe voice memos for you. Once installed, Claude can invoke it via the `Bash` tool:

```
You: Transcribe ~/Desktop/meeting.m4a and summarize the action items.
Claude: [runs `transcript ~/Desktop/meeting.m4a` via Bash, reads the markdown output, summarizes]
```

Output goes to stdout, so Claude captures it directly. Use `-f json` if you want Claude to do timestamp-aware reasoning.

## Development

```bash
git clone https://github.com/thbttrd/transcript-cli
cd transcript-cli
uv sync --all-extras

# Unit tests (fast, no audio, no network)
uv run pytest -m "not integration"

# Integration tests (require install)
uv run pytest -m integration
```

## Architecture

See [`docs/superpowers/specs/2026-05-09-transcript-cli-design.md`](docs/superpowers/specs/2026-05-09-transcript-cli-design.md) for the full design.

## Licence

Personal project. whisper.cpp is MIT. NeMo Sortformer is NSCLv1 (NVIDIA Source Code License — free for personal/commercial use).
