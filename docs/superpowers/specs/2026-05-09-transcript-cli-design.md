# Voice Memo Transcription CLI — Design

- **Status:** Draft (awaiting user review)
- **Date:** 2026-05-09
- **Project directory:** `/Users/thibauttroude/Codes/sandbox/transcript-app/`

## Summary

A polished personal CLI named `transcript` that turns voice memos (m4a / mp3 / wav / mp4 / any ffmpeg-readable audio) into speaker-labelled markdown transcripts on Apple Silicon Macs. It pairs **whisper.cpp** (CoreML + Metal + ANE-accelerated transcription) with **pyannote 3.1** (PyTorch MPS-accelerated diarization), running both stages in parallel. Designed to be invoked both directly by the user and by Claude Code as a transcription capability.

## Goals

- Run entirely locally on an Apple Silicon Mac. Audio never leaves the machine.
- Faster than real-time on a typical French interview (target: 30 min audio in <10 min wall time on M3 / 18 GB).
- Single command: `transcript foo.m4a` → markdown transcript with speaker labels on stdout.
- One-line install. Cleanly removable.
- Output structured enough that Claude Code can call the tool and parse the result.

## Non-goals

- Cross-platform support (Linux / Windows). Apple Silicon (`arm64-darwin`) only.
- Real-time / streaming transcription. Batch only.
- A GUI. CLI only.
- Custom training, fine-tuning, or model management beyond the default `large-v3`.
- A general "audio toolkit". Single purpose: audio in, transcript out.

## Decisions log (from brainstorming)

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | Tool scope | Polished personal CLI also usable by Claude Code | User explicitly asked for both |
| 2 | Default output format | Markdown | Human-readable AND parseable by Claude |
| 3 | Setup flow | README documents steps; one-line install script automates them | User preference for transparency + automation |
| 4 | Diarization backend | pyannote 3.1 | Best open-source quality; HF token is a 2-min one-time setup |
| 5 | Whisper integration | Subprocess to whisper.cpp `main` binary | Always-fresh upstream, full CoreML/Metal flag access |
| 6 | Default whisper model | `large-v3` | Best French quality; ~3 GB fits comfortably in 18 GB RAM |
| 7 | Default language | Auto-detect, `--language` flag for override | User has mixed-language use cases |
| 8 | Diarization default | On, `--no-diarize` for solo notes | All three use-cases supported (notes / interviews / meetings) |
| 9 | Output destination | Stdout by default, `-o` for file | Standard Unix convention |
| 10 | Package manager | `uv` | Modern, fast, lockfile-based |
| 11 | Data directory | `~/.local/share/transcript/` | Clean, XDG-style, portable |

## CLI contract

### Tool name and invocation

Command name: **`transcript`** (matches project directory). Installed globally so `transcript foo.m4a` works from anywhere on the user's `$PATH`.

### CLI surface

```
transcript <audio-file> [options]

POSITIONAL
  audio-file              Path to .m4a, .mp3, .wav, .mp4, or any ffmpeg-readable file

OUTPUT
  -o, --output PATH       Write transcript to file (default: stdout)
  -f, --format FORMAT     md (default) | json | srt | txt
  --no-timestamps         Omit [mm:ss] markers in markdown output

TRANSCRIPTION
  -l, --language LANG     Language code (default: auto-detect; use 'fr' for French)
  -m, --model MODEL       tiny | base | small | medium | large-v3 (default: large-v3)

DIARIZATION
  --no-diarize            Skip speaker labelling (faster, for solo notes)
  --speakers N            Fix speaker count when known (e.g. 2 for an interview)

UI
  -v, --verbose           Show step-by-step progress on stderr
  -q, --quiet             Suppress all progress (only final transcript)

UTILITY
  --version               Print version and exit
  --doctor                Check setup (whisper.cpp built, models present, HF token set, ffmpeg installed)
```

### Default markdown output (what stdout looks like)

```markdown
# voice-memo-2026-05-09.m4a

> Transcribed with whisper.cpp large-v3 (fr) + pyannote 3.1 · 2 speakers · 12m34s

## Speaker 1 [00:00]
Bonjour, nous allons parler de…

## Speaker 2 [00:14]
Oui, exactement, et je pense que…

## Speaker 1 [00:38]
…
```

### Typical invocations

```bash
# Personal note, solo, just text
transcript note.m4a --no-diarize

# Interview, French, save markdown next to audio
transcript interview.m4a -l fr --speakers 2 -o interview.md

# Meeting, JSON for downstream tooling
transcript meeting.m4a -f json -o meeting.json

# What Claude Code would invoke
transcript voice-memo.m4a   # → markdown to stdout
```

## Architecture

### Module breakdown

Each module has one job, a clean interface, and is testable in isolation.

```
src/transcript/
├── cli.py            argparse + dispatch. Zero business logic.
├── pipeline.py       Orchestrator: prepare audio → run T+D in parallel → merge → format.
├── audio.py          Detect format; ffmpeg-convert to 16 kHz mono WAV if needed.
├── transcribe.py     Subprocess wrapper around whisper.cpp; parses JSON → list[Word].
├── diarize.py        Loads pyannote pipeline (cached); returns list[Turn].
├── merge.py          Assigns words to speakers by timestamp midpoint;
│                     collapses consecutive same-speaker words into Utterances.
├── formatters/
│   ├── md.py         Markdown (default).
│   ├── json.py       Structured output.
│   ├── srt.py        Subtitle format.
│   └── txt.py        Plain prose with "Speaker N:" prefixes.
├── config.py         Resolves paths (whisper.cpp binary, model files);
│                     reads HF_TOKEN env var, falls back to macOS Keychain.
├── progress.py       Tiny stderr progress reporter (uses `rich` if -v, plain otherwise).
└── doctor.py         Self-check for --doctor flag.
```

### Internal data model

Four small dataclasses thread the entire pipeline (defined in `models.py`):

```python
@dataclass(frozen=True)
class Word:      text: str; start: float; end: float
@dataclass(frozen=True)
class Turn:      speaker: str; start: float; end: float
@dataclass(frozen=True)
class Utterance: speaker: str; start: float; end: float; text: str
@dataclass(frozen=True)
class Meta:      filename: str; duration: float; model: str; language: str; speaker_count: int
```

Every formatter consumes `(list[Utterance], Meta)`. That is the entire internal contract.

### Pipeline data flow

```
audio file (.m4a, .mp3, .wav, .mp4, …)
                    │
              audio.py: prepare()
                    │  (ffmpeg → temp 16 kHz mono WAV; passthrough if already 16 kHz mono WAV)
                    ▼
            16 kHz mono WAV
                    │
        ┌───────────┴───────────┐   ◄── ThreadPoolExecutor(max_workers=2)
        ▼                       ▼
  transcribe.run(wav)     diarize.run(wav)
  whisper.cpp subprocess  pyannote on MPS
  Metal + ANE             GPU
        │                       │
        ▼                       ▼
   list[Word]             list[Turn]
        └───────────┬───────────┘
                    ▼
            merge.assign(words, turns)
                    │
                    ▼
            list[Utterance]
                    │
                    ▼
       formatters[fmt].render(utterances, meta)
                    │
                    ▼
              stdout or -o file
```

### Why parallel transcribe + diarize

The two stages are **independent given the same prepared WAV**. Pyannote uses the GPU via PyTorch MPS; whisper.cpp uses Metal + the Apple Neural Engine — different silicon, no contention. Both stages release the GIL (pyannote uses PyTorch native ops; whisper.cpp runs as a subprocess), so a `ThreadPoolExecutor(max_workers=2)` is the correct primitive.

Wall-clock impact (rough estimate, M3 / 18 GB, 30 min French interview, large-v3):

- Sequential: ~3 min transcribe + ~2 min diarize ≈ **5 min**
- Parallel: max(3, 2) ≈ **3 min**

### Module boundary discipline

- **`transcribe.py` and `diarize.py` know nothing about each other.** Each takes a WAV path and returns its own typed list. Either could be swapped (e.g., to MLX-Whisper or NeMo Sortformer later) without touching the rest of the codebase.
- **`merge.py` is pure.** No I/O, no subprocesses — just timestamp arithmetic over two lists. Easy to unit-test exhaustively with synthetic fixtures.
- **Formatters know nothing about audio, models, or subprocess execution.** They only see `(list[Utterance], Meta)`.
- **`config.py` is the only place that touches the filesystem for paths and secrets.** Mockable in tests.

## Install & project layout

### Project layout

```
transcript-app/
├── README.md                 ← Manual install steps + usage docs
├── pyproject.toml            ← Package metadata, deps, entry point
├── uv.lock                   ← Pinned deps (managed by uv)
├── .python-version           ← Python 3.11 pin
├── scripts/
│   └── install.sh            ← One-line installer (automates the README)
├── src/
│   └── transcript/
│       ├── __init__.py
│       ├── __main__.py       ← `python -m transcript` entry
│       ├── cli.py
│       ├── pipeline.py
│       ├── audio.py
│       ├── transcribe.py
│       ├── diarize.py
│       ├── merge.py
│       ├── models.py
│       ├── config.py
│       ├── progress.py
│       ├── doctor.py
│       └── formatters/
│           ├── __init__.py
│           ├── md.py
│           ├── json.py
│           ├── srt.py
│           └── txt.py
└── tests/
    ├── conftest.py
    ├── test_merge.py
    ├── test_audio.py
    ├── test_config.py
    ├── test_formatters.py
    ├── test_cli.py
    ├── test_pipeline_integration.py
    └── fixtures/
        ├── tiny.wav             ← 8 s, 2 voices (generated via macOS `say`)
        ├── expected/
        │   ├── tiny.md
        │   ├── tiny.json
        │   ├── tiny.srt
        │   └── tiny.txt
        └── synthetic/           ← Hand-built Word/Turn lists for merge tests
```

`pyproject.toml` declares the entry point so `transcript` becomes a real shell command:

```toml
[project.scripts]
transcript = "transcript.cli:main"
```

### Where everything lives at runtime

| Path | What | Why |
|---|---|---|
| `~/.local/share/transcript/whisper.cpp/` | whisper.cpp source + built `main` binary | Built once during install; isolated from `$HOME` |
| `~/.local/share/transcript/models/ggml-large-v3.bin` | Whisper GGML weights (~3 GB) | Too big for repo; fetched by install script |
| `~/.local/share/transcript/models/ggml-large-v3-encoder.mlmodelc/` | CoreML-converted encoder | Generated locally during install (Apple Silicon only) |
| macOS Keychain entry `transcript / huggingface` | HF token | Never on disk in plaintext |
| `~/.cache/huggingface/` | pyannote model cache (~100 MB) | HuggingFace's default cache; populated on first diarize run |

`config.py` resolves these via `Path.home()` so paths stay portable.

### One-line install

User-visible install command (in the README — `<you>` is replaced with the GitHub owner once published):

```bash
curl -fsSL https://raw.githubusercontent.com/<you>/transcript-app/main/scripts/install.sh | bash
```

Or, when running locally from the cloned repo:

```bash
./scripts/install.sh
```

The same `install.sh` works in both modes — when invoked via `curl | bash` it has no local repo, so step 6 below clones one into `~/.local/share/transcript/transcript-app` before installing.

### What `scripts/install.sh` does (idempotent — safe to re-run)

```bash
#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="$HOME/.local/share/transcript"
WHISPER_DIR="$DATA_DIR/whisper.cpp"
MODELS_DIR="$DATA_DIR/models"
MODEL="large-v3"

# 1. Sanity checks
[[ "$(uname -s)" == "Darwin" ]] || { echo "macOS only"; exit 1; }
[[ "$(uname -m)" == "arm64" ]] || echo "⚠️  Not Apple Silicon — Metal/ANE unavailable"

# 2. System deps via Homebrew (skipped if present)
command -v brew    >/dev/null || { echo "Install Homebrew first: https://brew.sh"; exit 1; }
command -v ffmpeg  >/dev/null || brew install ffmpeg
command -v cmake   >/dev/null || brew install cmake
command -v uv      >/dev/null || brew install uv

# 3. whisper.cpp — clone + build with CoreML + Metal
mkdir -p "$DATA_DIR"
[[ -d "$WHISPER_DIR" ]] || git clone https://github.com/ggerganov/whisper.cpp "$WHISPER_DIR"
( cd "$WHISPER_DIR" && git pull --ff-only )
( cd "$WHISPER_DIR" && WHISPER_COREML=1 WHISPER_COREML_ALLOW_FALLBACK=1 make -j )

# 4. Models — GGML weights + CoreML encoder
mkdir -p "$MODELS_DIR"
if [[ ! -f "$MODELS_DIR/ggml-${MODEL}.bin" ]]; then
  ( cd "$WHISPER_DIR" && bash ./models/download-ggml-model.sh "$MODEL" )
  mv "$WHISPER_DIR/models/ggml-${MODEL}.bin" "$MODELS_DIR/"
fi
if [[ ! -d "$MODELS_DIR/ggml-${MODEL}-encoder.mlmodelc" ]]; then
  ( cd "$WHISPER_DIR" && bash ./models/generate-coreml-model.sh "$MODEL" )
  mv "$WHISPER_DIR/models/ggml-${MODEL}-encoder.mlmodelc" "$MODELS_DIR/"
fi

# 5. HuggingFace token → Keychain
if ! security find-generic-password -s transcript -a huggingface >/dev/null 2>&1; then
  echo
  echo "→ Need a HuggingFace token for pyannote (free)."
  echo "  1. Create one at https://huggingface.co/settings/tokens"
  echo "  2. Accept license at https://huggingface.co/pyannote/speaker-diarization-3.1"
  echo "  3. Accept license at https://huggingface.co/pyannote/segmentation-3.0"
  echo
  read -rsp "Paste your HF token (input hidden): " HF_TOKEN; echo
  security add-generic-password -s transcript -a huggingface -w "$HF_TOKEN"
fi

# 6. Install the CLI globally
#    If invoked from inside the cloned repo: install from $(pwd).
#    If invoked via `curl | bash`: clone the repo into the data dir and install from there.
if [[ -f "$(pwd)/pyproject.toml" ]] && grep -q "^name = \"transcript-app\"" "$(pwd)/pyproject.toml" 2>/dev/null; then
  REPO_DIR="$(pwd)"
else
  REPO_DIR="$DATA_DIR/transcript-app"
  [[ -d "$REPO_DIR" ]] || git clone https://github.com/<you>/transcript-app "$REPO_DIR"
  ( cd "$REPO_DIR" && git pull --ff-only )
fi
uv tool install --from "$REPO_DIR" transcript-app

# 7. Smoke check
echo
transcript --doctor
echo
echo "✅  Done. Try: transcript path/to/voice-memo.m4a"
```

### What the README contains

1. **Overview** — what the tool does, in 3 lines.
2. **Quick install** — the `curl | bash` one-liner, plus a collapsible "what does it do?" section linking to `scripts/install.sh`.
3. **Manual install** — the same numbered steps the script does, in case anything fails or the user wants full control.
4. **Usage** — realistic invocations with sample output (the same as the CLI section above).
5. **Troubleshooting** — `transcript --doctor`, common pyannote license errors, ffmpeg-not-found, M-series checks.
6. **For Claude Code users** — a 4-line snippet showing how Claude can invoke `transcript foo.m4a` via its `Bash` tool and consume the markdown output.

## Error handling

### Philosophy

**Fail fast at boundaries, with actionable messages.** Internal modules trust their inputs; the CLI and `--doctor` are the validation boundaries.

Each error category gets a distinct exit code so Claude Code (or shell scripts) can branch on failures:

| Code | Meaning | Example |
|---|---|---|
| 0 | Success | — |
| 1 | Generic / unexpected | uncaught exception |
| 2 | Bad CLI usage | unknown flag, missing arg (argparse default) |
| 10 | Audio file problem | missing file, unreadable, ffmpeg conversion failed |
| 11 | Setup incomplete | whisper.cpp not built, model missing, ffmpeg not on PATH |
| 12 | Authentication problem | HF token missing or rejected, license not accepted |
| 13 | Runtime resource problem | out of memory, GPU unavailable when required |

Every error also prints **what to try next** on stderr. Example:

```
✗ pyannote refused the download (HTTP 401)
  → Likely cause: license not accepted on HuggingFace.
    1. Sign in at https://huggingface.co
    2. Click "Agree" on:
       - https://huggingface.co/pyannote/speaker-diarization-3.1
       - https://huggingface.co/pyannote/segmentation-3.0
    3. Re-run the same command.
```

### Failure modes handled explicitly

| Where | Failure | What we do |
|---|---|---|
| `audio.py` | File missing / unreadable | exit 10 + path + suggestion |
| `audio.py` | ffmpeg not installed | exit 11 + `brew install ffmpeg` hint |
| `audio.py` | Audio < 0.5 s | exit 10 + "audio too short to transcribe" |
| `transcribe.py` | whisper.cpp binary not found | exit 11 + "run scripts/install.sh" |
| `transcribe.py` | Model file missing | exit 11 + path + install hint |
| `transcribe.py` | whisper.cpp non-zero exit | exit 1 + last 20 lines of its stderr |
| `diarize.py` | HF token missing | exit 12 + "set HF_TOKEN env var or run install.sh" |
| `diarize.py` | HF 401 / license refused | exit 12 + the multi-line hint above |
| `diarize.py` | MPS unavailable | warn + fall back to CPU automatically |
| `pipeline.py` | Either parallel task raises | cancel sibling, propagate first error |
| `cli.py` | Uncaught exception | exit 1 + traceback only with `-v`; otherwise one-line message |

### `--doctor` subcommand

`transcript --doctor` runs all boundary checks in one pass and prints a green/red checklist:

- whisper.cpp binary present and executable
- whisper GGML model present at expected path
- CoreML encoder present (warn-only on non-Apple-Silicon)
- ffmpeg on `$PATH`
- HF token reachable (env var or Keychain)
- pyannote pipeline loadable (light, no audio)
- MPS available
- Sample inference on a 2-second built-in fixture (optional, behind `--doctor --deep`)

The README points to `transcript --doctor` as the first troubleshooting step.

## Testing strategy

### Unit tests (fast, no audio, no network)

Target: <1 s total runtime.

| Module | What's tested | How |
|---|---|---|
| `merge.py` | Word→speaker assignment; edge cases (word straddling turn boundary, gap, silence between turns); utterance collapsing | Hand-built `list[Word]` and `list[Turn]` fixtures, golden `list[Utterance]` |
| `formatters/*` | Each format renders a known utterance list correctly | Golden-file comparison (`tests/fixtures/expected/*.md|.json|.srt|.txt`) |
| `audio.py` | Format detection from extension + ffprobe; conversion command construction | mock `subprocess.run` |
| `config.py` | Path resolution; env-var precedence over Keychain; missing token raises clean error | mock `keyring`, `os.environ` |
| `cli.py` | Argparse maps flags to pipeline kwargs correctly | invoke `main(argv=[...])`, assert calls |

### Integration tests (slow, real binaries, opt-in)

Gated behind `pytest -m integration`.

- `tests/fixtures/tiny.wav`: an 8-second clip, two TTS voices alternating (generated once via macOS `say` with two different voices, committed to repo).
- One end-to-end test runs the full pipeline against `tiny.wav` with whisper `base` model (small, fast) + real pyannote, asserts the output contains expected words and exactly 2 speakers.
- Skipped automatically if whisper.cpp binary or HF token is unavailable, with a clear `pytest skip` reason.

### Smoke test in CI / `--doctor`

Same as `--doctor` above — all boundaries reachable, no inference. Runs in seconds.

### What we deliberately don't test

- Whisper.cpp transcription accuracy (not our code).
- Pyannote diarization accuracy (not our code).
- Audio quality at edge sample rates (ffmpeg's job).

### Coverage targets

- `>90 %` on pure modules (`merge`, `formatters`, `audio`, `config`, `cli`).
- Integration tests are correctness-checking, not coverage-driving — no enforced coverage on `pipeline`, `transcribe`, `diarize`.

## Out of scope (future versions)

- Batch processing (transcribe a directory of files).
- Configurable diarization backend (`--backend nemo` / `--backend simple`).
- Custom whisper.cpp build flags from CLI.
- Persistent live progress UI (would need `rich` always-on, not just `-v`).
- Speaker label overrides (`--speaker-1-name "Alice"`).
- Automatic uninstall script.
- Linux support (would mostly work; CoreML stays a no-op).
- Word-level timestamps in the markdown output (currently utterance-level only).

## Open questions to validate during implementation

- Exact whisper.cpp CLI flags for word-level JSON output (`-ml 1 --split-on-word -ojf`?) — verify against the version that gets cloned by `install.sh` at implementation time, since flag spellings have drifted in past releases.
- Whether `pyannote.Pipeline.__call__` accepts `num_speakers=N` directly, or needs `min_speakers=N, max_speakers=N` for fixed-count diarization — verify against the pyannote 3.1 docs at implementation time.
- Whether `uv tool install --from <local-path>` works on a project that bundles a non-Python build (whisper.cpp). Fallback: install the Python package only and rely on `~/.local/share/transcript/whisper.cpp/main` for the binary, which is the current design.

---

*This design is the input to the writing-plans skill, which will produce a step-by-step implementation plan.*
