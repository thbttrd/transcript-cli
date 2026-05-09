# Voice Memo Transcription CLI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `transcript` CLI as specified in `docs/superpowers/specs/2026-05-09-transcript-cli-design.md` — a polished personal CLI that transcribes voice memos using whisper.cpp + pyannote with full Apple Silicon acceleration.

**Architecture:** Python 3.11 package using `uv`, distributed as a single CLI command `transcript`. Shells out to a locally-built whisper.cpp `main` binary for transcription; calls pyannote 3.1 in-process for diarization; runs both stages in parallel via `ThreadPoolExecutor`; merges results by timestamp; emits markdown by default.

**Tech Stack:** Python 3.11 · uv · pytest + pytest-mock · pyannote.audio 3.1 · torch (MPS) · keyring · whisper.cpp (external) · ffmpeg (external)

**Spec:** `docs/superpowers/specs/2026-05-09-transcript-cli-design.md`

**Conventions used throughout this plan:**
- Every code file uses Python 3.11+ syntax (`X | Y` unions, `list[T]`, etc.)
- All Python source lives under `src/transcript/`; all tests under `tests/`
- Run all `uv` commands from the project root unless noted otherwise
- Commit messages follow conventional commits (`feat:`, `test:`, `docs:`, `chore:`)
- The engineer should NOT run integration tests (`pytest -m integration`) until Task 17 — they require the install script + HF token

---

## Task 1: Project bootstrap (uv + pyproject.toml + package skeleton)

**Files:**
- Create: `.python-version`
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `src/transcript/__init__.py`
- Create: `src/transcript/__main__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Pin Python version**

Create `.python-version`:
```
3.11
```

- [ ] **Step 2: Create `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
dist/
*.egg-info/
.venv/
venv/

# Tests / tools
.pytest_cache/
.coverage
.coverage.*
htmlcov/
.ruff_cache/

# Editor
.vscode/
.idea/
*.swp
.DS_Store

# Project-specific
*.log
```

- [ ] **Step 3: Create `pyproject.toml`**

```toml
[project]
name = "transcript-app"
version = "0.1.0"
description = "Local voice-memo transcription with speaker diarization (whisper.cpp + pyannote)"
requires-python = ">=3.11,<3.13"
authors = [{ name = "Thibaut Troude" }]
dependencies = [
    "pyannote.audio>=3.1,<3.4",
    "torch>=2.2",
    "torchaudio>=2.2",
    "keyring>=24.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "pytest-mock>=3.12",
    "ruff>=0.5",
]

[project.scripts]
transcript = "transcript.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/transcript"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = ["-ra", "--strict-markers"]
markers = [
    "integration: real whisper.cpp + pyannote, requires install + HF token",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
```

- [ ] **Step 4: Create empty package files**

`src/transcript/__init__.py`:
```python
__version__ = "0.1.0"
```

`src/transcript/__main__.py`:
```python
from transcript.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```

`tests/__init__.py`: empty file.

`tests/conftest.py`:
```python
import sys
from pathlib import Path

# Ensure src/ is on the path during tests without requiring an editable install at every step.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
```

- [ ] **Step 5: Create a placeholder `cli.py` so `__main__` resolves**

`src/transcript/cli.py`:
```python
def main(argv: list[str] | None = None) -> int:
    print("transcript 0.1.0 (skeleton)")
    return 0
```

- [ ] **Step 6: Verify uv can install the project**

Run: `uv sync --all-extras`
Expected: succeeds, creates `.venv/`, installs all deps including pyannote and torch (this is the slow first install — ~1–2 GB download).

- [ ] **Step 7: Verify the entry point works**

Run: `uv run transcript`
Expected: `transcript 0.1.0 (skeleton)`

- [ ] **Step 8: Commit**

```bash
git add .python-version .gitignore pyproject.toml src/ tests/ uv.lock
git commit -m "chore: bootstrap python package with uv"
```

---

## Task 2: Data model (`models.py`)

**Files:**
- Create: `src/transcript/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing tests**

`tests/test_models.py`:
```python
import pytest
from transcript.models import Meta, Turn, Utterance, Word


def test_word_is_frozen():
    w = Word(text="hello", start=0.0, end=0.5)
    with pytest.raises(AttributeError):
        w.text = "world"  # type: ignore[misc]


def test_turn_fields():
    t = Turn(speaker="Speaker 1", start=1.0, end=3.5)
    assert t.speaker == "Speaker 1"
    assert t.start == 1.0
    assert t.end == 3.5


def test_utterance_fields():
    u = Utterance(speaker="Speaker 2", start=0.0, end=1.0, text="bonjour")
    assert u.text == "bonjour"


def test_meta_fields():
    m = Meta(
        filename="voice.m4a",
        duration=754.0,
        model="large-v3",
        language="fr",
        speaker_count=2,
    )
    assert m.duration == 754.0
    assert m.speaker_count == 2
```

- [ ] **Step 2: Run the test to confirm failure**

Run: `uv run pytest tests/test_models.py -v`
Expected: ImportError — `transcript.models` doesn't exist.

- [ ] **Step 3: Implement `models.py`**

`src/transcript/models.py`:
```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Word:
    text: str
    start: float  # seconds
    end: float    # seconds


@dataclass(frozen=True)
class Turn:
    speaker: str  # e.g. "Speaker 1"
    start: float
    end: float


@dataclass(frozen=True)
class Utterance:
    speaker: str
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Meta:
    filename: str
    duration: float       # seconds
    model: str            # e.g. "large-v3"
    language: str         # ISO code e.g. "fr"
    speaker_count: int    # 1 if --no-diarize, otherwise pyannote-detected count
```

- [ ] **Step 4: Run the test to confirm pass**

Run: `uv run pytest tests/test_models.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/transcript/models.py tests/test_models.py
git commit -m "feat(models): add Word, Turn, Utterance, Meta dataclasses"
```

---

## Task 3: Merge logic (`merge.py`)

This is the algorithmic heart of the project: assigning each transcribed word to a speaker turn by timestamp midpoint, then collapsing consecutive same-speaker words into utterances.

**Files:**
- Create: `src/transcript/merge.py`
- Create: `tests/test_merge.py`

- [ ] **Step 1: Write failing tests covering the key cases**

`tests/test_merge.py`:
```python
from transcript.merge import assign
from transcript.models import Turn, Utterance, Word


def w(text: str, s: float, e: float) -> Word:
    return Word(text=text, start=s, end=e)


def t(speaker: str, s: float, e: float) -> Turn:
    return Turn(speaker=speaker, start=s, end=e)


def test_empty_inputs_returns_empty():
    assert assign([], []) == []


def test_words_with_no_turns_get_unknown_speaker():
    words = [w("hello", 0.0, 0.5)]
    result = assign(words, [])
    assert result == [Utterance(speaker="Unknown", start=0.0, end=0.5, text="hello")]


def test_single_speaker_collapses_into_one_utterance():
    words = [w(" bon", 0.0, 0.3), w("jour", 0.3, 0.6), w(" amis", 0.6, 1.0)]
    turns = [t("Speaker 1", 0.0, 1.0)]
    result = assign(words, turns)
    assert result == [
        Utterance(speaker="Speaker 1", start=0.0, end=1.0, text="bonjour amis")
    ]


def test_speaker_change_creates_two_utterances():
    words = [
        w(" hello", 0.0, 0.5),
        w(" world", 0.5, 1.0),
        w(" hi", 2.0, 2.3),
        w(" there", 2.3, 2.7),
    ]
    turns = [
        t("Speaker 1", 0.0, 1.5),
        t("Speaker 2", 1.5, 3.0),
    ]
    result = assign(words, turns)
    assert result == [
        Utterance(speaker="Speaker 1", start=0.0, end=1.0, text="hello world"),
        Utterance(speaker="Speaker 2", start=2.0, end=2.7, text="hi there"),
    ]


def test_word_midpoint_is_what_decides_assignment():
    # Word from 0.9 to 1.2; midpoint 1.05; turn boundary at 1.0.
    # Midpoint is in turn 2, so word is assigned to Speaker 2.
    words = [w(" overlap", 0.9, 1.2)]
    turns = [t("Speaker 1", 0.0, 1.0), t("Speaker 2", 1.0, 2.0)]
    result = assign(words, turns)
    assert result == [Utterance(speaker="Speaker 2", start=0.9, end=1.2, text="overlap")]


def test_three_speakers_alternating():
    words = [
        w(" a", 0.0, 0.2),
        w(" b", 1.0, 1.2),
        w(" c", 2.0, 2.2),
        w(" d", 3.0, 3.2),
    ]
    turns = [
        t("Speaker 1", 0.0, 0.5),
        t("Speaker 2", 0.9, 1.5),
        t("Speaker 3", 1.9, 2.5),
        t("Speaker 1", 2.9, 3.5),
    ]
    result = assign(words, turns)
    assert [u.speaker for u in result] == ["Speaker 1", "Speaker 2", "Speaker 3", "Speaker 1"]
    assert [u.text for u in result] == ["a", "b", "c", "d"]


def test_word_in_gap_between_turns_is_unknown():
    words = [w(" lonely", 1.0, 1.5)]
    turns = [t("Speaker 1", 0.0, 0.5), t("Speaker 2", 2.0, 3.0)]
    result = assign(words, turns)
    assert result == [Utterance(speaker="Unknown", start=1.0, end=1.5, text="lonely")]


def test_text_is_stripped_and_concatenated_in_order():
    words = [w("  he", 0.0, 0.1), w("llo", 0.1, 0.2), w("  world  ", 0.2, 0.3)]
    turns = [t("Speaker 1", 0.0, 0.5)]
    result = assign(words, turns)
    assert result[0].text == "hello  world"
```

- [ ] **Step 2: Run the test to confirm failure**

Run: `uv run pytest tests/test_merge.py -v`
Expected: ImportError — `transcript.merge` doesn't exist.

- [ ] **Step 3: Implement `merge.py`**

`src/transcript/merge.py`:
```python
from transcript.models import Turn, Utterance, Word

UNKNOWN = "Unknown"


def _speaker_at(t: float, turns: list[Turn]) -> str:
    for turn in turns:
        if turn.start <= t <= turn.end:
            return turn.speaker
    return UNKNOWN


def assign(words: list[Word], turns: list[Turn]) -> list[Utterance]:
    """Assign each word to a speaker by timestamp midpoint, then collapse runs."""
    if not words:
        return []

    utterances: list[Utterance] = []
    current_speaker: str | None = None
    current_words: list[Word] = []

    def flush() -> None:
        if not current_words:
            return
        utterances.append(
            Utterance(
                speaker=current_speaker or UNKNOWN,
                start=current_words[0].start,
                end=current_words[-1].end,
                text="".join(w.text for w in current_words).strip(),
            )
        )

    for word in words:
        midpoint = (word.start + word.end) / 2
        speaker = _speaker_at(midpoint, turns)
        if speaker != current_speaker and current_words:
            flush()
            current_words = []
        current_speaker = speaker
        current_words.append(word)

    flush()
    return utterances
```

- [ ] **Step 4: Run the test to confirm pass**

Run: `uv run pytest tests/test_merge.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/transcript/merge.py tests/test_merge.py
git commit -m "feat(merge): assign words to speakers and collapse into utterances"
```

---

## Task 4: Markdown formatter (`formatters/md.py`)

**Files:**
- Create: `src/transcript/formatters/__init__.py`
- Create: `src/transcript/formatters/md.py`
- Create: `tests/test_formatters_md.py`

- [ ] **Step 1: Write failing tests**

`tests/test_formatters_md.py`:
```python
from transcript.formatters.md import render
from transcript.models import Meta, Utterance


def _meta(speakers: int = 2) -> Meta:
    return Meta(
        filename="voice.m4a",
        duration=754.0,
        model="large-v3",
        language="fr",
        speaker_count=speakers,
    )


def test_md_with_timestamps():
    utterances = [
        Utterance(speaker="Speaker 1", start=0.0, end=14.2, text="bonjour"),
        Utterance(speaker="Speaker 2", start=14.2, end=38.5, text="oui"),
    ]
    out = render(utterances, _meta(), with_timestamps=True)
    assert "# voice.m4a" in out
    assert "## Speaker 1 [00:00]" in out
    assert "## Speaker 2 [00:14]" in out
    assert "bonjour" in out
    assert "oui" in out
    assert "12m34s" in out  # 754 seconds duration
    assert "2 speakers" in out
    assert "large-v3" in out
    assert "fr" in out


def test_md_without_timestamps():
    utterances = [Utterance(speaker="Speaker 1", start=0.0, end=1.0, text="bonjour")]
    out = render(utterances, _meta(speakers=1), with_timestamps=False)
    assert "## Speaker 1\n" in out
    assert "[00:00]" not in out
    assert "1 speaker" in out  # singular


def test_md_empty_utterances_still_has_header():
    out = render([], _meta(), with_timestamps=True)
    assert "# voice.m4a" in out
    assert "## Speaker" not in out


def test_md_timestamps_format_for_long_duration():
    # 1h 23m 45s = 5025s
    m = Meta(filename="long.m4a", duration=5025.0, model="large-v3", language="fr", speaker_count=1)
    out = render([], m, with_timestamps=True)
    assert "1h23m45s" in out
```

- [ ] **Step 2: Run the test to confirm failure**

Run: `uv run pytest tests/test_formatters_md.py -v`
Expected: ImportError — `transcript.formatters.md` doesn't exist.

- [ ] **Step 3: Create the formatters package**

`src/transcript/formatters/__init__.py`:
```python
"""Output formatters. Each module exposes `render(utterances, meta, **kwargs) -> str`."""
```

- [ ] **Step 4: Implement `md.py`**

`src/transcript/formatters/md.py`:
```python
from transcript.models import Meta, Utterance


def _format_timestamp(seconds: float) -> str:
    """Format `seconds` as mm:ss for utterance timestamps."""
    total = int(seconds)
    return f"{total // 60:02d}:{total % 60:02d}"


def _format_duration(seconds: float) -> str:
    """Format `seconds` as a compact human duration (12m34s, 1h23m45s)."""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def render(utterances: list[Utterance], meta: Meta, *, with_timestamps: bool = True) -> str:
    speaker_word = "speaker" if meta.speaker_count == 1 else "speakers"
    lines: list[str] = [
        f"# {meta.filename}",
        "",
        (
            f"> Transcribed with whisper.cpp {meta.model} ({meta.language}) "
            f"+ pyannote 3.1 · {meta.speaker_count} {speaker_word} · "
            f"{_format_duration(meta.duration)}"
        ),
        "",
    ]
    for u in utterances:
        ts = f" [{_format_timestamp(u.start)}]" if with_timestamps else ""
        lines.append(f"## {u.speaker}{ts}")
        lines.append(u.text)
        lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 5: Run the test to confirm pass**

Run: `uv run pytest tests/test_formatters_md.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/transcript/formatters/ tests/test_formatters_md.py
git commit -m "feat(formatters): add markdown renderer with timestamps and duration"
```

---

## Task 5: JSON formatter (`formatters/json.py`)

**Files:**
- Create: `src/transcript/formatters/json.py`
- Create: `tests/test_formatters_json.py`

- [ ] **Step 1: Write failing tests**

`tests/test_formatters_json.py`:
```python
import json

from transcript.formatters.json import render
from transcript.models import Meta, Utterance


def test_json_structure():
    utterances = [
        Utterance(speaker="Speaker 1", start=0.0, end=1.0, text="bonjour"),
        Utterance(speaker="Speaker 2", start=1.0, end=2.0, text="salut"),
    ]
    meta = Meta(filename="v.m4a", duration=2.0, model="large-v3", language="fr", speaker_count=2)
    out = render(utterances, meta)
    data = json.loads(out)

    assert data["meta"]["filename"] == "v.m4a"
    assert data["meta"]["duration"] == 2.0
    assert data["meta"]["model"] == "large-v3"
    assert data["meta"]["language"] == "fr"
    assert data["meta"]["speaker_count"] == 2

    assert len(data["utterances"]) == 2
    assert data["utterances"][0] == {
        "speaker": "Speaker 1",
        "start": 0.0,
        "end": 1.0,
        "text": "bonjour",
    }


def test_json_is_pretty_printed():
    out = render([], Meta(filename="x", duration=0.0, model="m", language="fr", speaker_count=0))
    assert "\n" in out  # 2-space indent uses newlines
```

- [ ] **Step 2: Run the test to confirm failure**

Run: `uv run pytest tests/test_formatters_json.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `json.py`**

`src/transcript/formatters/json.py`:
```python
import json
from dataclasses import asdict

from transcript.models import Meta, Utterance


def render(utterances: list[Utterance], meta: Meta) -> str:
    payload = {
        "meta": asdict(meta),
        "utterances": [asdict(u) for u in utterances],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)
```

- [ ] **Step 4: Run the test to confirm pass**

Run: `uv run pytest tests/test_formatters_json.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/transcript/formatters/json.py tests/test_formatters_json.py
git commit -m "feat(formatters): add JSON renderer"
```

---

## Task 6: SRT formatter (`formatters/srt.py`)

**Files:**
- Create: `src/transcript/formatters/srt.py`
- Create: `tests/test_formatters_srt.py`

- [ ] **Step 1: Write failing tests**

`tests/test_formatters_srt.py`:
```python
from transcript.formatters.srt import render
from transcript.models import Meta, Utterance


META = Meta(filename="v.m4a", duration=10.0, model="large-v3", language="fr", speaker_count=2)


def test_srt_basic_format():
    utterances = [
        Utterance(speaker="Speaker 1", start=0.0, end=2.5, text="bonjour"),
        Utterance(speaker="Speaker 2", start=2.5, end=5.0, text="salut"),
    ]
    out = render(utterances, META)
    expected = (
        "1\n"
        "00:00:00,000 --> 00:00:02,500\n"
        "Speaker 1: bonjour\n"
        "\n"
        "2\n"
        "00:00:02,500 --> 00:00:05,000\n"
        "Speaker 2: salut\n"
    )
    assert out == expected


def test_srt_long_timestamp():
    u = Utterance(speaker="X", start=3661.123, end=3662.456, text="a")
    out = render([u], META)
    assert "01:01:01,123 --> 01:01:02,456" in out


def test_srt_empty():
    assert render([], META) == ""
```

- [ ] **Step 2: Run the test to confirm failure**

Run: `uv run pytest tests/test_formatters_srt.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `srt.py`**

`src/transcript/formatters/srt.py`:
```python
from transcript.models import Meta, Utterance


def _srt_time(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def render(utterances: list[Utterance], meta: Meta) -> str:  # noqa: ARG001 (meta unused but keeps signature uniform)
    blocks: list[str] = []
    for i, u in enumerate(utterances, start=1):
        blocks.append(
            f"{i}\n"
            f"{_srt_time(u.start)} --> {_srt_time(u.end)}\n"
            f"{u.speaker}: {u.text}\n"
        )
    return "\n".join(blocks)
```

- [ ] **Step 4: Run the test to confirm pass**

Run: `uv run pytest tests/test_formatters_srt.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/transcript/formatters/srt.py tests/test_formatters_srt.py
git commit -m "feat(formatters): add SRT renderer"
```

---

## Task 7: Plain-text formatter (`formatters/txt.py`) and dispatcher

**Files:**
- Create: `src/transcript/formatters/txt.py`
- Create: `tests/test_formatters_txt.py`
- Modify: `src/transcript/formatters/__init__.py` (add `get(name)` dispatcher)
- Create: `tests/test_formatters_dispatcher.py`

- [ ] **Step 1: Write failing tests for txt formatter**

`tests/test_formatters_txt.py`:
```python
from transcript.formatters.txt import render
from transcript.models import Meta, Utterance


META = Meta(filename="v.m4a", duration=5.0, model="large-v3", language="fr", speaker_count=2)


def test_txt_basic():
    utterances = [
        Utterance(speaker="Speaker 1", start=0.0, end=1.0, text="bonjour"),
        Utterance(speaker="Speaker 2", start=1.0, end=2.0, text="salut"),
    ]
    out = render(utterances, META)
    assert out == "Speaker 1: bonjour\nSpeaker 2: salut\n"


def test_txt_empty():
    assert render([], META) == ""
```

- [ ] **Step 2: Run the test to confirm failure**

Run: `uv run pytest tests/test_formatters_txt.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `txt.py`**

`src/transcript/formatters/txt.py`:
```python
from transcript.models import Meta, Utterance


def render(utterances: list[Utterance], meta: Meta) -> str:  # noqa: ARG001
    return "".join(f"{u.speaker}: {u.text}\n" for u in utterances)
```

- [ ] **Step 4: Confirm txt tests pass**

Run: `uv run pytest tests/test_formatters_txt.py -v`
Expected: 2 passed.

- [ ] **Step 5: Write failing tests for the dispatcher**

`tests/test_formatters_dispatcher.py`:
```python
import pytest

from transcript.formatters import get


def test_get_returns_callable_per_format():
    for name in ("md", "json", "srt", "txt"):
        assert callable(get(name))


def test_get_unknown_format_raises():
    with pytest.raises(ValueError, match="unknown format"):
        get("xml")


def test_get_md_supports_with_timestamps_kwarg():
    from transcript.models import Meta, Utterance

    fn = get("md")
    meta = Meta(filename="v", duration=1.0, model="m", language="fr", speaker_count=1)
    out = fn([Utterance(speaker="A", start=0.0, end=1.0, text="hi")], meta, with_timestamps=False)
    assert "[00:00]" not in out
```

- [ ] **Step 6: Run dispatcher test to confirm failure**

Run: `uv run pytest tests/test_formatters_dispatcher.py -v`
Expected: ImportError on `get`.

- [ ] **Step 7: Implement the dispatcher**

Replace `src/transcript/formatters/__init__.py`:
```python
"""Output formatters. Each module exposes `render(utterances, meta, **kwargs) -> str`."""
from collections.abc import Callable

from transcript.formatters import json as _json
from transcript.formatters import md as _md
from transcript.formatters import srt as _srt
from transcript.formatters import txt as _txt

_REGISTRY: dict[str, Callable[..., str]] = {
    "md": _md.render,
    "json": _json.render,
    "srt": _srt.render,
    "txt": _txt.render,
}


def get(name: str) -> Callable[..., str]:
    if name not in _REGISTRY:
        raise ValueError(f"unknown format: {name!r} (expected one of {list(_REGISTRY)})")
    return _REGISTRY[name]
```

- [ ] **Step 8: Run all formatter tests to confirm pass**

Run: `uv run pytest tests/test_formatters_*.py -v`
Expected: 14 passed (4 md + 2 json + 3 srt + 2 txt + 3 dispatcher).

- [ ] **Step 9: Commit**

```bash
git add src/transcript/formatters/ tests/test_formatters_txt.py tests/test_formatters_dispatcher.py
git commit -m "feat(formatters): add txt renderer and format dispatcher"
```

---

## Task 8: Configuration (`config.py`) — paths and HF token

**Files:**
- Create: `src/transcript/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

`tests/test_config.py`:
```python
from pathlib import Path

import pytest

from transcript import config


def test_data_dir_under_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert config.data_dir() == tmp_path / ".local" / "share" / "transcript"


def test_whisper_binary_path(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert config.whisper_binary() == tmp_path / ".local" / "share" / "transcript" / "whisper.cpp" / "main"


def test_whisper_model_path(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert config.whisper_model("large-v3") == (
        tmp_path / ".local" / "share" / "transcript" / "models" / "ggml-large-v3.bin"
    )


def test_hf_token_env_var_takes_precedence(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "from-env")
    assert config.hf_token() == "from-env"


def test_hf_token_falls_back_to_keyring(monkeypatch, mocker):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    mock_get = mocker.patch("transcript.config.keyring.get_password", return_value="from-keychain")
    assert config.hf_token() == "from-keychain"
    mock_get.assert_called_once_with("transcript", "huggingface")


def test_hf_token_missing_raises(monkeypatch, mocker):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    mocker.patch("transcript.config.keyring.get_password", return_value=None)
    with pytest.raises(config.MissingTokenError):
        config.hf_token()
```

- [ ] **Step 2: Run the test to confirm failure**

Run: `uv run pytest tests/test_config.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `config.py`**

`src/transcript/config.py`:
```python
import os
from pathlib import Path

import keyring


class MissingTokenError(RuntimeError):
    """Raised when no HuggingFace token is available."""


_KEYRING_SERVICE = "transcript"
_KEYRING_USER = "huggingface"


def data_dir() -> Path:
    return Path.home() / ".local" / "share" / "transcript"


def whisper_dir() -> Path:
    return data_dir() / "whisper.cpp"


def whisper_binary() -> Path:
    return whisper_dir() / "main"


def models_dir() -> Path:
    return data_dir() / "models"


def whisper_model(name: str) -> Path:
    return models_dir() / f"ggml-{name}.bin"


def whisper_coreml_encoder(name: str) -> Path:
    return models_dir() / f"ggml-{name}-encoder.mlmodelc"


def hf_token() -> str:
    """Return the HuggingFace token. Env var wins; Keychain is fallback."""
    if env := os.environ.get("HF_TOKEN"):
        return env
    if kc := keyring.get_password(_KEYRING_SERVICE, _KEYRING_USER):
        return kc
    raise MissingTokenError(
        "No HuggingFace token found. Set $HF_TOKEN or run scripts/install.sh."
    )
```

- [ ] **Step 4: Run the test to confirm pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/transcript/config.py tests/test_config.py
git commit -m "feat(config): resolve data paths and HF token (env or Keychain)"
```

---

## Task 9: Audio preparation (`audio.py`)

**Files:**
- Create: `src/transcript/audio.py`
- Create: `tests/test_audio.py`

- [ ] **Step 1: Write failing tests**

`tests/test_audio.py`:
```python
from pathlib import Path

import pytest

from transcript import audio


def test_prepare_passes_through_already_correct_wav(tmp_path, mocker):
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"")
    mock_probe = mocker.patch(
        "transcript.audio._probe", return_value={"sample_rate": 16000, "channels": 1, "duration": 5.0}
    )
    mock_run = mocker.patch("transcript.audio.subprocess.run")
    out_path, duration = audio.prepare(wav)
    assert out_path == wav
    assert duration == 5.0
    mock_run.assert_not_called()
    mock_probe.assert_called_once_with(wav)


def test_prepare_converts_m4a_via_ffmpeg(tmp_path, mocker):
    src = tmp_path / "voice.m4a"
    src.write_bytes(b"")
    mocker.patch(
        "transcript.audio._probe", return_value={"sample_rate": 44100, "channels": 2, "duration": 7.5}
    )
    mock_run = mocker.patch("transcript.audio.subprocess.run", return_value=mocker.Mock(returncode=0))
    out_path, duration = audio.prepare(src)
    assert out_path != src
    assert out_path.suffix == ".wav"
    assert duration == 7.5
    args, _ = mock_run.call_args
    cmd = args[0]
    assert cmd[0] == "ffmpeg"
    assert "-ar" in cmd and "16000" in cmd
    assert "-ac" in cmd and "1" in cmd
    assert str(src) in cmd
    assert str(out_path) in cmd


def test_prepare_missing_file_raises(tmp_path):
    with pytest.raises(audio.AudioError, match="not found"):
        audio.prepare(tmp_path / "missing.m4a")


def test_prepare_short_audio_raises(tmp_path, mocker):
    src = tmp_path / "short.wav"
    src.write_bytes(b"")
    mocker.patch(
        "transcript.audio._probe", return_value={"sample_rate": 16000, "channels": 1, "duration": 0.2}
    )
    with pytest.raises(audio.AudioError, match="too short"):
        audio.prepare(src)


def test_ffmpeg_missing_raises(tmp_path, mocker):
    src = tmp_path / "voice.m4a"
    src.write_bytes(b"")
    mocker.patch(
        "transcript.audio._probe", return_value={"sample_rate": 44100, "channels": 2, "duration": 5.0}
    )
    mocker.patch("transcript.audio.subprocess.run", side_effect=FileNotFoundError("ffmpeg"))
    with pytest.raises(audio.AudioError, match="ffmpeg"):
        audio.prepare(src)
```

- [ ] **Step 2: Run the test to confirm failure**

Run: `uv run pytest tests/test_audio.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `audio.py`**

`src/transcript/audio.py`:
```python
import json
import subprocess
import tempfile
from pathlib import Path

MIN_DURATION_S = 0.5


class AudioError(RuntimeError):
    """User-facing audio preparation error."""


def _probe(path: Path) -> dict:
    """Return basic metadata for `path` via ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, check=True, text=True)
    except FileNotFoundError as e:
        raise AudioError("ffprobe not found — install ffmpeg (`brew install ffmpeg`)") from e
    except subprocess.CalledProcessError as e:
        raise AudioError(f"could not read audio: {e.stderr.strip()}") from e

    data = json.loads(result.stdout)
    audio_streams = [s for s in data["streams"] if s["codec_type"] == "audio"]
    if not audio_streams:
        raise AudioError(f"no audio stream in {path}")
    s = audio_streams[0]
    return {
        "sample_rate": int(s["sample_rate"]),
        "channels": int(s["channels"]),
        "duration": float(data["format"]["duration"]),
    }


def prepare(path: Path) -> tuple[Path, float]:
    """Prepare audio for whisper.cpp.

    Returns (wav_path, duration_seconds).
    Passes through if already 16 kHz mono WAV; otherwise converts via ffmpeg
    to a temp WAV that the caller is expected to clean up later
    (we leave it for OS-level temp cleanup since pyannote may also need it).
    """
    if not path.exists():
        raise AudioError(f"audio file not found: {path}")

    info = _probe(path)
    duration = info["duration"]
    if duration < MIN_DURATION_S:
        raise AudioError(f"audio too short to transcribe ({duration:.2f}s)")

    is_correct_wav = (
        path.suffix.lower() == ".wav"
        and info["sample_rate"] == 16000
        and info["channels"] == 1
    )
    if is_correct_wav:
        return path, duration

    out_path = Path(tempfile.mkstemp(suffix=".wav", prefix="transcript-")[1])
    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-y",
        "-i", str(path),
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except FileNotFoundError as e:
        raise AudioError("ffmpeg not found — install with `brew install ffmpeg`") from e
    except subprocess.CalledProcessError as e:
        raise AudioError(f"ffmpeg conversion failed: {e.stderr.decode().strip()}") from e

    return out_path, duration
```

- [ ] **Step 4: Run the test to confirm pass**

Run: `uv run pytest tests/test_audio.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/transcript/audio.py tests/test_audio.py
git commit -m "feat(audio): probe and prepare audio (passthrough or ffmpeg-convert)"
```

---

## Task 10: Progress reporter (`progress.py`)

**Files:**
- Create: `src/transcript/progress.py`
- Create: `tests/test_progress.py`

- [ ] **Step 1: Write failing tests**

`tests/test_progress.py`:
```python
import io

from transcript.progress import Progress


def test_progress_silent_when_quiet():
    buf = io.StringIO()
    p = Progress(verbose=False, quiet=True, stream=buf)
    p.step("doing thing")
    assert buf.getvalue() == ""


def test_progress_compact_when_default():
    buf = io.StringIO()
    p = Progress(verbose=False, quiet=False, stream=buf)
    p.step("preparing audio")
    p.step("transcribing")
    out = buf.getvalue()
    assert "preparing audio" in out
    assert "transcribing" in out


def test_progress_verbose_includes_timing():
    buf = io.StringIO()
    p = Progress(verbose=True, quiet=False, stream=buf)
    p.step("preparing audio")
    p.done("preparing audio")
    out = buf.getvalue()
    assert "preparing audio" in out
    # Verbose mode should print "ok" or similar marker on done()
    assert "ok" in out.lower() or "✓" in out
```

- [ ] **Step 2: Run the test to confirm failure**

Run: `uv run pytest tests/test_progress.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `progress.py`**

`src/transcript/progress.py`:
```python
import sys
import time
from typing import TextIO


class Progress:
    """Tiny stderr progress reporter — no rich/tqdm to keep deps minimal."""

    def __init__(
        self,
        *,
        verbose: bool = False,
        quiet: bool = False,
        stream: TextIO | None = None,
    ) -> None:
        self.verbose = verbose
        self.quiet = quiet
        self.stream = stream if stream is not None else sys.stderr
        self._step_started: dict[str, float] = {}

    def step(self, label: str) -> None:
        if self.quiet:
            return
        self._step_started[label] = time.monotonic()
        prefix = "→" if self.verbose else "·"
        self.stream.write(f"{prefix} {label}\n")
        self.stream.flush()

    def done(self, label: str) -> None:
        if self.quiet or not self.verbose:
            return
        elapsed = time.monotonic() - self._step_started.get(label, time.monotonic())
        self.stream.write(f"  ✓ {label} ok ({elapsed:.1f}s)\n")
        self.stream.flush()

    def warn(self, msg: str) -> None:
        if self.quiet:
            return
        self.stream.write(f"⚠️  {msg}\n")
        self.stream.flush()
```

- [ ] **Step 4: Run the test to confirm pass**

Run: `uv run pytest tests/test_progress.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/transcript/progress.py tests/test_progress.py
git commit -m "feat(progress): add minimal stderr step reporter"
```

---

## Task 11: Whisper.cpp wrapper (`transcribe.py`)

**Files:**
- Create: `src/transcript/transcribe.py`
- Create: `tests/test_transcribe.py`
- Create: `tests/fixtures/whisper_output_sample.json`

- [ ] **Step 1: Create a representative whisper.cpp JSON output fixture**

Create `tests/fixtures/whisper_output_sample.json` with this exact content:
```json
{
  "systeminfo": "fake",
  "model": {"type": "large-v3", "multilingual": true},
  "params": {"model": "ggml-large-v3.bin", "language": "fr"},
  "result": {"language": "fr"},
  "transcription": [
    {
      "timestamps": {"from": "00:00:00,000", "to": "00:00:01,000"},
      "offsets": {"from": 0, "to": 1000},
      "text": " Bonjour",
      "tokens": [
        {"text": " Bonjour", "offsets": {"from": 0, "to": 1000}, "id": 1234}
      ]
    },
    {
      "timestamps": {"from": "00:00:01,000", "to": "00:00:02,000"},
      "offsets": {"from": 1000, "to": 2000},
      "text": " monde",
      "tokens": [
        {"text": " monde", "offsets": {"from": 1000, "to": 2000}, "id": 5678}
      ]
    },
    {
      "timestamps": {"from": "00:00:02,000", "to": "00:00:02,500"},
      "offsets": {"from": 2000, "to": 2500},
      "text": "[_BEG_]",
      "tokens": [
        {"text": "[_BEG_]", "offsets": {"from": 2000, "to": 2500}, "id": 50360}
      ]
    }
  ]
}
```

This fixture mimics the JSON whisper.cpp produces with `--output-json-full -ml 1 --split-on-word`. The third token is a special marker we must filter out.

- [ ] **Step 2: Write failing tests**

`tests/test_transcribe.py`:
```python
import json
import shutil
from pathlib import Path

import pytest

from transcript import transcribe
from transcript.models import Word

FIXTURE = Path(__file__).parent / "fixtures" / "whisper_output_sample.json"


def test_parse_words_extracts_tokens_with_seconds():
    data = json.loads(FIXTURE.read_text())
    words = transcribe._parse_words(data)
    assert words == [
        Word(text=" Bonjour", start=0.0, end=1.0),
        Word(text=" monde", start=1.0, end=2.0),
    ]


def test_parse_words_skips_special_tokens():
    data = json.loads(FIXTURE.read_text())
    words = transcribe._parse_words(data)
    assert all("[_" not in w.text for w in words)


def test_run_invokes_whisper_with_correct_flags(tmp_path, mocker):
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"")
    mocker.patch("transcript.transcribe.config.whisper_binary", return_value=Path("/fake/main"))
    mocker.patch(
        "transcript.transcribe.config.whisper_model", return_value=Path("/fake/ggml-large-v3.bin")
    )
    out_json = tmp_path / "out.json"

    def fake_run(cmd, *args, **kwargs):
        # whisper.cpp writes <output_prefix>.json — emulate that
        json_file = Path(cmd[cmd.index("-of") + 1] + ".json")
        json_file.write_text(FIXTURE.read_text())
        return mocker.Mock(returncode=0)

    mock_run = mocker.patch("transcript.transcribe.subprocess.run", side_effect=fake_run)
    words = transcribe.run(wav, model="large-v3", language="fr")

    assert len(words) == 2
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "/fake/main"
    assert "-m" in cmd and "/fake/ggml-large-v3.bin" in cmd
    assert "-f" in cmd and str(wav) in cmd
    assert "-l" in cmd and "fr" in cmd
    assert "-ml" in cmd and "1" in cmd
    assert "--split-on-word" in cmd
    assert "-ojf" in cmd or "--output-json-full" in cmd


def test_run_auto_language_when_none(tmp_path, mocker):
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"")
    mocker.patch("transcript.transcribe.config.whisper_binary", return_value=Path("/fake/main"))
    mocker.patch(
        "transcript.transcribe.config.whisper_model", return_value=Path("/fake/ggml-large-v3.bin")
    )

    def fake_run(cmd, *args, **kwargs):
        json_file = Path(cmd[cmd.index("-of") + 1] + ".json")
        json_file.write_text(FIXTURE.read_text())
        return mocker.Mock(returncode=0)

    mock_run = mocker.patch("transcript.transcribe.subprocess.run", side_effect=fake_run)
    transcribe.run(wav, model="large-v3", language=None)
    cmd = mock_run.call_args[0][0]
    assert "-l" in cmd and "auto" in cmd


def test_run_missing_binary_raises(tmp_path, mocker):
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"")
    mocker.patch(
        "transcript.transcribe.config.whisper_binary", return_value=Path("/nope/main")
    )
    with pytest.raises(transcribe.TranscribeError, match="not found"):
        transcribe.run(wav, model="large-v3", language="fr")


def test_run_propagates_whisper_failure(tmp_path, mocker):
    import subprocess as sp
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"")
    mocker.patch("transcript.transcribe.config.whisper_binary", return_value=Path("/fake/main"))
    mocker.patch(
        "transcript.transcribe.config.whisper_model", return_value=Path("/fake/ggml-large-v3.bin")
    )
    err = sp.CalledProcessError(1, "main", stderr=b"failed badly")
    mocker.patch("transcript.transcribe.subprocess.run", side_effect=err)
    with pytest.raises(transcribe.TranscribeError, match="failed badly"):
        transcribe.run(wav, model="large-v3", language="fr")
```

- [ ] **Step 3: Run the test to confirm failure**

Run: `uv run pytest tests/test_transcribe.py -v`
Expected: ImportError.

- [ ] **Step 4: Implement `transcribe.py`**

`src/transcript/transcribe.py`:
```python
import json
import subprocess
import tempfile
from pathlib import Path

from transcript import config
from transcript.models import Word


class TranscribeError(RuntimeError):
    """User-facing transcription error."""


def _parse_words(data: dict) -> list[Word]:
    """Pull word-level tokens out of whisper.cpp's --output-json-full payload."""
    words: list[Word] = []
    for segment in data.get("transcription", []):
        for tok in segment.get("tokens", []):
            text: str = tok.get("text", "")
            stripped = text.strip()
            # Skip whisper's special marker tokens like [_BEG_], [_TT_3], etc.
            if not stripped or stripped.startswith("[_"):
                continue
            offsets = tok.get("offsets", {})
            start_ms = int(offsets.get("from", 0))
            end_ms = int(offsets.get("to", 0))
            words.append(Word(text=text, start=start_ms / 1000.0, end=end_ms / 1000.0))
    return words


def run(wav_path: Path, *, model: str, language: str | None) -> list[Word]:
    """Transcribe a 16 kHz mono WAV using whisper.cpp; return word-level Words."""
    binary = config.whisper_binary()
    if not binary.exists():
        raise TranscribeError(
            f"whisper.cpp binary not found at {binary}. Run scripts/install.sh."
        )
    model_path = config.whisper_model(model)
    if not model_path.exists():
        raise TranscribeError(
            f"whisper model {model_path.name} not found. Run scripts/install.sh."
        )

    out_prefix = Path(tempfile.mkdtemp(prefix="transcript-")) / "whisper-out"
    cmd = [
        str(binary),
        "-m", str(model_path),
        "-f", str(wav_path),
        "-l", language or "auto",
        "-ml", "1",
        "--split-on-word",
        "-ojf",
        "-of", str(out_prefix),
        "--no-prints",
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace") if e.stderr else ""
        raise TranscribeError(f"whisper.cpp failed: {stderr.strip()}") from e

    json_file = out_prefix.with_suffix(out_prefix.suffix + ".json")
    if not json_file.exists():
        # whisper.cpp writes <prefix>.json
        json_file = Path(str(out_prefix) + ".json")
    data = json.loads(json_file.read_text())
    return _parse_words(data)
```

- [ ] **Step 5: Run the test to confirm pass**

Run: `uv run pytest tests/test_transcribe.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add src/transcript/transcribe.py tests/test_transcribe.py tests/fixtures/whisper_output_sample.json
git commit -m "feat(transcribe): wrap whisper.cpp subprocess and parse JSON output"
```

---

## Task 12: Pyannote wrapper (`diarize.py`)

**Files:**
- Create: `src/transcript/diarize.py`
- Create: `tests/test_diarize.py`

- [ ] **Step 1: Write failing tests**

`tests/test_diarize.py`:
```python
from pathlib import Path

import pytest

from transcript import diarize
from transcript.models import Turn


class _FakeTrack:
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end


class _FakeAnnotation:
    def __init__(self, items: list[tuple[float, float, str]]):
        self._items = items

    def itertracks(self, yield_label: bool = False):
        for s, e, lbl in self._items:
            if yield_label:
                yield _FakeTrack(s, e), None, lbl
            else:
                yield _FakeTrack(s, e), None


def test_relabel_assigns_speaker_n_in_first_appearance_order():
    fake = _FakeAnnotation([
        (0.0, 1.0, "spk_zzz"),
        (1.0, 2.0, "spk_aaa"),
        (2.0, 3.0, "spk_zzz"),
        (3.0, 4.0, "spk_aaa"),
        (4.0, 5.0, "spk_qqq"),
    ])
    turns = diarize._to_turns(fake)
    assert turns == [
        Turn("Speaker 1", 0.0, 1.0),
        Turn("Speaker 2", 1.0, 2.0),
        Turn("Speaker 1", 2.0, 3.0),
        Turn("Speaker 2", 3.0, 4.0),
        Turn("Speaker 3", 4.0, 5.0),
    ]


def test_run_calls_pipeline_with_num_speakers(mocker, tmp_path):
    wav = tmp_path / "x.wav"; wav.write_bytes(b"")
    mocker.patch("transcript.diarize.config.hf_token", return_value="hf_xxx")
    mock_pipe = mocker.MagicMock()
    mock_pipe.return_value = _FakeAnnotation([(0.0, 1.0, "a")])
    mock_from = mocker.patch(
        "transcript.diarize.Pipeline.from_pretrained", return_value=mock_pipe
    )
    mocker.patch("transcript.diarize.torch.device", side_effect=lambda x: x)
    mocker.patch("transcript.diarize.torch.backends.mps.is_available", return_value=True)

    diarize.run(wav, num_speakers=2)
    mock_from.assert_called_once()
    mock_pipe.to.assert_called_once_with("mps")
    # When num_speakers is set, both min and max are pinned
    _, kwargs = mock_pipe.call_args
    assert kwargs == {"min_speakers": 2, "max_speakers": 2}


def test_run_omits_speaker_kwargs_when_unspecified(mocker, tmp_path):
    wav = tmp_path / "x.wav"; wav.write_bytes(b"")
    mocker.patch("transcript.diarize.config.hf_token", return_value="hf_xxx")
    mock_pipe = mocker.MagicMock()
    mock_pipe.return_value = _FakeAnnotation([])
    mocker.patch("transcript.diarize.Pipeline.from_pretrained", return_value=mock_pipe)
    mocker.patch("transcript.diarize.torch.device", side_effect=lambda x: x)
    mocker.patch("transcript.diarize.torch.backends.mps.is_available", return_value=True)

    diarize.run(wav, num_speakers=None)
    _, kwargs = mock_pipe.call_args
    assert kwargs == {}


def test_run_falls_back_to_cpu_when_mps_unavailable(mocker, tmp_path):
    wav = tmp_path / "x.wav"; wav.write_bytes(b"")
    mocker.patch("transcript.diarize.config.hf_token", return_value="hf_xxx")
    mock_pipe = mocker.MagicMock()
    mock_pipe.return_value = _FakeAnnotation([])
    mocker.patch("transcript.diarize.Pipeline.from_pretrained", return_value=mock_pipe)
    mocker.patch("transcript.diarize.torch.device", side_effect=lambda x: x)
    mocker.patch("transcript.diarize.torch.backends.mps.is_available", return_value=False)

    diarize.run(wav, num_speakers=None)
    mock_pipe.to.assert_called_once_with("cpu")


def test_run_401_raises_actionable_error(mocker, tmp_path):
    wav = tmp_path / "x.wav"; wav.write_bytes(b"")
    mocker.patch("transcript.diarize.config.hf_token", return_value="hf_xxx")

    class FakeHTTPError(Exception):
        def __init__(self):
            self.response = type("R", (), {"status_code": 401})()

    mocker.patch(
        "transcript.diarize.Pipeline.from_pretrained", side_effect=FakeHTTPError()
    )
    with pytest.raises(diarize.DiarizeError, match="license"):
        diarize.run(wav, num_speakers=None)
```

- [ ] **Step 2: Run the test to confirm failure**

Run: `uv run pytest tests/test_diarize.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `diarize.py`**

`src/transcript/diarize.py`:
```python
from pathlib import Path

import torch
from pyannote.audio import Pipeline

from transcript import config
from transcript.models import Turn


class DiarizeError(RuntimeError):
    """User-facing diarization error."""


_PIPELINE_NAME = "pyannote/speaker-diarization-3.1"


def _to_turns(annotation) -> list[Turn]:
    """Relabel pyannote's opaque speaker IDs as Speaker 1, Speaker 2, … in first-appearance order."""
    label_map: dict[str, str] = {}
    turns: list[Turn] = []
    for segment, _track, label in annotation.itertracks(yield_label=True):
        if label not in label_map:
            label_map[label] = f"Speaker {len(label_map) + 1}"
        turns.append(Turn(speaker=label_map[label], start=segment.start, end=segment.end))
    return turns


def run(wav_path: Path, *, num_speakers: int | None) -> list[Turn]:
    """Diarize `wav_path` using pyannote 3.1; return turns relabeled as Speaker N."""
    token = config.hf_token()
    try:
        pipeline = Pipeline.from_pretrained(_PIPELINE_NAME, use_auth_token=token)
    except Exception as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status in (401, 403):
            raise DiarizeError(
                "pyannote refused the download (HTTP {0}). Likely cause: license not accepted.\n"
                "  1. Sign in at https://huggingface.co\n"
                "  2. Click 'Agree' on:\n"
                "     - https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "     - https://huggingface.co/pyannote/segmentation-3.0\n"
                "  3. Re-run the same command.".format(status)
            ) from e
        raise DiarizeError(f"could not load pyannote pipeline: {e}") from e

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipeline.to(torch.device(device))

    kwargs: dict = {}
    if num_speakers is not None:
        kwargs["min_speakers"] = num_speakers
        kwargs["max_speakers"] = num_speakers
    annotation = pipeline(str(wav_path), **kwargs)
    return _to_turns(annotation)
```

- [ ] **Step 4: Run the test to confirm pass**

Run: `uv run pytest tests/test_diarize.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/transcript/diarize.py tests/test_diarize.py
git commit -m "feat(diarize): wrap pyannote with MPS fallback and friendly 401 message"
```

---

## Task 13: Pipeline orchestrator (`pipeline.py`)

**Files:**
- Create: `src/transcript/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests**

`tests/test_pipeline.py`:
```python
from pathlib import Path

import pytest

from transcript import pipeline
from transcript.models import Meta, Turn, Word


def _setup_mocks(mocker, tmp_path):
    wav = tmp_path / "in.m4a"
    wav.write_bytes(b"")
    prepared = tmp_path / "prepared.wav"
    prepared.write_bytes(b"")

    mocker.patch("transcript.pipeline.audio.prepare", return_value=(prepared, 5.0))
    mocker.patch(
        "transcript.pipeline.transcribe.run",
        return_value=[Word(" hi", 0.0, 1.0), Word(" there", 1.0, 2.0)],
    )
    mocker.patch(
        "transcript.pipeline.diarize.run",
        return_value=[Turn("Speaker 1", 0.0, 1.5), Turn("Speaker 2", 1.5, 3.0)],
    )
    return wav


def test_pipeline_returns_rendered_text(tmp_path, mocker):
    wav = _setup_mocks(mocker, tmp_path)
    out = pipeline.run(
        audio_path=wav,
        model="large-v3",
        language="fr",
        diarize=True,
        num_speakers=None,
        format_name="md",
        with_timestamps=True,
    )
    assert "## Speaker 1" in out
    assert "## Speaker 2" in out
    assert "# in.m4a" in out


def test_pipeline_no_diarize_assigns_single_speaker(tmp_path, mocker):
    wav = _setup_mocks(mocker, tmp_path)
    diarize_spy = mocker.patch("transcript.pipeline.diarize.run")
    out = pipeline.run(
        audio_path=wav,
        model="large-v3",
        language="fr",
        diarize=False,
        num_speakers=None,
        format_name="md",
        with_timestamps=True,
    )
    diarize_spy.assert_not_called()
    assert "## Speaker 1" in out
    assert "## Speaker 2" not in out


def test_pipeline_passes_num_speakers_to_diarize(tmp_path, mocker):
    wav = _setup_mocks(mocker, tmp_path)
    diarize_spy = mocker.patch(
        "transcript.pipeline.diarize.run",
        return_value=[Turn("Speaker 1", 0.0, 1.0)],
    )
    pipeline.run(
        audio_path=wav,
        model="large-v3",
        language="fr",
        diarize=True,
        num_speakers=2,
        format_name="md",
        with_timestamps=True,
    )
    _, kwargs = diarize_spy.call_args
    assert kwargs == {"num_speakers": 2}


def test_pipeline_meta_reflects_inputs(tmp_path, mocker):
    wav = _setup_mocks(mocker, tmp_path)
    out = pipeline.run(
        audio_path=wav,
        model="large-v3",
        language="fr",
        diarize=True,
        num_speakers=None,
        format_name="json",
        with_timestamps=True,
    )
    import json
    data = json.loads(out)
    assert data["meta"]["filename"] == "in.m4a"
    assert data["meta"]["model"] == "large-v3"
    assert data["meta"]["language"] == "fr"
    assert data["meta"]["duration"] == 5.0
    assert data["meta"]["speaker_count"] == 2  # two distinct speakers in fixture turns
```

- [ ] **Step 2: Run the test to confirm failure**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `pipeline.py`**

`src/transcript/pipeline.py`:
```python
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from transcript import audio, diarize, formatters, merge, transcribe
from transcript.models import Meta, Turn
from transcript.progress import Progress


def run(
    *,
    audio_path: Path,
    model: str,
    language: str | None,
    diarize: bool,
    num_speakers: int | None,
    format_name: str,
    with_timestamps: bool,
    progress: Progress | None = None,
) -> str:
    progress = progress or Progress(quiet=True)

    progress.step("preparing audio")
    wav, duration = audio.prepare(audio_path)
    progress.done("preparing audio")

    if diarize:
        progress.step("transcribing + diarizing (parallel)")
        with ThreadPoolExecutor(max_workers=2) as ex:
            words_fut = ex.submit(transcribe.run, wav, model=model, language=language)
            turns_fut = ex.submit(diarize_module_run, wav, num_speakers)
            words = words_fut.result()
            turns = turns_fut.result()
        progress.done("transcribing + diarizing (parallel)")
    else:
        progress.step("transcribing")
        words = transcribe.run(wav, model=model, language=language)
        turns = [Turn(speaker="Speaker 1", start=0.0, end=duration)]
        progress.done("transcribing")

    progress.step("merging")
    utterances = merge.assign(words, turns)
    progress.done("merging")

    speaker_count = len({t.speaker for t in turns}) if turns else 0
    meta = Meta(
        filename=audio_path.name,
        duration=duration,
        model=model,
        language=language or "auto",
        speaker_count=speaker_count,
    )

    render = formatters.get(format_name)
    # Only md supports with_timestamps; pass it conditionally
    if format_name == "md":
        return render(utterances, meta, with_timestamps=with_timestamps)
    return render(utterances, meta)


# Indirection so tests can patch `transcript.pipeline.diarize.run`
def diarize_module_run(wav: Path, num_speakers: int | None):
    return diarize.run(wav, num_speakers=num_speakers)
```

- [ ] **Step 4: Run the test to confirm pass**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/transcript/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): orchestrate prepare/transcribe/diarize/merge/render in parallel"
```

---

## Task 14: Doctor (`doctor.py`)

**Files:**
- Create: `src/transcript/doctor.py`
- Create: `tests/test_doctor.py`

- [ ] **Step 1: Write failing tests**

`tests/test_doctor.py`:
```python
from pathlib import Path

from transcript import doctor


def test_doctor_all_green(mocker, tmp_path):
    mocker.patch("transcript.doctor.config.whisper_binary", return_value=tmp_path / "main")
    mocker.patch(
        "transcript.doctor.config.whisper_model", return_value=tmp_path / "ggml-large-v3.bin"
    )
    mocker.patch(
        "transcript.doctor.config.whisper_coreml_encoder",
        return_value=tmp_path / "encoder.mlmodelc",
    )
    (tmp_path / "main").write_bytes(b"")
    (tmp_path / "ggml-large-v3.bin").write_bytes(b"")
    (tmp_path / "encoder.mlmodelc").mkdir()
    mocker.patch("transcript.doctor.shutil.which", return_value="/opt/homebrew/bin/ffmpeg")
    mocker.patch("transcript.doctor.config.hf_token", return_value="hf_xxx")
    mocker.patch("transcript.doctor.torch.backends.mps.is_available", return_value=True)

    code, report = doctor.check()
    assert code == 0
    assert "✓" in report
    assert "✗" not in report


def test_doctor_reports_missing_binary(mocker, tmp_path):
    mocker.patch("transcript.doctor.config.whisper_binary", return_value=tmp_path / "missing")
    mocker.patch(
        "transcript.doctor.config.whisper_model", return_value=tmp_path / "ggml-large-v3.bin"
    )
    (tmp_path / "ggml-large-v3.bin").write_bytes(b"")
    mocker.patch(
        "transcript.doctor.config.whisper_coreml_encoder",
        return_value=tmp_path / "encoder.mlmodelc",
    )
    (tmp_path / "encoder.mlmodelc").mkdir()
    mocker.patch("transcript.doctor.shutil.which", return_value="/opt/homebrew/bin/ffmpeg")
    mocker.patch("transcript.doctor.config.hf_token", return_value="hf_xxx")
    mocker.patch("transcript.doctor.torch.backends.mps.is_available", return_value=True)

    code, report = doctor.check()
    assert code != 0
    assert "✗" in report
    assert "whisper" in report.lower()


def test_doctor_reports_missing_token(mocker, tmp_path):
    from transcript.config import MissingTokenError

    mocker.patch("transcript.doctor.config.whisper_binary", return_value=tmp_path / "main")
    mocker.patch(
        "transcript.doctor.config.whisper_model", return_value=tmp_path / "ggml-large-v3.bin"
    )
    mocker.patch(
        "transcript.doctor.config.whisper_coreml_encoder",
        return_value=tmp_path / "encoder.mlmodelc",
    )
    (tmp_path / "main").write_bytes(b"")
    (tmp_path / "ggml-large-v3.bin").write_bytes(b"")
    (tmp_path / "encoder.mlmodelc").mkdir()
    mocker.patch("transcript.doctor.shutil.which", return_value="/opt/homebrew/bin/ffmpeg")
    mocker.patch("transcript.doctor.config.hf_token", side_effect=MissingTokenError("missing"))
    mocker.patch("transcript.doctor.torch.backends.mps.is_available", return_value=True)

    code, report = doctor.check()
    assert code != 0
    assert "HF" in report or "token" in report.lower()
```

- [ ] **Step 2: Run the test to confirm failure**

Run: `uv run pytest tests/test_doctor.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `doctor.py`**

`src/transcript/doctor.py`:
```python
import shutil

import torch

from transcript import config


def _check(label: str, ok: bool, hint: str | None = None) -> tuple[bool, str]:
    mark = "✓" if ok else "✗"
    line = f"  {mark} {label}"
    if not ok and hint:
        line += f"\n      → {hint}"
    return ok, line


def check() -> tuple[int, str]:
    """Run all boundary checks; return (exit_code, multiline_report)."""
    results: list[tuple[bool, str]] = []

    bin_path = config.whisper_binary()
    results.append(_check(
        f"whisper.cpp binary at {bin_path}",
        bin_path.exists(),
        hint="run scripts/install.sh to clone and build whisper.cpp",
    ))

    model_path = config.whisper_model("large-v3")
    results.append(_check(
        f"whisper model at {model_path}",
        model_path.exists(),
        hint="run scripts/install.sh to download ggml-large-v3.bin",
    ))

    encoder_path = config.whisper_coreml_encoder("large-v3")
    results.append(_check(
        f"CoreML encoder at {encoder_path}",
        encoder_path.exists(),
        hint="run scripts/install.sh to generate the CoreML encoder",
    ))

    results.append(_check(
        "ffmpeg on $PATH",
        shutil.which("ffmpeg") is not None,
        hint="brew install ffmpeg",
    ))

    try:
        config.hf_token()
        token_ok = True
    except config.MissingTokenError:
        token_ok = False
    results.append(_check(
        "HF token in env or Keychain",
        token_ok,
        hint="set $HF_TOKEN or run scripts/install.sh",
    ))

    results.append(_check(
        "PyTorch MPS available",
        torch.backends.mps.is_available(),
        hint="diarization will fall back to CPU (slower)",
    ))

    report_lines = ["transcript --doctor"] + [line for _, line in results]
    code = 0 if all(ok for ok, _ in results) else 1
    return code, "\n".join(report_lines)
```

- [ ] **Step 4: Run the test to confirm pass**

Run: `uv run pytest tests/test_doctor.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/transcript/doctor.py tests/test_doctor.py
git commit -m "feat(doctor): add boundary self-check with green/red report"
```

---

## Task 15: CLI glue (`cli.py`)

**Files:**
- Modify: `src/transcript/cli.py` (replace placeholder from Task 1)
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests**

`tests/test_cli.py`:
```python
from pathlib import Path

import pytest

from transcript import cli


def test_main_no_args_prints_usage_and_exits_2(capsys):
    code = cli.main([])
    captured = capsys.readouterr()
    assert code == 2
    assert "usage" in captured.err.lower() or "usage" in captured.out.lower()


def test_main_version_prints_and_exits(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "0.1.0" in captured.out


def test_main_doctor_invokes_doctor_check(mocker, capsys):
    mocker.patch("transcript.cli.doctor.check", return_value=(0, "all good"))
    code = cli.main(["--doctor"])
    captured = capsys.readouterr()
    assert code == 0
    assert "all good" in captured.out


def test_main_dispatches_to_pipeline_with_defaults(tmp_path, mocker):
    f = tmp_path / "v.m4a"
    f.write_bytes(b"")
    spy = mocker.patch("transcript.cli.pipeline.run", return_value="# ok\n")
    code = cli.main([str(f)])
    assert code == 0
    _, kwargs = spy.call_args
    assert kwargs["audio_path"] == f
    assert kwargs["model"] == "large-v3"
    assert kwargs["language"] is None
    assert kwargs["diarize"] is True
    assert kwargs["num_speakers"] is None
    assert kwargs["format_name"] == "md"
    assert kwargs["with_timestamps"] is True


def test_main_no_diarize_flag(tmp_path, mocker):
    f = tmp_path / "v.m4a"; f.write_bytes(b"")
    spy = mocker.patch("transcript.cli.pipeline.run", return_value="ok")
    cli.main([str(f), "--no-diarize"])
    _, kwargs = spy.call_args
    assert kwargs["diarize"] is False


def test_main_speakers_flag(tmp_path, mocker):
    f = tmp_path / "v.m4a"; f.write_bytes(b"")
    spy = mocker.patch("transcript.cli.pipeline.run", return_value="ok")
    cli.main([str(f), "--speakers", "2"])
    _, kwargs = spy.call_args
    assert kwargs["num_speakers"] == 2


def test_main_writes_to_output_file_when_o_given(tmp_path, mocker):
    f = tmp_path / "v.m4a"; f.write_bytes(b"")
    out = tmp_path / "out.md"
    mocker.patch("transcript.cli.pipeline.run", return_value="# transcript\n")
    code = cli.main([str(f), "-o", str(out)])
    assert code == 0
    assert out.read_text() == "# transcript\n"


def test_main_audio_error_exit_10(tmp_path, mocker, capsys):
    from transcript.audio import AudioError

    f = tmp_path / "v.m4a"; f.write_bytes(b"")
    mocker.patch("transcript.cli.pipeline.run", side_effect=AudioError("file not found"))
    code = cli.main([str(f)])
    captured = capsys.readouterr()
    assert code == 10
    assert "file not found" in captured.err


def test_main_transcribe_error_exit_11(tmp_path, mocker, capsys):
    from transcript.transcribe import TranscribeError

    f = tmp_path / "v.m4a"; f.write_bytes(b"")
    mocker.patch("transcript.cli.pipeline.run", side_effect=TranscribeError("not built"))
    code = cli.main([str(f)])
    captured = capsys.readouterr()
    assert code == 11
    assert "not built" in captured.err


def test_main_diarize_error_exit_12(tmp_path, mocker, capsys):
    from transcript.diarize import DiarizeError

    f = tmp_path / "v.m4a"; f.write_bytes(b"")
    mocker.patch("transcript.cli.pipeline.run", side_effect=DiarizeError("license missing"))
    code = cli.main([str(f)])
    captured = capsys.readouterr()
    assert code == 12
    assert "license missing" in captured.err


def test_main_missing_token_error_exit_12(tmp_path, mocker, capsys):
    from transcript.config import MissingTokenError

    f = tmp_path / "v.m4a"; f.write_bytes(b"")
    mocker.patch("transcript.cli.pipeline.run", side_effect=MissingTokenError("no token"))
    code = cli.main([str(f)])
    captured = capsys.readouterr()
    assert code == 12
    assert "no token" in captured.err
```

- [ ] **Step 2: Run the test to confirm failure**

Run: `uv run pytest tests/test_cli.py -v`
Expected: many failures — placeholder `cli.py` doesn't have the real surface.

- [ ] **Step 3: Replace `cli.py` with the real implementation**

Replace `src/transcript/cli.py`:
```python
import argparse
import sys
from pathlib import Path

from transcript import __version__, doctor, pipeline
from transcript.audio import AudioError
from transcript.config import MissingTokenError
from transcript.diarize import DiarizeError
from transcript.progress import Progress
from transcript.transcribe import TranscribeError

EXIT_OK = 0
EXIT_ERR = 1
EXIT_USAGE = 2
EXIT_AUDIO = 10
EXIT_SETUP = 11
EXIT_AUTH = 12
EXIT_RESOURCE = 13


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="transcript",
        description="Local voice-memo transcription with speaker diarization.",
    )
    p.add_argument("audio_file", nargs="?", type=Path, help="Audio file (.m4a, .mp3, .wav, .mp4, …)")
    p.add_argument("-o", "--output", type=Path, help="Write transcript to file (default: stdout)")
    p.add_argument(
        "-f", "--format",
        choices=["md", "json", "srt", "txt"],
        default="md",
        help="Output format (default: md)",
    )
    p.add_argument("--no-timestamps", action="store_true", help="Omit [mm:ss] markers in markdown")
    p.add_argument("-l", "--language", default=None, help="Language code (default: auto-detect)")
    p.add_argument(
        "-m", "--model",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        default="large-v3",
        help="Whisper model (default: large-v3)",
    )
    p.add_argument("--no-diarize", action="store_true", help="Skip speaker labelling")
    p.add_argument("--speakers", type=int, default=None, help="Fix speaker count when known")
    p.add_argument("-v", "--verbose", action="store_true", help="Show step-by-step progress")
    p.add_argument("-q", "--quiet", action="store_true", help="Suppress all progress")
    p.add_argument("--version", action="version", version=f"transcript {__version__}")
    p.add_argument("--doctor", action="store_true", help="Check setup and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.doctor:
        code, report = doctor.check()
        print(report)
        return code

    if args.audio_file is None:
        parser.print_usage(sys.stderr)
        return EXIT_USAGE

    progress = Progress(verbose=args.verbose, quiet=args.quiet)

    try:
        out = pipeline.run(
            audio_path=args.audio_file,
            model=args.model,
            language=args.language,
            diarize=not args.no_diarize,
            num_speakers=args.speakers,
            format_name=args.format,
            with_timestamps=not args.no_timestamps,
            progress=progress,
        )
    except AudioError as e:
        print(f"✗ {e}", file=sys.stderr)
        return EXIT_AUDIO
    except TranscribeError as e:
        print(f"✗ {e}", file=sys.stderr)
        return EXIT_SETUP
    except (DiarizeError, MissingTokenError) as e:
        print(f"✗ {e}", file=sys.stderr)
        return EXIT_AUTH
    except Exception as e:  # last-resort
        if args.verbose:
            raise
        print(f"✗ unexpected error: {e}", file=sys.stderr)
        return EXIT_ERR

    if args.output:
        args.output.write_text(out)
    else:
        sys.stdout.write(out)
    return EXIT_OK
```

- [ ] **Step 4: Run the test to confirm pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: 11 passed.

- [ ] **Step 5: Run the full unit test suite as a regression check**

Run: `uv run pytest -m "not integration" -v`
Expected: ~50 passed total.

- [ ] **Step 6: Commit**

```bash
git add src/transcript/cli.py tests/test_cli.py
git commit -m "feat(cli): wire argparse to pipeline with explicit exit codes"
```

---

## Task 16: Install script (`scripts/install.sh`)

**Files:**
- Create: `scripts/install.sh`

- [ ] **Step 1: Create the install script**

`scripts/install.sh`:
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

# 6. Install the CLI globally — clone repo if not invoked from inside it
if [[ -f "$(pwd)/pyproject.toml" ]] && grep -q '^name = "transcript-app"' "$(pwd)/pyproject.toml" 2>/dev/null; then
  REPO_DIR="$(pwd)"
else
  REPO_DIR="$DATA_DIR/transcript-app"
  [[ -d "$REPO_DIR" ]] || git clone https://github.com/<you>/transcript-app "$REPO_DIR"
  ( cd "$REPO_DIR" && git pull --ff-only )
fi
uv tool install --from "$REPO_DIR" transcript-app

# 7. Smoke check
echo
transcript --doctor || true   # report-only; doctor exit code != 0 is informative, not fatal here
echo
echo "✅  Done. Try: transcript path/to/voice-memo.m4a"
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x scripts/install.sh`

- [ ] **Step 3: Lint the script with shellcheck (if available)**

Run: `command -v shellcheck && shellcheck scripts/install.sh || echo "shellcheck not installed, skipping"`
Expected: no warnings; or skipped silently.

- [ ] **Step 4: Commit**

```bash
git add scripts/install.sh
git commit -m "feat(install): add idempotent install script for whisper.cpp + token"
```

---

## Task 17: Generate integration fixture and write integration test

**Files:**
- Create: `tests/fixtures/tiny.wav` (generated, but kept under git)
- Create: `tests/test_pipeline_integration.py`
- Create: `scripts/generate_tiny_wav.sh`

- [ ] **Step 1: Create the fixture-generation helper**

`scripts/generate_tiny_wav.sh`:
```bash
#!/usr/bin/env bash
# Generates an 8-second test fixture with two macOS voices alternating.
# Run from project root: bash scripts/generate_tiny_wav.sh
set -euo pipefail

OUT="tests/fixtures/tiny.wav"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Two distinct voices — Thomas (fr) and Amelie (fr) — alternating short phrases
say -v "Thomas"  -o "$TMP/a.aiff" "Bonjour, comment allez-vous aujourd'hui?"
say -v "Amelie"  -o "$TMP/b.aiff" "Très bien merci, et vous?"
say -v "Thomas"  -o "$TMP/c.aiff" "Je vais bien aussi."
say -v "Amelie"  -o "$TMP/d.aiff" "Parfait."

# Concatenate with short gaps and convert to 16 kHz mono WAV
ffmpeg -y \
  -i "$TMP/a.aiff" -i "$TMP/b.aiff" -i "$TMP/c.aiff" -i "$TMP/d.aiff" \
  -filter_complex "[0:a][1:a][2:a][3:a]concat=n=4:v=0:a=1[a]" \
  -map "[a]" -ar 16000 -ac 1 -c:a pcm_s16le \
  "$OUT"

echo "Wrote $OUT ($(du -h "$OUT" | cut -f1))"
```

- [ ] **Step 2: Generate the fixture**

Run: `chmod +x scripts/generate_tiny_wav.sh && bash scripts/generate_tiny_wav.sh`
Expected: writes `tests/fixtures/tiny.wav` (a few hundred KB). If `say -v Thomas` fails because that voice isn't installed, substitute any French voice the user has — `say -v '?' | grep fr` lists available French voices. Document the substitution in a comment in the script.

- [ ] **Step 3: Write the integration test**

`tests/test_pipeline_integration.py`:
```python
import json
import shutil
import sys
from pathlib import Path

import pytest

from transcript import config, pipeline

FIXTURE = Path(__file__).parent / "fixtures" / "tiny.wav"


def _setup_ready() -> tuple[bool, str]:
    if not FIXTURE.exists():
        return False, "tiny.wav fixture not generated (run scripts/generate_tiny_wav.sh)"
    if not config.whisper_binary().exists():
        return False, "whisper.cpp binary not built (run scripts/install.sh)"
    base_model = config.whisper_model("base")
    if not base_model.exists():
        return False, f"whisper 'base' model missing at {base_model} (download via whisper.cpp's download-ggml-model.sh)"
    try:
        config.hf_token()
    except config.MissingTokenError:
        return False, "HF_TOKEN missing"
    return True, ""


@pytest.mark.integration
def test_full_pipeline_against_tiny_wav():
    ok, reason = _setup_ready()
    if not ok:
        pytest.skip(reason)

    out = pipeline.run(
        audio_path=FIXTURE,
        model="base",            # use the small model to keep this test fast
        language="fr",
        diarize=True,
        num_speakers=2,
        format_name="json",
        with_timestamps=True,
    )
    data = json.loads(out)
    # Two French voices — pyannote should detect 2 speakers
    assert data["meta"]["speaker_count"] == 2
    # Some recognisable French content should be present
    text = " ".join(u["text"] for u in data["utterances"]).lower()
    assert any(word in text for word in ("bonjour", "merci", "très", "bien"))


@pytest.mark.integration
def test_no_diarize_returns_single_speaker():
    ok, reason = _setup_ready()
    if not ok:
        pytest.skip(reason)

    out = pipeline.run(
        audio_path=FIXTURE,
        model="base",
        language="fr",
        diarize=False,
        num_speakers=None,
        format_name="json",
        with_timestamps=True,
    )
    data = json.loads(out)
    speakers = {u["speaker"] for u in data["utterances"]}
    assert speakers == {"Speaker 1"}
```

- [ ] **Step 4: Manually run the install script and integration tests**

Run: `bash scripts/install.sh` (this is the long one — clones whisper.cpp, builds it with CoreML, downloads `large-v3`, prompts for HF token).

Then download the `base` model too (faster for tests):
```bash
cd ~/.local/share/transcript/whisper.cpp && \
  bash ./models/download-ggml-model.sh base && \
  mv models/ggml-base.bin ~/.local/share/transcript/models/
```

Then run: `uv run pytest -m integration -v`
Expected: 2 passed (or skipped with a clear reason if any prerequisite is missing).

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_tiny_wav.sh tests/fixtures/tiny.wav tests/test_pipeline_integration.py
git commit -m "test(integration): add tiny.wav fixture and end-to-end pipeline test"
```

---

## Task 18: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write the README**

`README.md`:
````markdown
# transcript

Local voice-memo transcription with speaker diarization on Apple Silicon Macs.
Pairs **whisper.cpp** (CoreML + Metal + ANE) with **pyannote 3.1** (PyTorch MPS) and merges
their output into a markdown transcript. Audio never leaves your machine.

```
$ transcript interview.m4a
# interview.m4a

> Transcribed with whisper.cpp large-v3 (fr) + pyannote 3.1 · 2 speakers · 12m34s

## Speaker 1 [00:00]
Bonjour, nous allons parler de…

## Speaker 2 [00:14]
Oui, exactement, et je pense que…
```

## Quick install

```bash
curl -fsSL https://raw.githubusercontent.com/<you>/transcript-app/main/scripts/install.sh | bash
```

After install: `transcript --doctor` to verify everything is in place.

<details>
<summary>What does the install script actually do?</summary>

1. Checks you're on macOS / Apple Silicon and have Homebrew.
2. Installs `ffmpeg`, `cmake`, `uv` if missing (via brew).
3. Clones `whisper.cpp` to `~/.local/share/transcript/whisper.cpp` and builds it with `WHISPER_COREML=1`.
4. Downloads `ggml-large-v3.bin` (~3 GB) and generates the CoreML encoder.
5. Prompts for your HuggingFace token, stores it in macOS Keychain.
6. Installs the `transcript` CLI globally via `uv tool install`.
7. Runs `transcript --doctor` to confirm.

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
bash ./models/generate-coreml-model.sh large-v3
mkdir -p ~/.local/share/transcript/models
mv models/ggml-large-v3.bin ~/.local/share/transcript/models/
mv models/ggml-large-v3-encoder.mlmodelc ~/.local/share/transcript/models/

# HF token (visit https://huggingface.co/settings/tokens, accept licences on
# pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0)
export HF_TOKEN=hf_xxxxxxxxxx
# or store in Keychain so you don't need the env var:
security add-generic-password -s transcript -a huggingface -w "$HF_TOKEN"

# Install the CLI
git clone https://github.com/<you>/transcript-app
cd transcript-app
uv tool install --from "$(pwd)" transcript-app
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
| `pyannote refused (HTTP 401)` | License not accepted | Sign in to HF, click "Agree" on the two pyannote model pages, re-run |
| `ffmpeg not found` | Missing system dep | `brew install ffmpeg` |
| Hangs at "transcribing" | First diarize run downloading model | Wait — pyannote weights (~100 MB) cache to `~/.cache/huggingface/`, only downloaded once |
| `--doctor` says CoreML encoder missing | Built without `WHISPER_COREML=1` | Re-run install script |

## For Claude Code users

Add this single capability to your Claude Code session and Claude can transcribe voice memos for you. Once installed, Claude can invoke it via the `Bash` tool:

```
You: Transcribe ~/Desktop/meeting.m4a and summarize the action items.
Claude: [runs `transcript ~/Desktop/meeting.m4a` via Bash, reads the markdown output, summarizes]
```

Output goes to stdout, so Claude captures it directly. Use `-f json` if you want Claude to do timestamp-aware reasoning.

## Development

```bash
git clone https://github.com/<you>/transcript-app
cd transcript-app
uv sync --all-extras

# Unit tests (fast, no audio, no network)
uv run pytest -m "not integration"

# Integration tests (require install + HF token)
uv run pytest -m integration
```

## Architecture

See [`docs/superpowers/specs/2026-05-09-transcript-cli-design.md`](docs/superpowers/specs/2026-05-09-transcript-cli-design.md) for the full design.

## Licence

Personal project. whisper.cpp is MIT. pyannote is MIT (model weights have their own gated licence — see HuggingFace).
````

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with quick + manual install, usage, and troubleshooting"
```

---

## Final integration check

Once Tasks 1–18 are complete, run the full quality gate:

```bash
# All unit tests pass
uv run pytest -m "not integration" -v

# Lint clean
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/

# Doctor reports green
transcript --doctor

# Real run on a sample file
transcript path/to/any/voice-memo.m4a --no-diarize --quiet
```

Expected: all unit tests pass, ruff clean, doctor all green, transcript prints valid markdown.

---

## Out-of-scope for this plan (deferred to future iterations)

These were explicitly listed in the spec's "Out of scope" section and should NOT be implemented in this plan:

- Batch processing of a directory
- Configurable diarization backend (`--backend nemo` / `--backend simple`)
- Custom whisper.cpp build flags from CLI
- Always-on rich progress UI
- Speaker-name overrides (`--speaker-1-name "Alice"`)
- Uninstall script
- Linux support
- Word-level timestamps in markdown output
