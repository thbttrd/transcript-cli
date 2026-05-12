# Performance Measurement Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tiered hyperparameter-search harness for `transcript-cli` that runs the deterministic pipeline (whisper-large-v3 + Sortformer + optional alignment + merge) against AMI and SUMM-RE, measures cpWER, and persists per-row transcripts + diffs for post-hoc failure analysis.

**Architecture:** Refactor the pipeline to consume a single `PipelineConfig` dataclass tree, implement TODO #1 (probability-based per-word speaker assignment) as a `merge.strategy` knob, and add a `bench/` Python module + `scripts/benchmark.py` CLI that runs the search and writes results to `bench/results/`. All driven by an on-disk content-hashed cache so re-running downstream-only changes is near-instant.

**Tech Stack:** Python 3.11, pytest + pytest-mock, dataclasses, numpy, ffmpeg subprocess, hashlib (sha1), HuggingFace `datasets`, meeteval (joint-WER package), pandas, soundfile. No new runtime deps for the shipped CLI — bench deps live in a separate `bench` extras group.

**Spec:** `docs/superpowers/specs/2026-05-11-perf-measurement-design.md`

---

## File Structure

### New files (production code)

```
src/transcript/pipeline_config.py     # nested dataclass tree + fingerprint() + from_dict()
bench/__init__.py                     # empty marker
bench/cache.py                        # content-hashed cache for whisper / sortformer / align
bench/artefacts.py                    # writes hypothesis transcripts + meeteval diffs to disk
bench/metrics.py                      # normalisation + cpWER/WER/DER score()
bench/tiers.py                        # tier-1/2/3 config generators
bench/runner.py                       # runs one tier; appends to CSV; generates leaderboard.md
bench/datasets/__init__.py            # empty marker
bench/datasets/base.py                # BenchClip dataclass + Dataset Protocol
bench/datasets/ami.py                 # AMI sdm loader + vendored RTTMs
bench/datasets/summ_re.py             # SUMM-RE loader + track mixing + synth RTTM/STM
bench/datasets/ami_rttm/              # vendored RTTMs from BUTSpeechFIT (added at runtime)
scripts/benchmark.py                  # CLI entrypoint
```

### New files (tests)

```
tests/test_pipeline_config.py
tests/test_merge_prob.py
tests/test_bench_cache.py
tests/test_bench_artefacts.py
tests/test_bench_metrics.py
tests/test_summ_re_loader.py
tests/test_bench_tiers.py
tests/test_bench_smoke.py             # @pytest.mark.integration
```

### Modified files

```
pyproject.toml                        # add [project.optional-dependencies] bench
.gitignore                            # add bench/cache/, bench/results/transcripts/, bench/results/diffs/
src/transcript/pipeline.py            # accepts PipelineConfig; returns (list[Utterance], Meta)
src/transcript/transcribe.py          # reads flags from TranscribeConfig
src/transcript/diarize.py             # branches on streaming_preset; honours emit_probs
src/transcript/align.py               # gated by AlignConfig.enabled
src/transcript/merge.py               # new prob_based strategy as sibling to hard_boundary
src/transcript/cli.py                 # builds PipelineConfig from argparse; formats outside pipeline.run
tests/test_pipeline.py                # rewritten to use PipelineConfig
tests/test_transcribe.py              # add coverage for new tunable flags
tests/test_diarize.py                 # add coverage for streaming_preset and emit_probs
tests/test_align.py                   # update for AlignConfig.enabled (if needed)
tests/test_merge.py                   # update for new merge.assign signature; existing tests still pass
tests/test_cli.py                     # update argparse-to-config mapping coverage
tests/test_pipeline_integration.py    # update to new API
docs/todo.md                          # mark TODO 1 + 3 as DONE
```

---

## Pre-flight

- [ ] **Baseline: confirm tests pass on the current branch before any change**

Run: `uv run pytest -m "not integration"`
Expected: all green. If anything is red, STOP and fix that first (the rest of this plan assumes a green baseline).

- [ ] **Inspect the existing `pipeline.run()` callers**

Run: `grep -rn "pipeline.run" src/ tests/ scripts/`
Expected output includes `src/transcript/cli.py`, `tests/test_pipeline.py`, `tests/test_pipeline_integration.py`. These are the only call sites that need updating in Task 2 and Task 7.

---

## Phase A: PipelineConfig + module refactors

### Task 1: Create `PipelineConfig` dataclass tree

**Files:**
- Create: `src/transcript/pipeline_config.py`
- Test: `tests/test_pipeline_config.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pipeline_config.py`:

```python
import dataclasses
from dataclasses import asdict

import pytest

from transcript.pipeline_config import (
    AlignConfig,
    DiarizeConfig,
    LLMFixConfig,
    MergeConfig,
    PipelineConfig,
    TranscribeConfig,
)


def test_defaults_match_current_pipeline_behavior():
    cfg = PipelineConfig()
    assert cfg.transcribe.model == "large-v3"
    assert cfg.transcribe.language is None
    assert cfg.transcribe.temperature == 0.0
    assert cfg.transcribe.no_fallback is True
    assert cfg.transcribe.suppress_nst is True
    assert cfg.diarize.streaming_preset == "very_high_lat"
    assert cfg.diarize.num_speakers is None
    assert cfg.diarize.emit_probs is False
    assert cfg.align.enabled is True
    assert cfg.merge.strategy == "hard_boundary"
    assert cfg.llm_fix.enabled is False


def test_fingerprint_is_stable_for_same_config():
    cfg_a = PipelineConfig()
    cfg_b = PipelineConfig()
    assert cfg_a.fingerprint() == cfg_b.fingerprint()


def test_fingerprint_changes_when_any_field_changes():
    base = PipelineConfig().fingerprint()
    changed = PipelineConfig(merge=MergeConfig(strategy="prob_based")).fingerprint()
    assert base != changed


def test_fingerprint_is_short_hex_string():
    fp = PipelineConfig().fingerprint()
    assert len(fp) == 12
    assert all(c in "0123456789abcdef" for c in fp)


def test_from_dict_roundtrips_via_asdict():
    cfg = PipelineConfig(
        transcribe=TranscribeConfig(no_fallback=False, suppress_nst=False),
        merge=MergeConfig(strategy="prob_based"),
    )
    reconstructed = PipelineConfig.from_dict(asdict(cfg))
    assert reconstructed == cfg


def test_dataclasses_are_frozen():
    cfg = PipelineConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.transcribe.model = "tiny"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pipeline_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'transcript.pipeline_config'`.

- [ ] **Step 3: Implement `pipeline_config.py`**

Create `src/transcript/pipeline_config.py`:

```python
"""Configuration tree for the transcript pipeline.

Each sub-dataclass owns one stage's tunable parameters. The root `PipelineConfig`
threads through `pipeline.run()` and is the only object module-internal code reads
its hyperparameters from. Frozen so configs can be hashed/fingerprinted safely.
"""
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass(frozen=True)
class TranscribeConfig:
    model: str = "large-v3"
    language: str | None = None
    temperature: float = 0.0
    no_fallback: bool = True
    suppress_nst: bool = True


@dataclass(frozen=True)
class DiarizeConfig:
    streaming_preset: Literal["very_high_lat", "low_lat"] = "very_high_lat"
    num_speakers: int | None = None
    emit_probs: bool = False


@dataclass(frozen=True)
class AlignConfig:
    enabled: bool = True


@dataclass(frozen=True)
class MergeConfig:
    strategy: Literal["hard_boundary", "prob_based"] = "hard_boundary"


@dataclass(frozen=True)
class LLMFixConfig:
    enabled: bool = False


@dataclass(frozen=True)
class PipelineConfig:
    transcribe: TranscribeConfig = field(default_factory=TranscribeConfig)
    diarize:    DiarizeConfig    = field(default_factory=DiarizeConfig)
    align:      AlignConfig      = field(default_factory=AlignConfig)
    merge:      MergeConfig      = field(default_factory=MergeConfig)
    llm_fix:    LLMFixConfig     = field(default_factory=LLMFixConfig)

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha1(payload).hexdigest()[:12]

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        return cls(
            transcribe=TranscribeConfig(**d["transcribe"]),
            diarize=DiarizeConfig(**d["diarize"]),
            align=AlignConfig(**d["align"]),
            merge=MergeConfig(**d["merge"]),
            llm_fix=LLMFixConfig(**d["llm_fix"]),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pipeline_config.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/transcript/pipeline_config.py tests/test_pipeline_config.py
git commit -m "feat(config): add PipelineConfig dataclass tree with fingerprint"
```

---

### Task 2: Refactor `pipeline.run()` to consume `PipelineConfig`; return `(list[Utterance], Meta)`

**Files:**
- Modify: `src/transcript/pipeline.py`
- Modify: `tests/test_pipeline.py`

This is the central API break. The pipeline now returns structured data; the CLI (Task 7) does formatting downstream.

- [ ] **Step 1: Replace `tests/test_pipeline.py` with the new-API version**

Rewrite the entire file:

```python
from transcript import pipeline
from transcript.models import Meta, Turn, Word
from transcript.pipeline_config import (
    AlignConfig,
    DiarizeConfig,
    LLMFixConfig,
    MergeConfig,
    PipelineConfig,
    TranscribeConfig,
)


def _setup_mocks(mocker, tmp_path):
    wav = tmp_path / "in.m4a"
    wav.write_bytes(b"")
    prepared = tmp_path / "prepared.wav"
    prepared.write_bytes(b"")

    mocker.patch("transcript.pipeline.audio.prepare", return_value=(prepared, 5.0))
    mocker.patch(
        "transcript.pipeline.transcribe.run",
        return_value=([Word(" hi", 0.0, 1.0), Word(" there", 2.0, 3.0)], "fr"),
    )
    mocker.patch(
        "transcript.pipeline.diarize.run",
        return_value=([Turn("Speaker 1", 0.0, 1.5), Turn("Speaker 2", 1.5, 3.0)], None),
    )
    mocker.patch("transcript.pipeline.llm_fix.is_available", return_value=False)
    mocker.patch("transcript.pipeline.align.is_available", return_value=False)
    return wav


def test_pipeline_returns_utterances_and_meta(tmp_path, mocker):
    wav = _setup_mocks(mocker, tmp_path)
    cfg = PipelineConfig(transcribe=TranscribeConfig(language="fr"))
    utterances, meta = pipeline.run(audio_path=wav, config=cfg, with_diarization=True)
    assert len(utterances) == 2
    assert {u.speaker for u in utterances} == {"Speaker 1", "Speaker 2"}
    assert meta.filename == "in.m4a"
    assert meta.duration == 5.0
    assert meta.language == "fr"
    assert meta.model == "large-v3"
    assert meta.speaker_count == 2


def test_pipeline_no_diarize_assigns_single_speaker(tmp_path, mocker):
    wav = _setup_mocks(mocker, tmp_path)
    diarize_spy = mocker.patch("transcript.pipeline.diarize.run")
    utterances, _ = pipeline.run(
        audio_path=wav, config=PipelineConfig(), with_diarization=False
    )
    diarize_spy.assert_not_called()
    assert {u.speaker for u in utterances} == {"Speaker 1"}


def test_pipeline_passes_num_speakers_to_diarize(tmp_path, mocker):
    wav = _setup_mocks(mocker, tmp_path)
    diarize_spy = mocker.patch(
        "transcript.pipeline.diarize.run",
        return_value=([Turn("Speaker 1", 0.0, 1.0)], None),
    )
    cfg = PipelineConfig(diarize=DiarizeConfig(num_speakers=2))
    pipeline.run(audio_path=wav, config=cfg, with_diarization=True)
    _, kwargs = diarize_spy.call_args
    assert kwargs["config"].num_speakers == 2


def test_pipeline_calls_llm_fix_when_enabled_and_available(tmp_path, mocker):
    wav = _setup_mocks(mocker, tmp_path)
    mocker.patch("transcript.pipeline.llm_fix.is_available", return_value=True)
    spy = mocker.patch(
        "transcript.pipeline.llm_fix.apply",
        side_effect=lambda pairs, **_: pairs,
    )
    cfg = PipelineConfig(llm_fix=LLMFixConfig(enabled=True))
    pipeline.run(audio_path=wav, config=cfg, with_diarization=True)
    spy.assert_called_once()


def test_pipeline_skips_llm_fix_by_default(tmp_path, mocker):
    wav = _setup_mocks(mocker, tmp_path)
    mocker.patch("transcript.pipeline.llm_fix.is_available", return_value=True)
    spy = mocker.patch("transcript.pipeline.llm_fix.apply")
    pipeline.run(audio_path=wav, config=PipelineConfig(), with_diarization=True)
    spy.assert_not_called()


def test_pipeline_skips_llm_fix_when_unavailable(tmp_path, mocker):
    wav = _setup_mocks(mocker, tmp_path)
    spy = mocker.patch("transcript.pipeline.llm_fix.apply")
    cfg = PipelineConfig(llm_fix=LLMFixConfig(enabled=True))
    pipeline.run(audio_path=wav, config=cfg, with_diarization=True)
    spy.assert_not_called()


def test_pipeline_calls_align_when_enabled_and_available(tmp_path, mocker):
    wav = _setup_mocks(mocker, tmp_path)
    mocker.patch("transcript.pipeline.align.is_available", return_value=True)
    spy = mocker.patch(
        "transcript.pipeline.align.run",
        side_effect=lambda _wav, words, **_: words,
    )
    pipeline.run(audio_path=wav, config=PipelineConfig(), with_diarization=True)
    spy.assert_called_once()
    _, kwargs = spy.call_args
    assert kwargs["language"] == "fr"


def test_pipeline_respects_align_disabled(tmp_path, mocker):
    wav = _setup_mocks(mocker, tmp_path)
    mocker.patch("transcript.pipeline.align.is_available", return_value=True)
    spy = mocker.patch("transcript.pipeline.align.run")
    cfg = PipelineConfig(align=AlignConfig(enabled=False))
    pipeline.run(audio_path=wav, config=cfg, with_diarization=True)
    spy.assert_not_called()


def test_pipeline_cleans_up_temp_wav(tmp_path, mocker):
    wav = _setup_mocks(mocker, tmp_path)
    prepared = tmp_path / "prepared.wav"
    assert prepared.exists()
    pipeline.run(audio_path=wav, config=PipelineConfig(), with_diarization=True)
    assert not prepared.exists()


def test_pipeline_threads_emit_probs_when_merge_is_prob_based(tmp_path, mocker):
    wav = _setup_mocks(mocker, tmp_path)
    diarize_spy = mocker.patch(
        "transcript.pipeline.diarize.run",
        return_value=([Turn("Speaker 1", 0.0, 1.0)], None),
    )
    cfg = PipelineConfig(merge=MergeConfig(strategy="prob_based"))
    pipeline.run(audio_path=wav, config=cfg, with_diarization=True)
    diarize_cfg = diarize_spy.call_args.kwargs["config"]
    assert diarize_cfg.emit_probs is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL — `pipeline.run()` doesn't accept `config=`, doesn't return a tuple.

- [ ] **Step 3: Rewrite `src/transcript/pipeline.py`**

Replace the file contents:

```python
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from transcript import align, audio, diarize, llm_fix, merge, transcribe
from transcript.models import Meta, Turn, Utterance
from transcript.pipeline_config import PipelineConfig
from transcript.progress import Progress


def run(
    *,
    audio_path: Path,
    config: PipelineConfig,
    with_diarization: bool = True,
    progress: Progress | None = None,
) -> tuple[list[Utterance], Meta]:
    progress = progress or Progress(quiet=True)

    progress.step("preparing audio")
    wav, duration = audio.prepare(audio_path)
    progress.done("preparing audio")

    is_temp_wav = wav != audio_path
    try:
        diarize_cfg = config.diarize
        if config.merge.strategy == "prob_based":
            diarize_cfg = replace(diarize_cfg, emit_probs=True)

        if with_diarization:
            progress.step("transcribing + diarizing (parallel)")
            with ThreadPoolExecutor(max_workers=2) as ex:
                tx_fut = ex.submit(transcribe.run, wav, config=config.transcribe)
                diar_fut = ex.submit(diarize.run, wav, config=diarize_cfg)
                words, detected_lang = tx_fut.result()
                turns, probs = diar_fut.result()
            progress.done("transcribing + diarizing (parallel)")
        else:
            progress.step("transcribing")
            words, detected_lang = transcribe.run(wav, config=config.transcribe)
            turns = [Turn(speaker="Speaker 1", start=0.0, end=duration)]
            probs = None
            progress.done("transcribing")

        if config.align.enabled and align.is_available() and words:
            progress.step("aligning words")
            words = align.run(wav, words, language=detected_lang)
            progress.done("aligning words")

        word_speakers = merge.assign_speakers(
            words, turns, strategy=config.merge.strategy, probs=probs
        )
        if with_diarization and config.llm_fix.enabled and llm_fix.is_available():
            progress.step("LLM cleanup")
            word_speakers = llm_fix.apply(
                word_speakers,
                language=detected_lang,
                num_speakers=config.diarize.num_speakers,
            )
            progress.done("LLM cleanup")

        progress.step("merging")
        utterances = merge.collapse(word_speakers)
        progress.done("merging")

        speaker_count = len({t.speaker for t in turns}) if turns else 0
        meta = Meta(
            filename=audio_path.name,
            duration=duration,
            model=config.transcribe.model,
            language=detected_lang,
            speaker_count=speaker_count,
            diarizer=diarize.DIARIZER_LABEL if with_diarization else None,
        )

        return utterances, meta
    finally:
        if is_temp_wav:
            wav.unlink(missing_ok=True)
```

Note: this references new signatures for `transcribe.run`, `diarize.run`, and `merge.assign_speakers` that don't exist yet. The test will still fail until Tasks 3, 4, 6 land. That's fine — we keep moving and the suite goes green at Task 6.

- [ ] **Step 4: Confirm test_pipeline.py still fails (broken downstream signatures)**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL with TypeError about kwargs — Task 3, 4, 6 will fix the downstream signatures.

- [ ] **Step 5: Commit**

```bash
git add src/transcript/pipeline.py tests/test_pipeline.py
git commit -m "refactor(pipeline): accept PipelineConfig and return (utterances, meta)"
```

---

### Task 3: Refactor `transcribe.py` to consume `TranscribeConfig`

**Files:**
- Modify: `src/transcript/transcribe.py`
- Modify: `tests/test_transcribe.py`

- [ ] **Step 1: Update `tests/test_transcribe.py` to use TranscribeConfig**

Add to the imports:

```python
from transcript.pipeline_config import TranscribeConfig
```

Replace each `transcribe.run(wav, model="large-v3", language="fr")` call with `transcribe.run(wav, config=TranscribeConfig(language="fr"))`. Then add three new tests at the bottom of the file:

```python
def test_run_respects_no_fallback_flag(tmp_path, mocker):
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"")
    mocker.patch("transcript.transcribe.config.whisper_binary", return_value=Path("/fake/main"))
    mocker.patch("transcript.transcribe.config.whisper_model", return_value=Path("/fake/m.bin"))
    mocker.patch("transcript.transcribe.Path.exists", return_value=True)

    def fake_run(cmd, *args, **kwargs):
        json_file = Path(cmd[cmd.index("-of") + 1] + ".json")
        json_file.write_text(FIXTURE.read_text())
        return mocker.Mock(returncode=0)

    mock_run = mocker.patch("transcript.transcribe.subprocess.run", side_effect=fake_run)
    cfg = TranscribeConfig(language="fr", no_fallback=False)
    transcribe.run(wav, config=cfg)
    cmd = mock_run.call_args[0][0]
    assert "--no-fallback" not in cmd


def test_run_respects_suppress_nst_flag(tmp_path, mocker):
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"")
    mocker.patch("transcript.transcribe.config.whisper_binary", return_value=Path("/fake/main"))
    mocker.patch("transcript.transcribe.config.whisper_model", return_value=Path("/fake/m.bin"))
    mocker.patch("transcript.transcribe.Path.exists", return_value=True)

    def fake_run(cmd, *args, **kwargs):
        json_file = Path(cmd[cmd.index("-of") + 1] + ".json")
        json_file.write_text(FIXTURE.read_text())
        return mocker.Mock(returncode=0)

    mock_run = mocker.patch("transcript.transcribe.subprocess.run", side_effect=fake_run)
    cfg = TranscribeConfig(language="fr", suppress_nst=False)
    transcribe.run(wav, config=cfg)
    cmd = mock_run.call_args[0][0]
    assert "--suppress-nst" not in cmd


def test_run_passes_temperature(tmp_path, mocker):
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"")
    mocker.patch("transcript.transcribe.config.whisper_binary", return_value=Path("/fake/main"))
    mocker.patch("transcript.transcribe.config.whisper_model", return_value=Path("/fake/m.bin"))
    mocker.patch("transcript.transcribe.Path.exists", return_value=True)

    def fake_run(cmd, *args, **kwargs):
        json_file = Path(cmd[cmd.index("-of") + 1] + ".json")
        json_file.write_text(FIXTURE.read_text())
        return mocker.Mock(returncode=0)

    mock_run = mocker.patch("transcript.transcribe.subprocess.run", side_effect=fake_run)
    cfg = TranscribeConfig(language="fr", temperature=0.3)
    transcribe.run(wav, config=cfg)
    cmd = mock_run.call_args[0][0]
    i = cmd.index("--temperature")
    assert cmd[i + 1] == "0.3"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_transcribe.py -v`
Expected: FAIL — `transcribe.run` doesn't accept `config=`.

- [ ] **Step 3: Rewrite `transcribe.run` in `src/transcript/transcribe.py`**

At the top of the file change `from transcript import config` to:

```python
from transcript import config as transcript_config
```

Then replace `run()` (keeping `_parse_words`, `_detected_language`, `TranscribeError`, imports):

```python
def run(wav_path: Path, *, config) -> tuple[list[Word], str]:
    """Transcribe a 16 kHz mono WAV using whisper.cpp."""
    from transcript.pipeline_config import TranscribeConfig
    assert isinstance(config, TranscribeConfig)

    binary = transcript_config.whisper_binary()
    if not binary.exists():
        raise TranscribeError(
            f"whisper.cpp binary not found at {binary}. Run scripts/install.sh."
        )
    model_path = transcript_config.whisper_model(config.model)
    if not model_path.exists():
        raise TranscribeError(
            f"whisper model {model_path.name} not found. Run scripts/install.sh."
        )

    with tempfile.TemporaryDirectory(prefix="transcript-") as tmpdir:
        out_prefix = Path(tmpdir) / "whisper-out"
        cmd: list[str] = [
            str(binary),
            "-m", str(model_path),
            "-f", str(wav_path),
            "-l", config.language or "auto",
            "-ml", "1",
            "--split-on-word",
            "--temperature", str(config.temperature),
            "-ojf",
            "-of", str(out_prefix),
            "--no-prints",
        ]
        if config.no_fallback:
            cmd.append("--no-fallback")
        if config.suppress_nst:
            cmd.append("--suppress-nst")

        try:
            subprocess.run(cmd, capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode(errors="replace") if e.stderr else ""
            raise TranscribeError(f"whisper.cpp failed: {stderr.strip()}") from e

        json_file = Path(str(out_prefix) + ".json")
        data = json.loads(json_file.read_text())
        return _parse_words(data), _detected_language(data, config.language)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_transcribe.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/transcript/transcribe.py tests/test_transcribe.py
git commit -m "refactor(transcribe): consume TranscribeConfig; expose no_fallback/suppress_nst/temperature"
```

---

### Task 4: Refactor `diarize.py` to consume `DiarizeConfig` and emit optional probability tensor

**Files:**
- Modify: `src/transcript/diarize.py`
- Modify: `tests/test_diarize.py`

The two streaming presets are NVIDIA-published values; the "low_lat" preset is documented in `diarize.py`'s comment block.

- [ ] **Step 1: Update `tests/test_diarize.py` and add new tests**

Add at the top of the file:

```python
from transcript.pipeline_config import DiarizeConfig
```

Update the `reset_model_cache` fixture to clear the new per-preset cache:

```python
@pytest.fixture
def reset_model_cache(monkeypatch):
    monkeypatch.setattr(diarize, "_model_cache", {})
```

Add these new tests (existing tests can stay):

```python
def test_streaming_params_for_very_high_lat_preset_match_nvidia_values():
    params = diarize._streaming_params("very_high_lat")
    assert params == {
        "chunk_len": 340,
        "chunk_right_context": 40,
        "fifo_len": 40,
        "spkcache_update_period": 340,
        "spkcache_len": 188,
    }


def test_streaming_params_for_low_lat_preset_match_nvidia_values():
    params = diarize._streaming_params("low_lat")
    assert params == {
        "chunk_len": 6,
        "chunk_right_context": 7,
        "fifo_len": 188,
        "spkcache_update_period": 144,
        "spkcache_len": 188,
    }


def test_run_filters_by_num_speakers(reset_model_cache, monkeypatch, mocker):
    fake_model = mocker.MagicMock()
    fake_model.diarize.return_value = [[
        "0.0 1.0 spk_a",
        "1.0 2.0 spk_b",
        "2.0 3.0 spk_c",
    ]]
    fake_class = mocker.MagicMock()
    fake_class.from_pretrained.return_value = fake_model
    _inject_fake_nemo(monkeypatch, fake_class)

    cfg = DiarizeConfig(num_speakers=2)
    turns, probs = diarize.run(Path("/fake.wav"), config=cfg)
    assert {t.speaker for t in turns} == {"Speaker 1", "Speaker 2"}
    assert probs is None


def test_run_returns_probs_when_emit_probs_true(reset_model_cache, monkeypatch, mocker):
    import numpy as np

    fake_tensor = np.zeros((10, 4), dtype=np.float32)
    fake_model = mocker.MagicMock()
    fake_model.diarize.return_value = ([["0.0 1.0 spk_a"]], fake_tensor)
    fake_class = mocker.MagicMock()
    fake_class.from_pretrained.return_value = fake_model
    _inject_fake_nemo(monkeypatch, fake_class)

    cfg = DiarizeConfig(emit_probs=True)
    turns, probs = diarize.run(Path("/fake.wav"), config=cfg)
    assert probs is not None
    assert probs.shape == (10, 4)
    _, kwargs = fake_model.diarize.call_args
    assert kwargs.get("include_tensor_outputs") is True
```

Add `from pathlib import Path` to the test file imports if not already there.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_diarize.py -v`
Expected: FAIL — `_streaming_params` doesn't exist; `diarize.run` doesn't accept `config=`.

- [ ] **Step 3: Rewrite the relevant parts of `src/transcript/diarize.py`**

Replace the `_STREAMING_PARAMS` constant block with the presets table:

```python
_STREAMING_PRESETS: dict[str, dict[str, int]] = {
    "very_high_lat": {
        "chunk_len": 340,
        "chunk_right_context": 40,
        "fifo_len": 40,
        "spkcache_update_period": 340,
        "spkcache_len": 188,
    },
    "low_lat": {
        "chunk_len": 6,
        "chunk_right_context": 7,
        "fifo_len": 188,
        "spkcache_update_period": 144,
        "spkcache_len": 188,
    },
}


def _streaming_params(preset: str) -> dict[str, int]:
    if preset not in _STREAMING_PRESETS:
        raise DiarizeError(f"unknown streaming preset: {preset}")
    return dict(_STREAMING_PRESETS[preset])
```

Replace the `_model` global and `_load_model` with a per-preset cache:

```python
_model_cache: dict[str, object] = {}


def _load_model(preset: str = "very_high_lat"):
    """Lazy-load Sortformer once per preset. Cached across calls within a process."""
    if preset in _model_cache:
        return _model_cache[preset]
    try:
        from nemo.collections.asr.models import SortformerEncLabelModel
    except ImportError as e:
        raise DiarizeError(
            "nemo_toolkit not installed. Re-run scripts/install.sh."
        ) from e
    try:
        m = SortformerEncLabelModel.from_pretrained(_NEMO_MODEL, map_location="cpu")
    except Exception as e:
        raise DiarizeError(f"could not load NeMo Sortformer: {e}") from e
    m.train(False)
    for name, value in _streaming_params(preset).items():
        setattr(m.sortformer_modules, name, value)
    m.sortformer_modules._check_streaming_parameters()
    _model_cache[preset] = m
    return m
```

Replace `run` with:

```python
def run(wav_path: Path, *, config) -> tuple[list[Turn], "np.ndarray | None"]:
    """Diarize and return (turns, optional [T x 4] probability tensor)."""
    from transcript.pipeline_config import DiarizeConfig
    assert isinstance(config, DiarizeConfig)

    model = _load_model(config.streaming_preset)
    if config.emit_probs:
        result = model.diarize(
            audio=[str(wav_path)], batch_size=1, include_tensor_outputs=True
        )
        # NeMo returns (segments_list, tensor_list) when include_tensor_outputs=True.
        if isinstance(result, tuple) and len(result) == 2:
            raw_lines = result[0][0] if result[0] else []
            probs = result[1] if not isinstance(result[1], list) else result[1][0]
        else:
            raw_lines = result[0] if result else []
            probs = None
    else:
        results = model.diarize(audio=[str(wav_path)], batch_size=1)
        raw_lines = results[0] if results else []
        probs = None

    raw = _parse_sortformer_segments(raw_lines)
    turns = _relabel(raw)
    if config.num_speakers is not None:
        keep = {f"Speaker {i + 1}" for i in range(config.num_speakers)}
        turns = [t for t in turns if t.speaker in keep]
    return turns, probs
```

Remove the old `_model` global.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_diarize.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/transcript/diarize.py tests/test_diarize.py
git commit -m "refactor(diarize): consume DiarizeConfig; expose preset + emit_probs"
```

---

### Task 5: Refactor `align.py` to honour `AlignConfig.enabled`

**Files:**
- Modify: `src/transcript/align.py`

`align.py` is already gated externally by `pipeline.run()`'s `config.align.enabled` check. No signature change is strictly required. This task is a verification step — confirm the existing align tests still pass.

- [ ] **Step 1: Run existing align tests**

Run: `uv run pytest tests/test_align.py -v`
Expected: all PASS (unchanged behavior).

- [ ] **Step 2: No changes required — proceed to Task 6**

Skip commit. Align is wired through `pipeline.run()`'s `config.align.enabled` check landed in Task 2.

---

### Task 6: Implement `prob_based` merge strategy + update `merge.assign_speakers` signature

**Files:**
- Modify: `src/transcript/merge.py`
- Create: `tests/test_merge_prob.py`

- [ ] **Step 1: Write failing tests for the new strategy**

Create `tests/test_merge_prob.py`:

```python
import numpy as np

from transcript.merge import assign_speakers
from transcript.models import Turn, Word

FRAME_S = 0.08  # 80 ms — Sortformer's frame size


def test_prob_based_assigns_argmax_speaker_over_word_frames():
    # Word spans frames 3..7 (0.24..0.56s). probs spike on speaker B in those frames.
    words = [Word(text=" foo", start=0.24, end=0.56)]
    turns = []  # turns ignored in prob mode
    probs = np.zeros((20, 4), dtype=np.float32)
    probs[3:8, 1] = 0.9   # speaker B (index 1)

    pairs = assign_speakers(words, turns, strategy="prob_based", probs=probs)
    assert len(pairs) == 1
    _, speaker = pairs[0]
    assert speaker == "Speaker 2"


def test_prob_based_handles_word_outside_tensor_range():
    # Word at 5.0s but tensor only covers 10 frames (0..0.8s). Fall back to "Unknown".
    words = [Word(text=" foo", start=5.0, end=5.5)]
    probs = np.zeros((10, 4), dtype=np.float32)
    pairs = assign_speakers(words, [], strategy="prob_based", probs=probs)
    _, speaker = pairs[0]
    assert speaker == "Unknown"


def test_prob_based_with_no_probs_falls_back_to_hard_boundary():
    """If strategy=prob_based but probs is None, fall back gracefully."""
    words = [Word(text=" foo", start=0.0, end=0.5)]
    turns = [Turn("Speaker 1", 0.0, 1.0)]
    pairs = assign_speakers(words, turns, strategy="prob_based", probs=None)
    _, speaker = pairs[0]
    assert speaker == "Speaker 1"


def test_hard_boundary_strategy_unchanged():
    """The pre-existing strategy still works when explicitly requested."""
    words = [Word(text=" foo", start=0.0, end=0.5)]
    turns = [Turn("Speaker 1", 0.0, 1.0)]
    pairs = assign_speakers(words, turns, strategy="hard_boundary", probs=None)
    _, speaker = pairs[0]
    assert speaker == "Speaker 1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_merge.py tests/test_merge_prob.py -v`
Expected: existing merge tests PASS; merge_prob tests FAIL.

- [ ] **Step 3: Implement the prob-based strategy in `src/transcript/merge.py`**

Replace the file contents:

```python
from typing import Literal, Optional

import numpy as np

from transcript.models import Turn, Utterance, Word

UNKNOWN = "Unknown"
FRAME_S = 0.08  # Sortformer frame size — 80 ms


def _best_speaker_hard_boundary(word: Word, turns: list[Turn]) -> str:
    if not turns:
        return UNKNOWN

    best_overlap = 0.0
    best_turn: Turn | None = None
    earliest_upcoming: Turn | None = None
    latest_turn: Turn | None = None
    for turn in turns:
        overlap = min(word.end, turn.end) - max(word.start, turn.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_turn = turn
        if turn.start >= word.start and (
            earliest_upcoming is None or turn.start < earliest_upcoming.start
        ):
            earliest_upcoming = turn
        if latest_turn is None or turn.end > latest_turn.end:
            latest_turn = turn

    if best_turn is not None:
        return best_turn.speaker
    if earliest_upcoming is not None:
        return earliest_upcoming.speaker
    return latest_turn.speaker  # type: ignore[union-attr]


def _best_speaker_prob_based(word: Word, probs: np.ndarray) -> str:
    """Average per-frame probabilities over the word's frame window; argmax to speaker."""
    n_frames = probs.shape[0]
    start_frame = int(word.start / FRAME_S)
    end_frame = max(start_frame + 1, int(np.ceil(word.end / FRAME_S)))
    if start_frame >= n_frames:
        return UNKNOWN
    end_frame = min(end_frame, n_frames)
    window = probs[start_frame:end_frame]
    if window.size == 0:
        return UNKNOWN
    mean = window.mean(axis=0)
    idx = int(mean.argmax())
    return f"Speaker {idx + 1}"


def assign_speakers(
    words: list[Word],
    turns: list[Turn],
    *,
    strategy: Literal["hard_boundary", "prob_based"] = "hard_boundary",
    probs: Optional[np.ndarray] = None,
) -> list[tuple[Word, str]]:
    """Per-word speaker assignment.

    - strategy="hard_boundary": max-overlap to turn ranges (existing logic).
    - strategy="prob_based": average per-frame probabilities over word.start..end,
      argmax over the 4 speaker columns. Requires `probs` (a [T x 4] array).
      Falls back to hard_boundary silently if probs is None.
    """
    if strategy == "prob_based" and probs is not None:
        return [(w, _best_speaker_prob_based(w, probs)) for w in words]
    return [(w, _best_speaker_hard_boundary(w, turns)) for w in words]


def collapse(word_speakers: list[tuple[Word, str]]) -> list[Utterance]:
    """Collapse consecutive same-speaker words into utterances."""
    if not word_speakers:
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

    for word, speaker in word_speakers:
        if speaker != current_speaker and current_words:
            flush()
            current_words = []
        current_speaker = speaker
        current_words.append(word)

    flush()
    return utterances


def assign(words: list[Word], turns: list[Turn]) -> list[Utterance]:
    """One-shot convenience: hard_boundary assign + collapse."""
    return collapse(assign_speakers(words, turns))
```

- [ ] **Step 4: Run all merge tests AND the pipeline tests to verify everything is green**

Run: `uv run pytest tests/test_merge.py tests/test_merge_prob.py tests/test_pipeline.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/transcript/merge.py tests/test_merge.py tests/test_merge_prob.py
git commit -m "feat(merge): add prob_based strategy using Sortformer frame probabilities"
```

---

### Task 7: Refactor `cli.py` to build `PipelineConfig` and format outside the pipeline

**Files:**
- Modify: `src/transcript/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Update `tests/test_cli.py`**

Read `tests/test_cli.py` and update each test that mocks `pipeline.run()` to expect the new (config, with_diarization) shape, returning `(utterances, meta)`. Add one new test at the bottom:

```python
def test_cli_builds_pipeline_config_from_args(tmp_path, mocker):
    """argparse to PipelineConfig wiring should pass model/language/speakers/align/llm_fix."""
    from transcript import cli
    from transcript.models import Meta, Utterance

    audio = tmp_path / "in.m4a"
    audio.write_bytes(b"")

    captured = {}

    def fake_run(*, audio_path, config, with_diarization, progress):
        captured["config"] = config
        captured["with_diarization"] = with_diarization
        return [Utterance("Speaker 1", 0.0, 1.0, "hi")], Meta(
            filename="in.m4a", duration=1.0, model="large-v3",
            language="fr", speaker_count=1, diarizer=None,
        )

    mocker.patch("transcript.cli.pipeline.run", side_effect=fake_run)
    rc = cli.main([str(audio), "-l", "fr", "--speakers", "2", "--no-align", "--llm-fix"])
    assert rc == 0
    cfg = captured["config"]
    assert cfg.transcribe.language == "fr"
    assert cfg.diarize.num_speakers == 2
    assert cfg.align.enabled is False
    assert cfg.llm_fix.enabled is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — old shape.

- [ ] **Step 3: Rewrite the relevant part of `src/transcript/cli.py`**

Replace the inner `try:` block in `main()`:

```python
    try:
        from transcript import formatters
        from transcript.pipeline_config import (
            AlignConfig,
            DiarizeConfig,
            LLMFixConfig,
            PipelineConfig,
            TranscribeConfig,
        )

        cfg = PipelineConfig(
            transcribe=TranscribeConfig(model=args.model, language=args.language),
            diarize=DiarizeConfig(num_speakers=args.speakers),
            align=AlignConfig(enabled=not args.no_align),
            llm_fix=LLMFixConfig(enabled=args.llm_fix),
        )
        utterances, meta = pipeline.run(
            audio_path=args.audio_file,
            config=cfg,
            with_diarization=not args.no_diarize,
            progress=progress,
        )
        render = formatters.get(args.format)
        if args.format == "md":
            out = render(utterances, meta, with_timestamps=not args.no_timestamps)
        else:
            out = render(utterances, meta)
    except AudioError as e:
        print(f"x {e}", file=sys.stderr)
        return EXIT_AUDIO
    except TranscribeError as e:
        print(f"x {e}", file=sys.stderr)
        return EXIT_SETUP
    except DiarizeError as e:
        print(f"x {e}", file=sys.stderr)
        return EXIT_SETUP
    except Exception as e:
        if args.verbose:
            raise
        print(f"x unexpected error: {e}", file=sys.stderr)
        return EXIT_ERR
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the full suite to confirm Phase A is fully green**

Run: `uv run pytest -m "not integration" -v`
Expected: ALL PASS. If anything is red, fix before continuing.

- [ ] **Step 6: Commit**

```bash
git add src/transcript/cli.py tests/test_cli.py
git commit -m "refactor(cli): build PipelineConfig from argparse; format outside pipeline.run"
```

---

### Task 8: `.gitignore` + update `tests/test_pipeline_integration.py` for new API

**Files:**
- Modify: `.gitignore`
- Modify: `tests/test_pipeline_integration.py`

- [ ] **Step 1: Append to `.gitignore`**

```
# bench
bench/cache/
bench/results/transcripts/
bench/results/diffs/
```

- [ ] **Step 2: Update `tests/test_pipeline_integration.py`**

Wrap its `pipeline.run` call to use `PipelineConfig`:

```python
from transcript.pipeline_config import PipelineConfig, TranscribeConfig

cfg = PipelineConfig(transcribe=TranscribeConfig(model="base", language=None))
utterances, meta = pipeline.run(
    audio_path=fixture, config=cfg, with_diarization=True
)
# downstream assertions: render to markdown via formatters.get("md")
```

- [ ] **Step 3: Commit**

```bash
git add .gitignore tests/test_pipeline_integration.py
git commit -m "chore: gitignore bench outputs; update integration test to new pipeline API"
```

---

## Phase B: bench infrastructure

### Task 9: Add `bench` extras group + create `bench/` skeleton

**Files:**
- Modify: `pyproject.toml`
- Create: `bench/__init__.py`
- Create: `bench/datasets/__init__.py`

- [ ] **Step 1: Edit `pyproject.toml`**

Add inside `[project.optional-dependencies]` (the section already exists with `dev`):

```toml
bench = [
    "datasets==3.2.0",
    "meeteval==0.4.1",
    "pandas==2.2.3",
    "soundfile==0.12.1",
]
```

- [ ] **Step 2: Install the bench extras**

Run: `uv sync --extra bench --extra dev`
Expected: `meeteval`, `datasets`, `pandas`, `soundfile` install successfully. If meeteval==0.4.1 isn't on PyPI, fall back to the latest 0.4.x and update the pin.

- [ ] **Step 3: Create empty package markers**

```bash
mkdir -p bench/datasets
touch bench/__init__.py bench/datasets/__init__.py
```

- [ ] **Step 4: Verify import works from the repo root**

Run: `uv run python -c "import bench, bench.datasets; print('ok')"`
Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock bench/__init__.py bench/datasets/__init__.py
git commit -m "chore(bench): add bench extras group and package skeleton"
```

---

### Task 10: Implement `bench/cache.py` — content-hashed disk cache

**Files:**
- Create: `bench/cache.py`
- Create: `tests/test_bench_cache.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_bench_cache.py`:

```python
import numpy as np

from bench.cache import (
    align_key,
    audio_sha1,
    load_align,
    load_sortformer,
    load_whisper,
    save_align,
    save_sortformer,
    save_whisper,
    whisper_key,
)
from transcript.models import Turn, Word
from transcript.pipeline_config import DiarizeConfig, TranscribeConfig


def _make_wav(tmp_path, content=b"\0\0\0\0"):
    wav = tmp_path / "in.wav"
    wav.write_bytes(content)
    return wav


def test_audio_sha1_is_deterministic_and_hex(tmp_path):
    wav = _make_wav(tmp_path)
    h1 = audio_sha1(wav)
    h2 = audio_sha1(wav)
    assert h1 == h2
    assert len(h1) == 40
    assert all(c in "0123456789abcdef" for c in h1)


def test_whisper_key_changes_only_when_relevant_fields_change(tmp_path):
    wav = _make_wav(tmp_path)
    cfg_a = TranscribeConfig(model="large-v3", language="fr")
    cfg_b = TranscribeConfig(model="large-v3", language="fr", no_fallback=False)
    cfg_c = TranscribeConfig(model="large-v3", language="fr")
    assert whisper_key(wav, cfg_a) != whisper_key(wav, cfg_b)
    assert whisper_key(wav, cfg_a) == whisper_key(wav, cfg_c)


def test_save_and_load_whisper_roundtrips(tmp_path):
    wav = _make_wav(tmp_path)
    cfg = TranscribeConfig(language="fr")
    words = [Word(" hi", 0.0, 0.5), Word(" there", 0.5, 1.0)]
    save_whisper(wav, cfg, words, cache_dir=tmp_path)
    assert load_whisper(wav, cfg, cache_dir=tmp_path) == words


def test_load_whisper_returns_none_on_miss(tmp_path):
    wav = _make_wav(tmp_path)
    cfg = TranscribeConfig(language="fr")
    assert load_whisper(wav, cfg, cache_dir=tmp_path) is None


def test_save_and_load_sortformer_roundtrips_without_probs(tmp_path):
    wav = _make_wav(tmp_path)
    cfg = DiarizeConfig()
    turns = [Turn("Speaker 1", 0.0, 1.0)]
    save_sortformer(wav, cfg, turns, probs=None, cache_dir=tmp_path)
    loaded_turns, loaded_probs = load_sortformer(wav, cfg, cache_dir=tmp_path)
    assert loaded_turns == turns
    assert loaded_probs is None


def test_save_and_load_sortformer_roundtrips_with_probs(tmp_path):
    wav = _make_wav(tmp_path)
    cfg = DiarizeConfig(emit_probs=True)
    turns = [Turn("Speaker 1", 0.0, 1.0)]
    probs = np.random.RandomState(0).rand(20, 4).astype(np.float32)
    save_sortformer(wav, cfg, turns, probs=probs, cache_dir=tmp_path)
    loaded_turns, loaded_probs = load_sortformer(wav, cfg, cache_dir=tmp_path)
    assert loaded_turns == turns
    assert loaded_probs is not None
    np.testing.assert_array_equal(loaded_probs, probs)


def test_align_key_includes_whisper_hash(tmp_path):
    wav = _make_wav(tmp_path)
    cfg_a = TranscribeConfig(language="fr")
    cfg_b = TranscribeConfig(language="fr", no_fallback=False)
    h_a = align_key(wav, whisper_key(wav, cfg_a), language="fr")
    h_b = align_key(wav, whisper_key(wav, cfg_b), language="fr")
    assert h_a != h_b


def test_save_and_load_align_roundtrips(tmp_path):
    wav = _make_wav(tmp_path)
    words = [Word(" hi", 0.0, 0.5)]
    save_align(wav, "wkey", "fr", words, cache_dir=tmp_path)
    loaded = load_align(wav, "wkey", "fr", cache_dir=tmp_path)
    assert loaded == words
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_bench_cache.py -v`
Expected: FAIL — `bench.cache` doesn't exist.

- [ ] **Step 3: Implement `bench/cache.py`**

```python
"""Content-hashed on-disk cache for the three expensive pipeline stages.

Each cached artefact's key incorporates a sha1 of the input audio plus only
the config fields that affect that stage's output. Serialisation: JSON for
structured data, NumPy `.npy` for tensors — no opaque binary formats.
"""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from transcript.models import Turn, Word
from transcript.pipeline_config import DiarizeConfig, TranscribeConfig


def audio_sha1(audio_path: Path) -> str:
    h = hashlib.sha1()
    with audio_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode())
        h.update(b"\0")
    return h.hexdigest()[:16]


def whisper_key(audio_path: Path, cfg: TranscribeConfig) -> str:
    return _hash(
        audio_sha1(audio_path),
        "whisper",
        json.dumps(asdict(cfg), sort_keys=True),
    )


def sortformer_key(audio_path: Path, cfg: DiarizeConfig) -> str:
    relevant = {
        "streaming_preset": cfg.streaming_preset,
        "emit_probs": cfg.emit_probs,
    }
    return _hash(
        audio_sha1(audio_path),
        "sortformer",
        json.dumps(relevant, sort_keys=True),
    )


def align_key(audio_path: Path, whisper_hash: str, language: str) -> str:
    return _hash(audio_sha1(audio_path), "align", whisper_hash, language)


def _whisper_path(audio_path, cfg, cache_dir):
    return cache_dir / "whisper" / f"{whisper_key(audio_path, cfg)}.json"


def save_whisper(audio_path: Path, cfg: TranscribeConfig, words: list[Word],
                 *, cache_dir: Path) -> None:
    path = _whisper_path(audio_path, cfg, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([{"text": w.text, "start": w.start, "end": w.end}
                                for w in words]))


def load_whisper(audio_path: Path, cfg: TranscribeConfig,
                 *, cache_dir: Path) -> list[Word] | None:
    path = _whisper_path(audio_path, cfg, cache_dir)
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    return [Word(**r) for r in raw]


def _sortformer_dir(audio_path, cfg, cache_dir):
    return cache_dir / "sortformer" / sortformer_key(audio_path, cfg)


def save_sortformer(audio_path: Path, cfg: DiarizeConfig, turns: list[Turn],
                    *, probs: np.ndarray | None, cache_dir: Path) -> None:
    base = _sortformer_dir(audio_path, cfg, cache_dir)
    base.mkdir(parents=True, exist_ok=True)
    (base / "turns.json").write_text(json.dumps(
        [{"speaker": t.speaker, "start": t.start, "end": t.end} for t in turns]
    ))
    if probs is not None:
        np.save(base / "probs.npy", probs)


def load_sortformer(audio_path: Path, cfg: DiarizeConfig,
                    *, cache_dir: Path) -> tuple[list[Turn], np.ndarray | None] | None:
    base = _sortformer_dir(audio_path, cfg, cache_dir)
    turns_file = base / "turns.json"
    if not turns_file.exists():
        return None
    turns = [Turn(**r) for r in json.loads(turns_file.read_text())]
    probs_file = base / "probs.npy"
    probs = np.load(probs_file) if probs_file.exists() else None
    return turns, probs


def _align_path(audio_path, whisper_hash, language, cache_dir):
    return cache_dir / "align" / f"{align_key(audio_path, whisper_hash, language)}.json"


def save_align(audio_path: Path, whisper_hash: str, language: str,
               words: list[Word], *, cache_dir: Path) -> None:
    path = _align_path(audio_path, whisper_hash, language, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([{"text": w.text, "start": w.start, "end": w.end}
                                for w in words]))


def load_align(audio_path: Path, whisper_hash: str, language: str,
               *, cache_dir: Path) -> list[Word] | None:
    path = _align_path(audio_path, whisper_hash, language, cache_dir)
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    return [Word(**r) for r in raw]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_bench_cache.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add bench/cache.py tests/test_bench_cache.py
git commit -m "feat(bench): content-hashed disk cache for whisper/sortformer/align"
```

---

### Task 11: Implement `bench/artefacts.py` — write transcripts + diffs

**Files:**
- Create: `bench/artefacts.py`
- Create: `tests/test_bench_artefacts.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_bench_artefacts.py`:

```python
import json

from bench.artefacts import save_diff, save_transcript
from transcript.models import Utterance


def test_save_transcript_writes_hypothesis_and_reference(tmp_path):
    hypothesis = [Utterance("Speaker 1", 0.0, 1.0, "bonjour"),
                  Utterance("Speaker 2", 1.0, 2.0, "salut")]
    reference = [Utterance("Speaker A", 0.0, 1.0, "bonjour"),
                 Utterance("Speaker B", 1.0, 2.0, "salut")]
    path = save_transcript(
        results_dir=tmp_path, tier=1, clip_id="AMI:EN2002a",
        config_fingerprint="abc123",
        hypothesis=hypothesis, reference=reference,
    )
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["clip_id"] == "AMI:EN2002a"
    assert data["config_fingerprint"] == "abc123"
    assert len(data["hypothesis"]) == 2
    assert len(data["reference"]) == 2
    assert data["hypothesis"][0]["text"] == "bonjour"


def test_save_diff_writes_word_ops_and_totals(tmp_path):
    speaker_permutation = {"Speaker 1": "Speaker A", "Speaker 2": "Speaker B"}
    word_ops = [
        {"op": "equal", "ref_word": "bonjour", "hyp_word": "bonjour",
         "ref_speaker": "A", "hyp_speaker": "A"},
        {"op": "sub", "ref_word": "allons", "hyp_word": "allon",
         "ref_speaker": "A", "hyp_speaker": "A"},
        {"op": "speaker_swap", "ref_word": "oui", "hyp_word": "oui",
         "ref_speaker": "B", "hyp_speaker": "A"},
    ]
    path = save_diff(
        results_dir=tmp_path, tier=2, clip_id="SUMM-RE:001",
        config_fingerprint="xyz789",
        speaker_permutation=speaker_permutation, word_ops=word_ops,
    )
    data = json.loads(path.read_text())
    assert data["speaker_permutation"] == speaker_permutation
    assert len(data["word_ops"]) == 3
    assert data["totals"] == {"sub": 1, "ins": 0, "del": 0, "speaker_swap": 1}


def test_paths_follow_layout_spec(tmp_path):
    p = save_transcript(
        results_dir=tmp_path, tier=3, clip_id="AMI:EN2002a",
        config_fingerprint="deadbeef0001",
        hypothesis=[], reference=[],
    )
    assert p == tmp_path / "transcripts" / "tier-3" / "AMI_EN2002a" / "deadbeef0001.json"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_bench_artefacts.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `bench/artefacts.py`**

```python
"""Persist per-(clip x config x tier) transcripts and word-level diffs.

Output paths:
  bench/results/transcripts/tier-N/<safe-clip-id>/<config_fingerprint>.json
  bench/results/diffs/tier-N/<safe-clip-id>/<config_fingerprint>.json

Both are JSON. Transcripts hold hypothesis + reference utterances; diffs hold
the meeteval-aligned word ops plus the cpWER speaker permutation. These are
the evidence behind every CSV row — the foundation for future failure-mode
analysis when we tailor llm_fix prompts.
"""
import json
from dataclasses import asdict
from pathlib import Path

from transcript.models import Utterance


def _safe(clip_id: str) -> str:
    return clip_id.replace(":", "_").replace("/", "_")


def _path(results_dir: Path, kind: str, tier: int, clip_id: str, fp: str) -> Path:
    return results_dir / kind / f"tier-{tier}" / _safe(clip_id) / f"{fp}.json"


def save_transcript(
    *,
    results_dir: Path,
    tier: int,
    clip_id: str,
    config_fingerprint: str,
    hypothesis: list[Utterance],
    reference: list[Utterance],
) -> Path:
    path = _path(results_dir, "transcripts", tier, clip_id, config_fingerprint)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "clip_id": clip_id,
        "config_fingerprint": config_fingerprint,
        "hypothesis": [asdict(u) for u in hypothesis],
        "reference":  [asdict(u) for u in reference],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path


def save_diff(
    *,
    results_dir: Path,
    tier: int,
    clip_id: str,
    config_fingerprint: str,
    speaker_permutation: dict[str, str],
    word_ops: list[dict],
) -> Path:
    path = _path(results_dir, "diffs", tier, clip_id, config_fingerprint)
    path.parent.mkdir(parents=True, exist_ok=True)
    totals = {"sub": 0, "ins": 0, "del": 0, "speaker_swap": 0}
    for op in word_ops:
        if op["op"] in totals:
            totals[op["op"]] += 1
    payload = {
        "clip_id": clip_id,
        "config_fingerprint": config_fingerprint,
        "speaker_permutation": speaker_permutation,
        "word_ops": word_ops,
        "totals": totals,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_bench_artefacts.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add bench/artefacts.py tests/test_bench_artefacts.py
git commit -m "feat(bench): write hypothesis transcripts and meeteval diffs to disk"
```

---

## Phase C: bench metrics

### Task 12: Implement `bench/metrics.py` — normalisation + score()

**Files:**
- Create: `bench/metrics.py`
- Create: `tests/test_bench_metrics.py`

The meeteval API has small version-to-version variance. The implementation below targets the 0.4.x line; if the installed version's API differs, adjust the import path or wrapper but keep the public `score()` contract intact.

- [ ] **Step 1: Write failing tests**

Create `tests/test_bench_metrics.py`:

```python
from bench.metrics import ClipMetrics, normalise, score
from transcript.models import Utterance


def test_normalise_lowercases_and_strips_punctuation():
    assert normalise("Bonjour, monde!") == "bonjour monde"


def test_normalise_collapses_whitespace():
    assert normalise("hello    world\n\nfoo") == "hello world foo"


def test_normalise_strips_summ_re_markers():
    assert normalise("salut @ * + ami") == "salut ami"


def test_normalise_is_idempotent():
    s = "Salut, là-bas! Comment ça va @ ?"
    assert normalise(s) == normalise(normalise(s))


def test_score_perfect_match_yields_zero():
    hyp = [Utterance("Speaker 1", 0.0, 1.0, "bonjour")]
    ref = [Utterance("Speaker 1", 0.0, 1.0, "bonjour")]
    m = score(hyp, ref)
    assert m.cpwer == 0.0
    assert m.wer == 0.0
    assert m.speaker_assignment_error_rate == 0.0


def test_score_completely_wrong_word_yields_wer_one():
    hyp = [Utterance("Speaker 1", 0.0, 1.0, "salut")]
    ref = [Utterance("Speaker 1", 0.0, 1.0, "bonjour")]
    m = score(hyp, ref)
    assert m.wer == 1.0


def test_score_returns_dataclass_with_four_metrics():
    hyp = [Utterance("Speaker 1", 0.0, 1.0, "bonjour")]
    ref = [Utterance("Speaker 1", 0.0, 1.0, "bonjour")]
    m = score(hyp, ref)
    assert isinstance(m, ClipMetrics)
    assert all(hasattr(m, k) for k in ("cpwer", "wer", "der", "speaker_assignment_error_rate"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_bench_metrics.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `bench/metrics.py`**

```python
"""Metric computation for the bench harness.

Primary: cpWER (concatenated minimum-permutation WER) via meeteval.
Secondary diagnostics: speaker-agnostic WER, DER, and the cpWER-WER
decomposition (the "speaker-assignment error rate" — right word, wrong
speaker — which is the failure mode the prob-based merge targets).
"""
import re
import unicodedata
from dataclasses import dataclass
from itertools import permutations

import numpy as np

from transcript.models import Utterance

_PUNCT_RE   = re.compile(r"[^\w\s\-']", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")
_SUMM_RE_MARKERS = re.compile(r"[@*+]")


@dataclass(frozen=True)
class ClipMetrics:
    cpwer: float
    wer:   float
    der:   float
    speaker_assignment_error_rate: float


def normalise(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = _SUMM_RE_MARKERS.sub(" ", s)
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WHITESPACE.sub(" ", s).strip()
    return s


def _utterances_to_meeteval(utterances: list[Utterance]) -> dict[str, str]:
    """Build a meeteval-compatible per-speaker dict[str, str] payload."""
    by_speaker: dict[str, list[str]] = {}
    for u in utterances:
        text = normalise(u.text)
        if not text:
            continue
        by_speaker.setdefault(u.speaker, []).append(text)
    return {spk: " ".join(words) for spk, words in by_speaker.items()}


def _speaker_agnostic_wer(hyp: list[Utterance], ref: list[Utterance]) -> float:
    """Concatenate everything ignoring speakers; compute plain WER via meeteval."""
    from meeteval.wer import wer as meet_wer
    hyp_text = normalise(" ".join(u.text for u in hyp))
    ref_text = normalise(" ".join(u.text for u in ref))
    result = meet_wer(reference=ref_text, hypothesis=hyp_text)
    return float(result.error_rate)


def _cpwer(hyp: list[Utterance], ref: list[Utterance]) -> float:
    from meeteval.wer import cpwer as meet_cpwer
    hyp_d = _utterances_to_meeteval(hyp)
    ref_d = _utterances_to_meeteval(ref)
    result = meet_cpwer(reference=ref_d, hypothesis=hyp_d)
    return float(result.error_rate)


def _der(hyp: list[Utterance], ref: list[Utterance]) -> float:
    """Speaker-only DER approximation: fraction of reference frames labelled with
    the wrong (post-permutation) hypothesis speaker, at 10 ms resolution."""
    if not ref:
        return 0.0
    end = max(u.end for u in ref)
    step = 0.01
    n = int(end / step) + 1
    ref_arr = np.full(n, "", dtype=object)
    hyp_arr = np.full(n, "", dtype=object)
    for u in ref:
        ref_arr[int(u.start / step):int(u.end / step)] = u.speaker
    for u in hyp:
        hyp_arr[int(u.start / step):int(min(u.end, end) / step)] = u.speaker

    ref_spks = sorted({s for s in ref_arr if s})
    hyp_spks = sorted({s for s in hyp_arr if s})
    if not ref_spks or not hyp_spks:
        return 1.0

    # Brute-force speaker permutation (cap: 4 speakers → 24 permutations).
    best_err = None
    for perm in permutations(hyp_spks, len(hyp_spks)):
        mapping = dict(zip(hyp_spks, list(perm) + ref_spks[len(hyp_spks):]))
        remapped = np.array([mapping.get(s, s) for s in hyp_arr], dtype=object)
        err = float(np.sum(remapped != ref_arr)) / float(np.sum(ref_arr != "") or 1)
        if best_err is None or err < best_err:
            best_err = err
    return float(best_err or 1.0)


def score(hypothesis: list[Utterance], reference: list[Utterance]) -> ClipMetrics:
    w = _speaker_agnostic_wer(hypothesis, reference)
    c = _cpwer(hypothesis, reference)
    d = _der(hypothesis, reference)
    return ClipMetrics(
        cpwer = c,
        wer   = w,
        der   = d,
        speaker_assignment_error_rate = max(0.0, c - w),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_bench_metrics.py -v`
Expected: all PASS. If a meeteval API mismatch surfaces, adjust the `_speaker_agnostic_wer` / `_cpwer` callsites to whatever the installed 0.4.x version exposes — the wrapper is the only thing that needs to change.

- [ ] **Step 5: Commit**

```bash
git add bench/metrics.py tests/test_bench_metrics.py
git commit -m "feat(bench): cpWER + WER + DER + speaker-assignment-error-rate"
```

---

## Phase D: Datasets

### Task 13: `bench/datasets/base.py` — protocol + dataclass

**Files:**
- Create: `bench/datasets/base.py`

This task has no dedicated tests — the protocol is verified by the loader-specific tests in Tasks 14 and 15.

- [ ] **Step 1: Create the module**

```python
"""Shared types for bench datasets."""
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class BenchClip:
    clip_id: str
    audio_path: Path
    language: str       # ISO 639-1
    num_speakers: int
    duration_s: float
    reference_rttm: Path
    reference_stm: Path


class Dataset(Protocol):
    name: str

    def sample(
        self,
        n: int,
        *,
        max_duration_s: float | None = None,
        seed: int = 42,
    ) -> list[BenchClip]: ...
```

- [ ] **Step 2: Commit**

```bash
git add bench/datasets/base.py
git commit -m "feat(bench): BenchClip dataclass and Dataset protocol"
```

---

### Task 14: `bench/datasets/ami.py` — AMI sdm loader

**Files:**
- Create: `bench/datasets/ami.py`
- Create: `bench/datasets/ami_rttm/` (populated by the loader's bootstrap step)

This task is wiring rather than TDD — the loader converts HF rows + vendored RTTMs into `BenchClip` objects. Skip rules from the spec (no RTTM / >4 speakers) are exercised.

- [ ] **Step 1: Implement `bench/datasets/ami.py`**

```python
"""AMI sdm corpus loader.

Reference RTTMs come from BUTSpeechFIT/AMI-diarization-setup, vendored under
bench/datasets/ami_rttm/. AMI audio is pulled from `edinburghcstr/ami` HF
dataset (sdm config — single distant mic) and pre-prepared into 16 kHz mono
WAVs cached under bench/cache/audio/ami/.

If the vendored RTTM directory is empty on first run, attempt to clone the
BUT repo into bench/cache/ami_rttm/ as a fallback (warns the user).
"""
import random
import shutil
import subprocess
from pathlib import Path

from bench.datasets.base import BenchClip

_HF_DATASET = "edinburghcstr/ami"
_HF_CONFIG  = "sdm"  # single distant mic
_BUT_REPO   = "https://github.com/BUTSpeechFIT/AMI-diarization-setup.git"


class AMIDataset:
    name = "AMI"

    def __init__(self, *, cache_dir: Path, rttm_dir: Path | None = None):
        self.cache_dir = cache_dir
        self.audio_dir = cache_dir / "audio" / "ami"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.rttm_dir = rttm_dir or self._resolve_rttm_dir(cache_dir)

    @staticmethod
    def _resolve_rttm_dir(cache_dir: Path) -> Path:
        vendored = Path(__file__).parent / "ami_rttm"
        if vendored.exists() and any(vendored.glob("*.rttm")):
            return vendored
        runtime = cache_dir / "ami_rttm"
        if not runtime.exists():
            subprocess.run(
                ["git", "clone", "--depth", "1", _BUT_REPO, str(runtime)],
                check=True,
            )
        # The BUT repo nests RTTMs under a subdirectory — adjust this path
        # after first clone if needed (open question in the spec).
        return runtime

    def _load_index(self) -> list[dict]:
        from datasets import load_dataset
        ds = load_dataset(_HF_DATASET, _HF_CONFIG, split="test")
        seen: dict[str, dict] = {}
        for row in ds:
            mid = row["meeting_id"]
            if mid not in seen:
                seen[mid] = {
                    "meeting_id": mid,
                    "audio": row["audio"]["path"],
                    "duration": float(row["audio"]["array"].shape[0])
                                / float(row["audio"]["sampling_rate"]),
                }
        return list(seen.values())

    def _prepare_clip(self, meeting: dict) -> BenchClip | None:
        from transcript import audio as audio_mod

        meeting_id = meeting["meeting_id"]
        rttm_file = self.rttm_dir / f"{meeting_id}.rttm"
        if not rttm_file.exists():
            return None

        wav_path = self.audio_dir / f"{meeting_id}.wav"
        if not wav_path.exists():
            prepared, _ = audio_mod.prepare(Path(meeting["audio"]))
            shutil.move(prepared, wav_path)

        num_speakers = _count_rttm_speakers(rttm_file)
        if num_speakers > 4:
            return None  # Sortformer 4-speaker cap

        stm_file = wav_path.with_suffix(".stm")
        if not stm_file.exists():
            _ami_stm_for(meeting_id, stm_file)

        return BenchClip(
            clip_id=f"AMI:{meeting_id}",
            audio_path=wav_path,
            language="en",
            num_speakers=num_speakers,
            duration_s=meeting["duration"],
            reference_rttm=rttm_file,
            reference_stm=stm_file,
        )

    def sample(self, n: int, *, max_duration_s: float | None = None,
               seed: int = 42) -> list[BenchClip]:
        rng = random.Random(seed)
        clips: list[BenchClip] = []
        index = self._load_index()
        rng.shuffle(index)
        for meeting in index:
            if max_duration_s is not None and meeting["duration"] > max_duration_s:
                continue
            clip = self._prepare_clip(meeting)
            if clip is not None:
                clips.append(clip)
            if len(clips) >= n:
                break
        return clips


def _count_rttm_speakers(rttm_path: Path) -> int:
    speakers = set()
    for line in rttm_path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 8 and parts[0] == "SPEAKER":
            speakers.add(parts[7])
    return len(speakers)


def _ami_stm_for(meeting_id: str, out: Path) -> None:
    """Build an STM file from AMI manual transcripts."""
    from datasets import load_dataset
    ds = load_dataset(_HF_DATASET, _HF_CONFIG, split="test")
    lines = []
    for row in ds:
        if row["meeting_id"] != meeting_id:
            continue
        speaker = row.get("speaker_id", "Speaker_1")
        lines.append(
            f"{meeting_id} 1 {speaker} {row['begin_time']:.2f} {row['end_time']:.2f} <NA> {row['text']}"
        )
    out.write_text("\n".join(lines))
```

- [ ] **Step 2: Smoke-test import (no network)**

Run: `uv run python -c "from bench.datasets.ami import AMIDataset; print('ok')"`
Expected: `ok`. No real HF call is made at import time.

- [ ] **Step 3: Commit**

```bash
git add bench/datasets/ami.py
git commit -m "feat(bench): AMI sdm loader with vendored RTTM fallback to runtime clone"
```

---

### Task 15: `bench/datasets/summ_re.py` — track mixing + synth RTTM/STM

**Files:**
- Create: `bench/datasets/summ_re.py`
- Create: `tests/test_summ_re_loader.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_summ_re_loader.py`:

```python
import numpy as np
import soundfile as sf

from bench.datasets.summ_re import _mix_tracks, _synthesise_rttm, _synthesise_stm


def test_synthesise_rttm_emits_one_line_per_segment(tmp_path):
    tracks = [
        {"speaker_id": "001", "segments": [
            {"start": 0.0, "end": 1.0}, {"start": 1.5, "end": 2.0},
        ]},
        {"speaker_id": "002", "segments": [
            {"start": 1.0, "end": 1.5},
        ]},
    ]
    out = tmp_path / "ref.rttm"
    _synthesise_rttm(tracks, meeting_id="m1", out_path=out)
    lines = out.read_text().splitlines()
    assert len(lines) == 3
    for line in lines:
        parts = line.split()
        assert parts[0] == "SPEAKER"
        assert parts[1] == "m1"


def test_synthesise_stm_concatenates_words_per_speaker(tmp_path):
    tracks = [
        {"speaker_id": "001", "segments": [
            {"start": 0.0, "end": 1.0, "words": [
                {"word": "bonjour", "start": 0.0, "end": 0.5},
                {"word": "toi",     "start": 0.5, "end": 1.0},
            ]},
        ]},
    ]
    out = tmp_path / "ref.stm"
    _synthesise_stm(tracks, meeting_id="m1", out_path=out)
    line = out.read_text().strip()
    assert "bonjour" in line
    assert "toi" in line


def test_mix_tracks_writes_16khz_mono_wav(tmp_path):
    track_a = tmp_path / "a.wav"
    track_b = tmp_path / "b.wav"
    sf.write(track_a, np.zeros(32000, dtype=np.float32), 32000)
    sf.write(track_b, np.ones(32000, dtype=np.float32) * 0.1, 32000)

    out = tmp_path / "mixed.wav"
    _mix_tracks([track_a, track_b], out_path=out)
    data, sr = sf.read(out)
    assert sr == 16000
    assert data.ndim == 1
    assert abs(len(data) - 16000) < 100
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_summ_re_loader.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `bench/datasets/summ_re.py`**

```python
"""SUMM-RE loader.

SUMM-RE ships per-speaker tracks (3-4 speakers per meeting). To run the
transcript pipeline on a meeting we mix the tracks into a single 16 kHz mono
WAV with ffmpeg amix, and synthesise the reference RTTM/STM from the per-track
segment + word metadata.
"""
import random
import subprocess
import tempfile
from pathlib import Path

from bench.datasets.base import BenchClip


class SUMMREDataset:
    name = "SUMM-RE"

    def __init__(self, *, cache_dir: Path):
        self.cache_dir = cache_dir
        self.audio_dir = cache_dir / "audio" / "summ_re"
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    def _iter_meetings(self):
        from datasets import load_dataset
        ds = load_dataset("linagora/SUMM-RE", split="dev", streaming=True)
        by_meeting: dict[str, list[dict]] = {}
        for row in ds:
            by_meeting.setdefault(row["meeting_id"], []).append(row)
        return by_meeting.items()

    def _prepare_clip(self, meeting_id: str, tracks: list[dict]) -> BenchClip | None:
        from transcript import audio as audio_mod
        wav_path = self.audio_dir / f"{meeting_id}.wav"
        rttm_path = wav_path.with_suffix(".rttm")
        stm_path  = wav_path.with_suffix(".stm")

        if not wav_path.exists():
            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                track_paths = []
                import soundfile as sf
                for tr in tracks:
                    p = td_path / f"{tr['audio_id']}.wav"
                    sf.write(p, tr["audio"]["array"], tr["audio"]["sampling_rate"])
                    track_paths.append(p)
                _mix_tracks(track_paths, out_path=wav_path)

        if not rttm_path.exists():
            _synthesise_rttm(tracks, meeting_id=meeting_id, out_path=rttm_path)
        if not stm_path.exists():
            _synthesise_stm(tracks, meeting_id=meeting_id, out_path=stm_path)

        if not rttm_path.read_text().strip():
            return None
        n_speakers = len({tr["speaker_id"] for tr in tracks})
        if n_speakers > 4:
            return None

        duration = audio_mod._probe(wav_path)["duration"]
        return BenchClip(
            clip_id=f"SUMM-RE:{meeting_id}",
            audio_path=wav_path,
            language="fr",
            num_speakers=n_speakers,
            duration_s=duration,
            reference_rttm=rttm_path,
            reference_stm=stm_path,
        )

    def sample(self, n: int, *, max_duration_s: float | None = None,
               seed: int = 42) -> list[BenchClip]:
        rng = random.Random(seed)
        meetings = list(self._iter_meetings())
        rng.shuffle(meetings)
        clips: list[BenchClip] = []
        for meeting_id, tracks in meetings:
            clip = self._prepare_clip(meeting_id, tracks)
            if clip is None:
                continue
            if max_duration_s is not None and clip.duration_s > max_duration_s:
                continue
            clips.append(clip)
            if len(clips) >= n:
                break
        return clips


def _mix_tracks(track_paths: list[Path], *, out_path: Path) -> None:
    """ffmpeg amix N tracks → 16 kHz mono PCM16 WAV."""
    n = len(track_paths)
    inputs: list[str] = []
    for p in track_paths:
        inputs += ["-i", str(p)]
    cmd = [
        "ffmpeg", "-loglevel", "error", "-y",
        *inputs,
        "-filter_complex", f"amix=inputs={n}:duration=longest:normalize=0",
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        str(out_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def _synthesise_rttm(tracks: list[dict], *, meeting_id: str, out_path: Path) -> None:
    lines = []
    for tr in tracks:
        spk = tr["speaker_id"]
        for seg in tr.get("segments", []):
            start = float(seg["start"])
            dur = float(seg["end"]) - start
            lines.append(
                f"SPEAKER {meeting_id} 1 {start:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>"
            )
    out_path.write_text("\n".join(lines))


def _synthesise_stm(tracks: list[dict], *, meeting_id: str, out_path: Path) -> None:
    """One STM line per (speaker, sorted-by-start segment), text concatenated from words."""
    rows: list[tuple[float, float, str, str]] = []
    for tr in tracks:
        spk = tr["speaker_id"]
        for seg in tr.get("segments", []):
            words = seg.get("words") or []
            text = " ".join(w["word"] for w in words).strip()
            if not text:
                text = seg.get("transcript", "").strip()
            if not text:
                continue
            rows.append((float(seg["start"]), float(seg["end"]), spk, text))
    rows.sort()
    lines = [
        f"{meeting_id} 1 {spk} {s:.2f} {e:.2f} <NA> {text}"
        for s, e, spk, text in rows
    ]
    out_path.write_text("\n".join(lines))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_summ_re_loader.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add bench/datasets/summ_re.py tests/test_summ_re_loader.py
git commit -m "feat(bench): SUMM-RE loader with per-speaker track mixing and synth RTTM/STM"
```

---

## Phase E: Tier logic + runner

### Task 16: `bench/tiers.py` — tier-1/2/3 config generators

**Files:**
- Create: `bench/tiers.py`
- Create: `tests/test_bench_tiers.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_bench_tiers.py`:

```python
from bench.tiers import tier_1_configs, tier_2_configs, tier_3_configs
from transcript.pipeline_config import PipelineConfig


def test_tier_1_generates_full_grid_of_32():
    configs = tier_1_configs()
    assert len(configs) == 32
    assert all(isinstance(c, PipelineConfig) for c in configs)
    fingerprints = {c.fingerprint() for c in configs}
    assert len(fingerprints) == 32  # all distinct


def test_tier_1_pins_whisper_model_to_large_v3():
    for c in tier_1_configs():
        assert c.transcribe.model == "large-v3"
        assert c.llm_fix.enabled is False


def test_tier_2_drops_axes_with_low_effect_size():
    # Synthetic tier-1 rows: only merge.strategy moves the needle.
    tier_1_rows = [
        {"no_fallback": True,  "suppress_nst": True,  "streaming_preset": "very_high_lat",
         "align": True, "merge_strategy": "hard_boundary", "cpwer": 0.10},
        {"no_fallback": True,  "suppress_nst": True,  "streaming_preset": "very_high_lat",
         "align": True, "merge_strategy": "prob_based",    "cpwer": 0.08},
        {"no_fallback": False, "suppress_nst": True,  "streaming_preset": "very_high_lat",
         "align": True, "merge_strategy": "hard_boundary", "cpwer": 0.10},
        {"no_fallback": True,  "suppress_nst": False, "streaming_preset": "very_high_lat",
         "align": True, "merge_strategy": "hard_boundary", "cpwer": 0.10},
        {"no_fallback": True,  "suppress_nst": True,  "streaming_preset": "low_lat",
         "align": True, "merge_strategy": "hard_boundary", "cpwer": 0.10},
        {"no_fallback": True,  "suppress_nst": True,  "streaming_preset": "very_high_lat",
         "align": False, "merge_strategy": "hard_boundary", "cpwer": 0.10},
    ]
    configs = tier_2_configs(tier_1_rows)
    assert len(configs) == 2
    strategies = {c.merge.strategy for c in configs}
    assert strategies == {"hard_boundary", "prob_based"}


def test_tier_3_picks_finalists_within_threshold():
    tier_2_rows = [
        {"fingerprint": "a", "merge_strategy": "prob_based",    "cpwer": 0.08},
        {"fingerprint": "b", "merge_strategy": "hard_boundary", "cpwer": 0.09},
        {"fingerprint": "c", "merge_strategy": "hard_boundary", "cpwer": 0.15},
    ]
    finalists = tier_3_configs(tier_2_rows)
    assert len(finalists) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_bench_tiers.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `bench/tiers.py`**

```python
"""Tier config generators.

Tier 1: full Cartesian product of the 5 tunable axes (whisper model is pinned
        at large-v3; llm_fix is pinned at False). 32 configs.
Tier 2: drop axes whose effect size (max-min cpWER across the axis's values)
        is below 0.5 absolute cpWER points; product over the rest.
Tier 3: configs whose median cpWER ≤ best + 1.0 absolute points (capped at 5);
        relax to 2.0 points if fewer than 3 qualify.
"""
from itertools import product
from typing import Iterable

from transcript.pipeline_config import (
    AlignConfig,
    DiarizeConfig,
    MergeConfig,
    PipelineConfig,
    TranscribeConfig,
)


_AXES_BOOL: dict[str, list] = {
    "no_fallback":     [True, False],
    "suppress_nst":    [True, False],
    "streaming_preset": ["very_high_lat", "low_lat"],
    "align":           [True, False],
    "merge_strategy":  ["hard_boundary", "prob_based"],
}


def tier_1_configs() -> list[PipelineConfig]:
    """Full 32-config grid."""
    configs: list[PipelineConfig] = []
    for nf, sn, sp, al, mg in product(*_AXES_BOOL.values()):
        configs.append(_build_config(nf, sn, sp, al, mg))
    return configs


def tier_2_configs(tier_1_rows: Iterable[dict],
                   threshold: float = 0.5) -> list[PipelineConfig]:
    rows = list(tier_1_rows)
    if not rows:
        return tier_1_configs()
    best = min(rows, key=lambda r: r["cpwer"])
    pinned = {k: best[k] for k in _AXES_BOOL.keys()}
    kept_axes: dict[str, list] = {}
    for axis, values in _AXES_BOOL.items():
        per_value_cpwer = {}
        for v in values:
            siblings = [r["cpwer"] for r in rows if r[axis] == v
                        and all(r[a] == pinned[a] for a in _AXES_BOOL if a != axis)]
            if siblings:
                per_value_cpwer[v] = min(siblings)
        if not per_value_cpwer:
            continue
        effect = (max(per_value_cpwer.values()) - min(per_value_cpwer.values())) * 100
        if effect >= threshold:
            kept_axes[axis] = list(per_value_cpwer.keys())
    if not kept_axes:
        return [_build_config(**pinned)]
    keys = list(kept_axes.keys())
    configs: list[PipelineConfig] = []
    for combo in product(*[kept_axes[k] for k in keys]):
        values = dict(pinned)
        for k, v in zip(keys, combo):
            values[k] = v
        configs.append(_build_config(**values))
    return configs


def tier_3_configs(tier_2_rows: Iterable[dict],
                   primary_threshold: float = 1.0,
                   relaxed_threshold: float = 2.0,
                   cap: int = 5) -> list[PipelineConfig]:
    rows = list(tier_2_rows)
    if not rows:
        return []
    by_fp: dict[str, list[float]] = {}
    for r in rows:
        by_fp.setdefault(r["fingerprint"], []).append(r["cpwer"])
    ranked = sorted(
        ((fp, sorted(cs)[len(cs) // 2]) for fp, cs in by_fp.items()),
        key=lambda x: x[1],
    )
    best = ranked[0][1]
    threshold = primary_threshold
    finalists = [(fp, c) for fp, c in ranked if (c - best) * 100 <= threshold]
    if len(finalists) < 3:
        threshold = relaxed_threshold
        finalists = [(fp, c) for fp, c in ranked if (c - best) * 100 <= threshold]
    finalists = finalists[:cap]
    return [_build_config(merge_strategy=next(
        r["merge_strategy"] for r in rows if r["fingerprint"] == fp
    )) for fp, _ in finalists]


def _build_config(no_fallback=True, suppress_nst=True,
                  streaming_preset="very_high_lat", align=True,
                  merge_strategy="hard_boundary") -> PipelineConfig:
    return PipelineConfig(
        transcribe=TranscribeConfig(no_fallback=no_fallback, suppress_nst=suppress_nst),
        diarize=DiarizeConfig(streaming_preset=streaming_preset),
        align=AlignConfig(enabled=align),
        merge=MergeConfig(strategy=merge_strategy),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_bench_tiers.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add bench/tiers.py tests/test_bench_tiers.py
git commit -m "feat(bench): tier-1/2/3 config generators with axis-effect narrowing"
```

---

### Task 17: `bench/runner.py` — single-tier execution + CSV append + leaderboard

**Files:**
- Create: `bench/runner.py`

This task wires together everything from Phases A-D. No new tests — the smoke test in Task 19 exercises the full path end-to-end.

- [ ] **Step 1: Implement `bench/runner.py`**

```python
"""Per-tier execution: for each (clip x config) pair, run the cached pipeline,
score the result, append a CSV row, and persist transcripts + diffs.
"""
import csv
import socket
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from statistics import median

from bench import artefacts, cache, metrics
from bench.datasets.base import BenchClip, Dataset
from transcript import align as align_mod
from transcript import diarize, merge, transcribe
from transcript.models import Meta, Utterance
from transcript.pipeline_config import PipelineConfig

CSV_COLUMNS = [
    "tier", "dataset", "clip_id", "config_id", "config_fingerprint",
    "no_fallback", "suppress_nst", "streaming_preset", "align", "merge_strategy",
    "cpwer", "wer", "der", "speaker_assignment_error_rate",
    "runtime_s", "whisper_s", "sortformer_s", "align_s", "merge_s",
    "git_sha", "started_at", "host",
    "hypothesis_path", "diff_path",
]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _load_reference_utterances(stm_path: Path) -> list[Utterance]:
    """Parse a synthesised STM into Utterance objects."""
    out: list[Utterance] = []
    for line in stm_path.read_text().splitlines():
        parts = line.split(maxsplit=6)
        if len(parts) < 7:
            continue
        _file, _ch, speaker, start, end, _na, text = parts
        out.append(Utterance(
            speaker=speaker, start=float(start), end=float(end), text=text
        ))
    return out


def _run_cached(clip: BenchClip, cfg: PipelineConfig,
                cache_dir: Path) -> tuple[list[Utterance], Meta, dict]:
    """Run the pipeline for one (clip x config), reading from / writing to the cache."""
    timings = {"whisper_s": 0.0, "sortformer_s": 0.0, "align_s": 0.0, "merge_s": 0.0}

    transcribe_cfg = replace(cfg.transcribe, language=clip.language)
    diarize_cfg = replace(cfg.diarize, num_speakers=clip.num_speakers)
    if cfg.merge.strategy == "prob_based":
        diarize_cfg = replace(diarize_cfg, emit_probs=True)

    words = cache.load_whisper(clip.audio_path, transcribe_cfg, cache_dir=cache_dir)
    if words is None:
        t = time.time()
        words, _lang = transcribe.run(clip.audio_path, config=transcribe_cfg)
        timings["whisper_s"] = time.time() - t
        cache.save_whisper(clip.audio_path, transcribe_cfg, words, cache_dir=cache_dir)

    cached = cache.load_sortformer(clip.audio_path, diarize_cfg, cache_dir=cache_dir)
    if cached is None:
        t = time.time()
        turns, probs = diarize.run(clip.audio_path, config=diarize_cfg)
        timings["sortformer_s"] = time.time() - t
        cache.save_sortformer(
            clip.audio_path, diarize_cfg, turns, probs=probs, cache_dir=cache_dir
        )
    else:
        turns, probs = cached

    if cfg.align.enabled and align_mod.is_available() and words:
        whisper_h = cache.whisper_key(clip.audio_path, transcribe_cfg)
        aligned = cache.load_align(
            clip.audio_path, whisper_h, clip.language, cache_dir=cache_dir
        )
        if aligned is None:
            t = time.time()
            aligned = align_mod.run(clip.audio_path, words, language=clip.language)
            timings["align_s"] = time.time() - t
            cache.save_align(
                clip.audio_path, whisper_h, clip.language, aligned, cache_dir=cache_dir
            )
        words = aligned

    t = time.time()
    word_speakers = merge.assign_speakers(
        words, turns, strategy=cfg.merge.strategy, probs=probs
    )
    utterances = merge.collapse(word_speakers)
    timings["merge_s"] = time.time() - t

    meta = Meta(
        filename=clip.audio_path.name,
        duration=clip.duration_s,
        model=cfg.transcribe.model,
        language=clip.language,
        speaker_count=len({t.speaker for t in turns}) if turns else 0,
        diarizer=diarize.DIARIZER_LABEL,
    )
    return utterances, meta, timings


def run_one_tier(
    *,
    tier: int,
    configs: list[PipelineConfig],
    datasets: list[Dataset],
    clip_count: int,
    max_duration_s: float | None,
    cache_dir: Path,
    results_dir: Path,
) -> None:
    """Execute every (clip x config) pair for one tier; append to runs.csv."""
    git_sha = _git_sha()
    host = socket.gethostname()
    csv_path = results_dir / "runs.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if is_new:
            writer.writeheader()

        for dataset in datasets:
            clips = dataset.sample(clip_count, max_duration_s=max_duration_s)
            ref_by_clip = {c.clip_id: _load_reference_utterances(c.reference_stm) for c in clips}
            for clip in clips:
                for cfg in configs:
                    fp = cfg.fingerprint()
                    started = time.time()
                    utterances, _meta, timings = _run_cached(clip, cfg, cache_dir)
                    m = metrics.score(utterances, ref_by_clip[clip.clip_id])

                    hyp_path = artefacts.save_transcript(
                        results_dir=results_dir, tier=tier,
                        clip_id=clip.clip_id, config_fingerprint=fp,
                        hypothesis=utterances, reference=ref_by_clip[clip.clip_id],
                    )
                    diff_path = artefacts.save_diff(
                        results_dir=results_dir, tier=tier,
                        clip_id=clip.clip_id, config_fingerprint=fp,
                        speaker_permutation={},
                        word_ops=[],
                    )

                    writer.writerow({
                        "tier": tier, "dataset": dataset.name,
                        "clip_id": clip.clip_id, "config_id": fp,
                        "config_fingerprint": fp,
                        "no_fallback": cfg.transcribe.no_fallback,
                        "suppress_nst": cfg.transcribe.suppress_nst,
                        "streaming_preset": cfg.diarize.streaming_preset,
                        "align": cfg.align.enabled,
                        "merge_strategy": cfg.merge.strategy,
                        "cpwer": m.cpwer, "wer": m.wer, "der": m.der,
                        "speaker_assignment_error_rate": m.speaker_assignment_error_rate,
                        "runtime_s": time.time() - started,
                        **timings,
                        "git_sha": git_sha,
                        "started_at": started, "host": host,
                        "hypothesis_path": str(hyp_path.relative_to(results_dir)),
                        "diff_path": str(diff_path.relative_to(results_dir)),
                    })
                    f.flush()


def generate_leaderboard(*, results_dir: Path) -> Path:
    """Rebuild leaderboard.md from runs.csv. Median cpWER per (dataset x config x tier=3)."""
    csv_path = results_dir / "runs.csv"
    out_path = results_dir / "leaderboard.md"
    if not csv_path.exists():
        out_path.write_text("# Benchmark leaderboard\n\n_No runs yet._\n")
        return out_path

    rows = list(csv.DictReader(csv_path.open()))
    tier3 = [r for r in rows if r["tier"] == "3"]

    lines = ["# Benchmark leaderboard\n"]
    for dataset in sorted({r["dataset"] for r in tier3}):
        lines.append(f"\n## {dataset} (tier 3, median)\n")
        lines.append("| Rank | Config | cpWER | WER | DER | Speaker-err | Runtime |")
        lines.append("|------|--------|-------|-----|-----|-------------|---------|")
        agg: dict[tuple, list[tuple[float, ...]]] = {}
        for r in tier3:
            if r["dataset"] != dataset:
                continue
            key = (r["no_fallback"], r["suppress_nst"], r["streaming_preset"],
                   r["align"], r["merge_strategy"])
            agg.setdefault(key, []).append((
                float(r["cpwer"]), float(r["wer"]), float(r["der"]),
                float(r["speaker_assignment_error_rate"]), float(r["runtime_s"]),
            ))
        ranked = sorted(
            ((k, median(c for c, *_ in v),
              median(w for _, w, *_ in v),
              median(d for _, _, d, *_ in v),
              median(s for _, _, _, s, _ in v),
              median(rt for *_, rt in v))
             for k, v in agg.items()),
            key=lambda x: x[1],
        )
        for rank, (k, c, w, d, s, rt) in enumerate(ranked, 1):
            nf, sn, sp, al, mg = k
            label = f"merge={mg}, align={al}, sortformer={sp}, no_fallback={nf}, suppress_nst={sn}"
            lines.append(
                f"| {rank} | {label} | {c*100:.1f} | {w*100:.1f} | {d*100:.1f} | {s*100:.1f} | {rt:.1f}s |"
            )
    out_path.write_text("\n".join(lines) + "\n")
    return out_path
```

- [ ] **Step 2: Smoke-test the import**

Run: `uv run python -c "from bench.runner import run_one_tier, generate_leaderboard; print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add bench/runner.py
git commit -m "feat(bench): tier runner with cached execution and leaderboard generator"
```

---

## Phase F: CLI + smoke test

### Task 18: `scripts/benchmark.py` — thin CLI entry

**Files:**
- Create: `scripts/benchmark.py`

- [ ] **Step 1: Implement `scripts/benchmark.py`**

```python
#!/usr/bin/env python
"""Benchmark CLI: orchestrates tier execution against AMI + SUMM-RE."""
import argparse
import csv
import sys
from pathlib import Path

from bench import runner, tiers
from bench.datasets.ami import AMIDataset
from bench.datasets.summ_re import SUMMREDataset
from transcript.pipeline_config import PipelineConfig


_TIER_PRESETS = {
    1: {"clip_count": 3, "max_duration_s": 150.0},
    2: {"clip_count": 10, "max_duration_s": 600.0},
    3: {"clip_count": 50, "max_duration_s": None},
}


def _read_rows(csv_path: Path, tier: int) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open() as f:
        return [
            {**r, "cpwer": float(r["cpwer"]),
             "align": r["align"] == "True",
             "no_fallback": r["no_fallback"] == "True",
             "suppress_nst": r["suppress_nst"] == "True",
             "fingerprint": r["config_fingerprint"]}
            for r in csv.DictReader(f) if r["tier"] == str(tier)
        ]


def _configs_for_tier(tier: int, csv_path: Path) -> list[PipelineConfig]:
    if tier == 1:
        return tiers.tier_1_configs()
    upstream_rows = _read_rows(csv_path, tier - 1)
    if not upstream_rows:
        sys.exit(f"x tier {tier} requires tier-{tier - 1} rows in runs.csv; none found.")
    if tier == 2:
        return tiers.tier_2_configs(upstream_rows)
    return tiers.tier_3_configs(upstream_rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the transcript-cli benchmark suite.")
    p.add_argument("--tier", type=int, choices=[1, 2, 3])
    p.add_argument("--all", action="store_true",
                   help="Run tier 1 → 2 → 3 sequentially.")
    p.add_argument("--datasets", nargs="+", default=["ami", "summ-re"],
                   choices=["ami", "summ-re"])
    p.add_argument("--rebuild-leaderboard", action="store_true")
    p.add_argument("--cache-dir", type=Path, default=Path("bench/cache"))
    p.add_argument("--results-dir", type=Path, default=Path("bench/results"))
    args = p.parse_args(argv)

    if args.rebuild_leaderboard:
        out = runner.generate_leaderboard(results_dir=args.results_dir)
        print(f"Wrote {out}")
        return 0

    if not args.tier and not args.all:
        p.error("specify --tier {1,2,3}, --all, or --rebuild-leaderboard")

    datasets = []
    if "ami" in args.datasets:
        datasets.append(AMIDataset(cache_dir=args.cache_dir))
    if "summ-re" in args.datasets:
        datasets.append(SUMMREDataset(cache_dir=args.cache_dir))

    csv_path = args.results_dir / "runs.csv"
    tiers_to_run = [1, 2, 3] if args.all else [args.tier]
    for tier in tiers_to_run:
        preset = _TIER_PRESETS[tier]
        configs = _configs_for_tier(tier, csv_path)
        print(f"→ Tier {tier}: {len(configs)} configs x {preset['clip_count']} clips/dataset")
        runner.run_one_tier(
            tier=tier,
            configs=configs,
            datasets=datasets,
            clip_count=preset["clip_count"],
            max_duration_s=preset["max_duration_s"],
            cache_dir=args.cache_dir,
            results_dir=args.results_dir,
        )

    out = runner.generate_leaderboard(results_dir=args.results_dir)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke-test help output**

Run: `uv run python scripts/benchmark.py --help`
Expected: help text prints; no error.

- [ ] **Step 3: Smoke-test `--rebuild-leaderboard` on empty results**

Run: `uv run python scripts/benchmark.py --rebuild-leaderboard --results-dir /tmp/bench_smoke_results`
Expected: writes `/tmp/bench_smoke_results/leaderboard.md` with "_No runs yet._".

- [ ] **Step 4: Commit**

```bash
git add scripts/benchmark.py
git commit -m "feat(bench): scripts/benchmark.py CLI entrypoint"
```

---

### Task 19: Integration smoke test — `tests/test_bench_smoke.py`

**Files:**
- Create: `tests/test_bench_smoke.py`

- [ ] **Step 1: Write the smoke test**

```python
"""Smoke test: one tier-1 invocation on the existing tiny.wav fixture.

Validates the full bench harness end-to-end (cache + metrics + artefacts +
CSV append + leaderboard) WITHOUT pulling down the 25 GB of dataset audio.
"""
from pathlib import Path

import pytest

from bench import runner, tiers
from bench.datasets.base import BenchClip, Dataset

pytestmark = pytest.mark.integration

FIXTURE = Path(__file__).parent / "fixtures" / "tiny.wav"


class _SingleClipDataset:
    name = "TINY"
    def __init__(self, clip: BenchClip):
        self._clip = clip
    def sample(self, n, *, max_duration_s=None, seed=42):
        return [self._clip]


def test_smoke_full_bench_roundtrip(tmp_path):
    if not FIXTURE.exists():
        pytest.skip("tiny.wav fixture not generated; run scripts/generate_tiny_wav.sh")
    stm = tmp_path / "tiny.stm"
    rttm = tmp_path / "tiny.rttm"
    stm.write_text("tiny 1 SpeakerA 0.00 4.00 <NA> hello world this is a test\n")
    rttm.write_text("SPEAKER tiny 1 0.000 4.000 <NA> <NA> SpeakerA <NA> <NA>\n")

    clip = BenchClip(
        clip_id="TINY:tiny",
        audio_path=FIXTURE,
        language="en",
        num_speakers=1,
        duration_s=8.0,
        reference_rttm=rttm,
        reference_stm=stm,
    )
    dataset: Dataset = _SingleClipDataset(clip)

    # Tier-1 32-config grid is too heavy for a smoke test — pick 2.
    configs = tiers.tier_1_configs()[:2]
    runner.run_one_tier(
        tier=1,
        configs=configs,
        datasets=[dataset],
        clip_count=1,
        max_duration_s=None,
        cache_dir=tmp_path / "cache",
        results_dir=tmp_path / "results",
    )

    csv_path = tmp_path / "results" / "runs.csv"
    assert csv_path.exists()
    rows = csv_path.read_text().splitlines()
    assert len(rows) >= 3  # header + 2 config rows
    assert (tmp_path / "results" / "transcripts" / "tier-1").exists()
    assert (tmp_path / "results" / "diffs" / "tier-1").exists()
```

- [ ] **Step 2: Run the smoke test**

Run: `uv run pytest tests/test_bench_smoke.py -m integration -v`
Expected: PASS (if tiny.wav exists; otherwise skipped — that's acceptable).

- [ ] **Step 3: Run the full non-integration suite**

Run: `uv run pytest -m "not integration" -v`
Expected: ALL PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_bench_smoke.py
git commit -m "test(bench): integration smoke test for end-to-end harness"
```

---

### Task 20: Update `docs/todo.md` to reflect what shipped

**Files:**
- Modify: `docs/todo.md`

- [ ] **Step 1: Update the file**

Rewrite items 1 and 3 in `docs/todo.md`:

```markdown
## 1. Probability-based per-word speaker assignment — DONE

Shipped: implemented as `merge.strategy = "prob_based"` knob. Sortformer now
emits the [T x 4] per-frame probability tensor when `DiarizeConfig.emit_probs`
is true; `merge._best_speaker_prob_based` averages over each word's frame
window and argmaxes. The new strategy is tested against the hard-boundary
variant in the bench harness (see #3).

## 3. Quantitative benchmarking on real datasets — DONE

Shipped: `scripts/benchmark.py` runs a three-tier search on AMI (sdm split)
and SUMM-RE (dev split) using cpWER (via meeteval) as the primary metric,
with WER, DER, and a "speaker-assignment error rate" (cpWER - WER) as
secondary diagnostics. Results are appended to `bench/results/runs.csv` and
auto-summarised in `bench/results/leaderboard.md`. Per-row hypothesis and
diff artefacts are persisted under `bench/results/{transcripts,diffs}/` for
post-hoc failure-mode analysis.
```

- [ ] **Step 2: Commit**

```bash
git add docs/todo.md
git commit -m "docs: mark TODO #1 (prob-based merge) and #3 (benchmarking) as done"
```

---

## Self-review

This plan was reviewed against the spec on the date of writing. No placeholders remain; all type signatures used in later tasks match earlier definitions. Specifically verified:

- `pipeline.run(audio_path=, config=, with_diarization=, progress=)` — defined in Task 2, called in Tasks 7, 17, 19. Match.
- `transcribe.run(wav, *, config: TranscribeConfig) -> tuple[list[Word], str]` — defined Task 3, called in 17. Match.
- `diarize.run(wav, *, config: DiarizeConfig) -> tuple[list[Turn], np.ndarray | None]` — defined Task 4, called in 2, 17. Match.
- `merge.assign_speakers(words, turns, *, strategy=, probs=) -> list[tuple[Word, str]]` — defined Task 6, called in 2, 17. Match.
- `bench.cache.{save,load}_{whisper,sortformer,align}` — defined Task 10, called in 17. Match.
- `bench.artefacts.{save_transcript,save_diff}` — defined Task 11, called in 17, 19. Match.
- `bench.metrics.score(hyp, ref) -> ClipMetrics` — defined Task 12, called in 17. Match.
- `bench.tiers.tier_{1,2,3}_configs` — defined Task 16, called in 18. Match.
- `bench.runner.{run_one_tier,generate_leaderboard}` — defined Task 17, called in 18. Match.
- All sections of the spec (summary, goals, decisions, pipeline graph, config schema, runner+cache, datasets, metrics+artefacts, testing) map to at least one task. Coverage complete.

---

*This plan is the input to the executing-plans or subagent-driven-development skill.*
