"""AMI loader unit tests.

Three concerns:
1. RTTM resolver layout handling (BUT nested `only_words/rttms/test/`).
2. Per-utterance → meeting WAV splicing (the HF dataset ships per-utterance
   audio; the pipeline expects a single meeting-level WAV).
3. Truncation: tiers cap meetings to a fixed duration so per-tier total
   compute is bounded.
"""
from pathlib import Path

import numpy as np
import soundfile as sf

from bench.datasets.ami import AMIDataset, _build_meeting_wav, _truncate_rttm


def _touch_rttm(dir_: Path, name: str) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    p = dir_ / f"{name}.rttm"
    p.write_text("SPEAKER m1 1 0.0 1.0 <NA> <NA> A <NA> <NA>\n")
    return p


def test_resolve_rttm_dir_prefers_vendored_when_populated(tmp_path, monkeypatch):
    """If the vendored package dir has *.rttm files, prefer it — no cache lookup."""
    vendored = tmp_path / "vendored_ami_rttm"
    _touch_rttm(vendored, "ES2002a")
    monkeypatch.setattr(
        "bench.datasets.ami._vendored_rttm_dir", lambda: vendored
    )

    cache_dir = tmp_path / "cache"  # never populated
    result = AMIDataset._resolve_rttm_dir(cache_dir)
    assert result == vendored


def test_resolve_rttm_dir_descends_into_but_nested_layout(tmp_path, monkeypatch):
    """When the BUT repo has been cloned (or pre-populated) into the cache,
    the resolver must return the nested `only_words/rttms/` directory, not
    the clone root."""
    empty_vendored = tmp_path / "empty_vendored"
    empty_vendored.mkdir()
    monkeypatch.setattr(
        "bench.datasets.ami._vendored_rttm_dir", lambda: empty_vendored
    )

    cache_dir = tmp_path / "cache"
    runtime = cache_dir / "ami_rttm"
    nested = runtime / "only_words" / "rttms"
    _touch_rttm(nested, "ES2002a")
    _touch_rttm(nested, "IS1009b")

    result = AMIDataset._resolve_rttm_dir(cache_dir)
    assert result == nested
    assert (result / "ES2002a.rttm").exists()
    assert (result / "IS1009b.rttm").exists()


def test_resolve_rttm_dir_descends_into_but_split_subdir(tmp_path, monkeypatch):
    """The BUT repo nests per-meeting RTTMs under
    `only_words/rttms/<split>/<meeting_id>.rttm` where <split> ∈ {train,dev,test}.
    The split root (`only_words/rttms/`) only contains concatenated *.rttm
    files (`dev.rttm`, `test.rttm`, `train.rttm`), which are not per-meeting.
    The resolver must descend into the `test/` split subdir since the AMI
    loader hardcodes `split=\"test\"`."""
    empty_vendored = tmp_path / "empty_vendored"
    empty_vendored.mkdir()
    monkeypatch.setattr(
        "bench.datasets.ami._vendored_rttm_dir", lambda: empty_vendored
    )

    cache_dir = tmp_path / "cache"
    runtime = cache_dir / "ami_rttm"
    rttms_root = runtime / "only_words" / "rttms"
    test_dir = rttms_root / "test"
    # Per-meeting RTTMs live in test/
    _touch_rttm(test_dir, "ES2002a")
    _touch_rttm(test_dir, "IS1009b")
    # Concatenated split-level files live at the rttms_root level — these
    # match `only_words/rttms/*.rttm` glob but aren't per-meeting RTTMs.
    (rttms_root / "test.rttm").write_text("SPEAKER ES2002a 1 0 1 <NA> <NA> A <NA> <NA>\n")
    (rttms_root / "dev.rttm").write_text("")
    (rttms_root / "train.rttm").write_text("")
    # The dev/ and train/ split subdirs also exist with their own meetings,
    # but we want the test split since AMI loader pins to test.
    _touch_rttm(rttms_root / "dev", "ES2003a")
    _touch_rttm(rttms_root / "train", "ES2004a")

    result = AMIDataset._resolve_rttm_dir(cache_dir)
    assert result == test_dir
    assert (result / "ES2002a.rttm").exists()
    assert (result / "IS1009b.rttm").exists()


def test_resolve_rttm_dir_falls_back_to_flat_runtime_layout(tmp_path, monkeypatch):
    """If the runtime dir already has flat RTTMs (manually placed), accept it."""
    empty_vendored = tmp_path / "empty_vendored"
    empty_vendored.mkdir()
    monkeypatch.setattr(
        "bench.datasets.ami._vendored_rttm_dir", lambda: empty_vendored
    )

    cache_dir = tmp_path / "cache"
    runtime = cache_dir / "ami_rttm"
    _touch_rttm(runtime, "ES2002a")

    result = AMIDataset._resolve_rttm_dir(cache_dir)
    assert result == runtime


# -- Per-utterance → meeting WAV splicing ----------------------------------

def _utterance(meeting_id, speaker, begin, end, samples, sr=16000):
    """Build a fake AMI row matching what HF streaming returns."""
    return {
        "meeting_id": meeting_id,
        "speaker_id": speaker,
        "begin_time": begin,
        "end_time": end,
        "text": "hi",
        "audio": {
            "path": f"{meeting_id}_{speaker}_{int(begin*100):07d}_{int(end*100):07d}.wav",
            "array": samples.astype(np.float32),
            "sampling_rate": sr,
        },
    }


def test_build_meeting_wav_places_utterances_at_correct_offsets(tmp_path, monkeypatch):
    """Two utterances spliced into a single meeting WAV must land at the
    sample offsets implied by their begin_time fields."""
    sr = 16000
    # Utterance A: 0.0 → 0.5s, all ones
    a = np.ones(int(0.5 * sr), dtype=np.float32)
    # Utterance B: 1.0 → 1.5s, all 0.5s (distinguishable from silence and A)
    b = np.full(int(0.5 * sr), 0.5, dtype=np.float32)
    rows = [
        _utterance("ES2002a", "A", 0.0, 0.5, a, sr),
        _utterance("ES2002a", "B", 1.0, 1.5, b, sr),
        _utterance("OTHER", "X", 0.0, 0.5, a, sr),  # ignored — different meeting
    ]
    monkeypatch.setattr(
        "bench.datasets.ami._load_dataset", lambda *a, **kw: rows
    )

    out = tmp_path / "ES2002a.wav"
    _build_meeting_wav("ES2002a", out)

    data, got_sr = sf.read(out)
    assert got_sr == sr
    assert data.ndim == 1
    # 1.5s of meeting → 1.5 * sr samples (plus or minus tail trimming slack)
    assert abs(len(data) - int(1.5 * sr)) < sr  # within 1 second
    # Sample 1000 (well inside utterance A) is ~1.0
    assert data[1000] > 0.9
    # Sample at 0.75s (silent gap between A and B) is ~0.0
    assert abs(data[int(0.75 * sr)]) < 0.01
    # Sample at 1.1s (inside utterance B) is ~0.5
    assert abs(data[int(1.1 * sr)] - 0.5) < 0.1


def test_build_meeting_wav_handles_overlapping_utterances(tmp_path, monkeypatch):
    """First-writer-wins for overlaps: AMI sdm rows in overlap regions are
    slices of the same source audio, so doubling the amplitude (e.g. by sum)
    would be wrong. Earlier begin_time gets to write."""
    sr = 16000
    a = np.full(int(0.5 * sr), 0.3, dtype=np.float32)  # 0.0 → 0.5
    b = np.full(int(0.5 * sr), 0.4, dtype=np.float32)  # 0.2 → 0.7, overlaps A
    rows = [
        _utterance("M1", "A", 0.0, 0.5, a, sr),
        _utterance("M1", "B", 0.2, 0.7, b, sr),
    ]
    monkeypatch.setattr(
        "bench.datasets.ami._load_dataset", lambda *a, **kw: rows
    )

    out = tmp_path / "M1.wav"
    _build_meeting_wav("M1", out)

    data, _ = sf.read(out)
    # A wrote [0.0, 0.5] first → buf[0.3s] is A's value (0.3), not 0.7.
    assert abs(data[int(0.3 * sr)] - 0.3) < 0.05
    # B writes only the non-overlapping tail [0.5, 0.7] → buf[0.6s] is 0.4.
    assert abs(data[int(0.6 * sr)] - 0.4) < 0.05


def test_build_meeting_wav_truncates_to_max_duration(tmp_path, monkeypatch):
    """When max_duration_s is set the output WAV is clipped to that length;
    utterances starting after the cap don't appear in the buffer."""
    sr = 16000
    a = np.ones(int(0.5 * sr), dtype=np.float32)  # 0.0 → 0.5s
    b = np.full(int(0.5 * sr), 0.5, dtype=np.float32)  # 2.0 → 2.5s, past cap
    rows = [
        _utterance("M1", "A", 0.0, 0.5, a, sr),
        _utterance("M1", "B", 2.0, 2.5, b, sr),
    ]
    monkeypatch.setattr(
        "bench.datasets.ami._load_dataset", lambda *a, **kw: rows
    )

    out = tmp_path / "M1.wav"
    _build_meeting_wav("M1", out, max_duration_s=1.0)

    data, got_sr = sf.read(out)
    assert got_sr == sr
    assert len(data) == int(1.0 * sr)
    # Utterance A at 0.1s is preserved
    assert data[int(0.1 * sr)] > 0.9


def test_build_meeting_wav_raises_for_unexpected_sample_rate(tmp_path, monkeypatch):
    rows = [
        _utterance("M1", "A", 0.0, 0.5, np.zeros(8000, dtype=np.float32), sr=8000),
    ]
    monkeypatch.setattr(
        "bench.datasets.ami._load_dataset", lambda *a, **kw: rows
    )

    out = tmp_path / "M1.wav"
    import pytest
    with pytest.raises(RuntimeError, match="sample rate"):
        _build_meeting_wav("M1", out)


# -- RTTM truncation -------------------------------------------------------

def test_truncate_rttm_drops_segments_starting_past_cap(tmp_path):
    src = tmp_path / "src.rttm"
    src.write_text(
        "SPEAKER M1 1 0.0 1.0 <NA> <NA> A <NA> <NA>\n"
        "SPEAKER M1 1 2.0 1.0 <NA> <NA> B <NA> <NA>\n"  # starts past cap=1.5
    )
    dst = tmp_path / "dst.rttm"
    _truncate_rttm(src, dst, max_duration_s=1.5)
    lines = [line for line in dst.read_text().splitlines() if line.strip()]
    assert len(lines) == 1
    assert lines[0].split()[7] == "A"


def test_truncate_rttm_clips_segments_crossing_the_cap(tmp_path):
    src = tmp_path / "src.rttm"
    src.write_text(
        "SPEAKER M1 1 0.0 5.0 <NA> <NA> A <NA> <NA>\n"  # 0.0 → 5.0, cap=3.0
    )
    dst = tmp_path / "dst.rttm"
    _truncate_rttm(src, dst, max_duration_s=3.0)
    parts = dst.read_text().strip().split()
    # duration is the 5th field (index 4 in zero-based)
    assert float(parts[4]) == 3.0  # clipped to cap - start = 3.0 - 0.0
    assert parts[7] == "A"
