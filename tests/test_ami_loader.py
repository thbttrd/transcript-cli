"""AMI loader unit tests — focused on the RTTM resolver layout handling.

The BUT repo nests RTTMs under `<root>/only_words/rttms/`; without descending
into that layout, `_prepare_clip` silently skips every meeting. These tests
pin the resolver to all three accepted layouts so the bug can't regress.
"""
from pathlib import Path

from bench.datasets.ami import AMIDataset


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
