"""Unit tests for the bench runner — leaderboard rendering, CSV schema
invariants, and the run_one_tier dispatch (with pipeline stages mocked).
"""
import csv
from pathlib import Path

import pytest

from bench import runner
from bench.datasets.base import BenchClip, Dataset
from transcript.models import Turn, Word
from transcript.pipeline_config import PipelineConfig

CSV_HEADER = ",".join(runner.CSV_COLUMNS)


def _row(**kw) -> str:
    """Build one CSV row using sensible defaults for the columns we don't care about."""
    defaults = {
        "tier": "2", "dataset": "AMI", "clip_id": "AMI:m1",
        "config_id": "abc", "config_fingerprint": "abc",
        "no_fallback": "True", "suppress_nst": "True",
        "streaming_preset": "very_high_lat", "align": "True",
        "cpwer": "0.10", "wer": "0.08", "der": "0.05",
        "speaker_assignment_error_rate": "0.02",
        "runtime_s": "1.5", "whisper_s": "1.0", "sortformer_s": "0.4",
        "align_s": "0.05", "merge_s": "0.05",
        "git_sha": "deadbee", "started_at": "0", "host": "localhost",
        "hypothesis_path": "transcripts/tier-2/AMI_m1/abc.json",
        "diff_path": "diffs/tier-2/AMI_m1/abc.json",
    }
    defaults.update(kw)
    return ",".join(defaults[c] for c in runner.CSV_COLUMNS)


def test_generate_leaderboard_writes_no_runs_yet_on_empty_dir(tmp_path):
    out = runner.generate_leaderboard(results_dir=tmp_path)
    assert out == tmp_path / "leaderboard.md"
    assert "_No runs yet._" in out.read_text()


def test_generate_leaderboard_ranks_configs_by_median_cpwer(tmp_path):
    csv_path = tmp_path / "runs.csv"
    csv_path.write_text(
        "\n".join([
            CSV_HEADER,
            _row(streaming_preset="low_lat",        cpwer="0.05", config_fingerprint="l"),
            _row(streaming_preset="very_high_lat",  cpwer="0.10", config_fingerprint="v"),
        ]) + "\n"
    )
    out = runner.generate_leaderboard(results_dir=tmp_path)
    md = out.read_text()
    assert "# Benchmark leaderboard (tier 2, median)" in md
    assert "## AMI" in md
    # The lower cpWER (low_lat) should rank first.
    low_idx = md.find("sortformer=low_lat")
    high_idx = md.find("sortformer=very_high_lat")
    assert low_idx >= 0 and high_idx >= 0
    assert low_idx < high_idx


class _OneClipDataset:
    name = "TINY"

    def __init__(self, clip: BenchClip):
        self._clip = clip

    def sample(self, n, *, max_duration_s=None, seed=42):
        return [self._clip]


def _make_clip(tmp_path: Path) -> BenchClip:
    wav = tmp_path / "fake.wav"
    wav.write_bytes(b"not-real-audio-just-hashable-bytes")
    stm = tmp_path / "fake.stm"
    stm.write_text("TINY 1 SpeakerA 0.00 1.00 <NA> hello\n")
    rttm = tmp_path / "fake.rttm"
    rttm.write_text("SPEAKER TINY 1 0.000 1.000 <NA> <NA> SpeakerA <NA> <NA>\n")
    return BenchClip(
        clip_id="TINY:fake", audio_path=wav, language="en",
        num_speakers=1, duration_s=1.0,
        reference_rttm=rttm, reference_stm=stm,
    )


def test_run_one_tier_writes_csv_with_expected_schema(tmp_path, mocker):
    clip = _make_clip(tmp_path)
    dataset: Dataset = _OneClipDataset(clip)
    mocker.patch(
        "bench.runner.transcribe.run",
        return_value=([Word(text=" hello", start=0.0, end=1.0)], "en"),
    )
    mocker.patch(
        "bench.runner.diarize.run",
        return_value=[Turn("Speaker 1", 0.0, 1.0)],
    )
    mocker.patch("bench.runner.align_mod.is_available", return_value=False)

    runner.run_one_tier(
        tier=1, configs=[PipelineConfig()], datasets=[dataset],
        clip_count=1, max_duration_s=None,
        cache_dir=tmp_path / "cache", results_dir=tmp_path / "results",
    )

    csv_path = tmp_path / "results" / "runs.csv"
    assert csv_path.exists()
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 1
    assert set(rows[0].keys()) == set(runner.CSV_COLUMNS)
    assert rows[0]["dataset"] == "TINY"
    assert rows[0]["clip_id"] == "TINY:fake"
    assert rows[0]["tier"] == "1"
    # Stage timings: first run is a cache miss, so the actual run time was
    # recorded for whisper/sortformer; align was skipped (is_available=False).
    assert float(rows[0]["whisper_s"]) >= 0.0
    assert float(rows[0]["sortformer_s"]) >= 0.0
    assert float(rows[0]["align_s"]) == runner.CACHED_STAGE_S


def test_run_one_tier_refuses_to_append_to_stale_header(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    # Stale header with the old merge_strategy column from the pre-revert
    # schema — exactly what bench/results/runs.v1.csv looks like.
    stale_header = (
        "tier,dataset,clip_id,config_id,config_fingerprint,no_fallback,"
        "suppress_nst,streaming_preset,align,merge_strategy,cpwer,wer,der,"
        "speaker_assignment_error_rate,runtime_s,whisper_s,sortformer_s,"
        "align_s,merge_s,git_sha,started_at,host,hypothesis_path,diff_path"
    )
    (results_dir / "runs.csv").write_text(stale_header + "\n")

    with pytest.raises(RuntimeError, match="header mismatch"):
        runner.run_one_tier(
            tier=1, configs=[], datasets=[], clip_count=0,
            max_duration_s=None,
            cache_dir=tmp_path / "cache", results_dir=results_dir,
        )


def test_generate_leaderboard_uses_highest_tier_present(tmp_path):
    """When the user only ran Tier 1, the leaderboard should still be meaningful
    by aggregating the highest tier that DID run (tier 1 here)."""
    csv_path = tmp_path / "runs.csv"
    csv_path.write_text(
        "\n".join([
            CSV_HEADER,
            _row(tier="1", cpwer="0.05"),
            _row(tier="2", cpwer="0.10"),
        ]) + "\n"
    )
    out = runner.generate_leaderboard(results_dir=tmp_path)
    md = out.read_text()
    # Uses tier-2 rows (highest tier present)
    assert "# Benchmark leaderboard (tier 2, median)" in md
    assert "## AMI" in md
    # The leaderboard contains exactly one ranked row (tier-2 only).
    body_lines = [line for line in md.splitlines() if line.startswith("| 1 ")]
    assert len(body_lines) == 1
    # That row's cpWER cell is the tier-2 value (0.10 → "10.0"), not the
    # tier-1 value (0.05 → "5.0"). The cpWER column is the 3rd cell.
    cpwer_cell = body_lines[0].split("|")[3].strip()
    assert cpwer_cell == "10.0"
