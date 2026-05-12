"""Unit tests for the bench runner — focused on the leaderboard generator.

The full end-to-end runner.run_one_tier path is exercised only by the
integration smoke test (tests/test_bench_smoke.py). This file covers the
deterministic, no-pipeline parts: CSV → markdown.
"""
from bench import runner

CSV_HEADER = ",".join(runner.CSV_COLUMNS)


def _row(**kw) -> str:
    """Build one CSV row using sensible defaults for the columns we don't care about."""
    defaults = {
        "tier": "3", "dataset": "AMI", "clip_id": "AMI:m1",
        "config_id": "abc", "config_fingerprint": "abc",
        "no_fallback": "True", "suppress_nst": "True",
        "streaming_preset": "very_high_lat", "align": "True",
        "merge_strategy": "hard_boundary",
        "cpwer": "0.10", "wer": "0.08", "der": "0.05",
        "speaker_assignment_error_rate": "0.02",
        "runtime_s": "1.5", "whisper_s": "1.0", "sortformer_s": "0.4",
        "align_s": "0.05", "merge_s": "0.05",
        "git_sha": "deadbee", "started_at": "0", "host": "localhost",
        "hypothesis_path": "transcripts/tier-3/AMI_m1/abc.json",
        "diff_path": "diffs/tier-3/AMI_m1/abc.json",
    }
    defaults.update(kw)
    return ",".join(defaults[c] for c in runner.CSV_COLUMNS)


def test_generate_leaderboard_writes_no_runs_yet_on_empty_dir(tmp_path):
    out = runner.generate_leaderboard(results_dir=tmp_path)
    assert out == tmp_path / "leaderboard.md"
    assert "_No runs yet._" in out.read_text()


def test_generate_leaderboard_ranks_tier3_configs_by_median_cpwer(tmp_path):
    csv_path = tmp_path / "runs.csv"
    csv_path.write_text(
        "\n".join([
            CSV_HEADER,
            _row(merge_strategy="prob_based",   cpwer="0.05", config_fingerprint="p"),
            _row(merge_strategy="hard_boundary", cpwer="0.10", config_fingerprint="h"),
        ]) + "\n"
    )
    out = runner.generate_leaderboard(results_dir=tmp_path)
    md = out.read_text()
    assert "# Benchmark leaderboard" in md
    assert "## AMI (tier 3, median)" in md
    # The lower cpWER (prob_based) should rank first.
    prob_idx = md.find("merge=prob_based")
    hard_idx = md.find("merge=hard_boundary")
    assert prob_idx >= 0 and hard_idx >= 0
    assert prob_idx < hard_idx


def test_generate_leaderboard_skips_non_tier3_rows(tmp_path):
    csv_path = tmp_path / "runs.csv"
    csv_path.write_text(
        "\n".join([
            CSV_HEADER,
            _row(tier="1", cpwer="0.05"),
            _row(tier="2", cpwer="0.05"),
        ]) + "\n"
    )
    out = runner.generate_leaderboard(results_dir=tmp_path)
    md = out.read_text()
    # Header still there, but no per-dataset section.
    assert "# Benchmark leaderboard" in md
    assert "## AMI" not in md
