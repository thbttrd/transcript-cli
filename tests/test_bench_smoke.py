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

    # Tier-1 16-config grid is too heavy for a smoke test — pick 2.
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
