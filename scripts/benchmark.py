#!/usr/bin/env python
"""Benchmark CLI: orchestrates tier execution against AMI + SUMM-RE."""
import argparse
import csv
import sys
from pathlib import Path

# `bench/` lives at the repo root and isn't wheel-packaged. Make it importable
# whether the script is invoked from the repo root or elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import runner, tiers  # noqa: E402
from bench.datasets.ami import AMIDataset  # noqa: E402
from bench.datasets.summ_re import SUMMREDataset  # noqa: E402
from transcript.pipeline_config import PipelineConfig  # noqa: E402

_TIER_PRESETS = {
    1: {"clip_count": 3,  "max_duration_s": 300.0},   # 5-min clips, quick smoke
    2: {"clip_count": 10, "max_duration_s": 900.0},   # 15-min clips, narrow axes
    3: {"clip_count": 50, "max_duration_s": None},    # full meetings, finalists
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
        args.results_dir.mkdir(parents=True, exist_ok=True)
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
