"""Tier config generators.

Tier 1: full Cartesian product of the 5 tunable axes (whisper model is pinned
        at large-v3; llm_fix is pinned at False). 32 configs.
Tier 2: drop axes whose effect size (max-min cpWER across the axis's values)
        is below 0.5 absolute cpWER points; product over the rest.
Tier 3: configs whose median cpWER <= best + 1.0 absolute points (capped at 5);
        relax to 2.0 points if fewer than 3 qualify.
"""
from collections.abc import Iterable
from itertools import product

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
    pinned = {k: best[k] for k in _AXES_BOOL}
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
        for k, v in zip(keys, combo, strict=True):
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


def _build_config(no_fallback: bool = True, suppress_nst: bool = True,
                  streaming_preset: str = "very_high_lat", align: bool = True,
                  merge_strategy: str = "hard_boundary") -> PipelineConfig:
    return PipelineConfig(
        transcribe=TranscribeConfig(no_fallback=no_fallback, suppress_nst=suppress_nst),
        diarize=DiarizeConfig(streaming_preset=streaming_preset),
        align=AlignConfig(enabled=align),
        merge=MergeConfig(strategy=merge_strategy),
    )
