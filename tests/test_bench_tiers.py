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
