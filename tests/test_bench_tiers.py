from bench.tiers import tier_1_configs, tier_2_configs
from transcript.pipeline_config import PipelineConfig


def test_tier_1_generates_full_grid_of_16():
    configs = tier_1_configs()
    assert len(configs) == 16
    assert all(isinstance(c, PipelineConfig) for c in configs)
    fingerprints = {c.fingerprint() for c in configs}
    assert len(fingerprints) == 16  # all distinct


def test_tier_1_pins_whisper_model_to_large_v3():
    for c in tier_1_configs():
        assert c.transcribe.model == "large-v3"
        assert c.llm_fix.enabled is False


def test_tier_2_drops_axes_with_low_effect_size():
    # Synthetic tier-1 rows: only streaming_preset moves the needle.
    tier_1_rows = [
        {"no_fallback": True,  "suppress_nst": True,  "streaming_preset": "very_high_lat",
         "align": True, "cpwer": 0.10},
        {"no_fallback": True,  "suppress_nst": True,  "streaming_preset": "low_lat",
         "align": True, "cpwer": 0.08},
        {"no_fallback": False, "suppress_nst": True,  "streaming_preset": "very_high_lat",
         "align": True, "cpwer": 0.10},
        {"no_fallback": True,  "suppress_nst": False, "streaming_preset": "very_high_lat",
         "align": True, "cpwer": 0.10},
        {"no_fallback": True,  "suppress_nst": True,  "streaming_preset": "very_high_lat",
         "align": False, "cpwer": 0.10},
    ]
    configs = tier_2_configs(tier_1_rows)
    assert len(configs) == 2
    presets = {c.diarize.streaming_preset for c in configs}
    assert presets == {"very_high_lat", "low_lat"}
