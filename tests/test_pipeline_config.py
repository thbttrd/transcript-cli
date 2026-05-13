import dataclasses
from dataclasses import asdict

import pytest

from transcript.pipeline_config import (
    AlignConfig,
    DiarizeConfig,
    LLMFixConfig,
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
    assert cfg.align.enabled is True
    assert cfg.llm_fix.enabled is False


def test_fingerprint_is_stable_for_same_config():
    cfg_a = PipelineConfig()
    cfg_b = PipelineConfig()
    assert cfg_a.fingerprint() == cfg_b.fingerprint()


def test_fingerprint_changes_when_any_field_changes():
    base = PipelineConfig().fingerprint()
    changed = PipelineConfig(
        diarize=DiarizeConfig(streaming_preset="low_lat")
    ).fingerprint()
    assert base != changed


def test_fingerprint_is_short_hex_string():
    fp = PipelineConfig().fingerprint()
    assert len(fp) == 12
    assert all(c in "0123456789abcdef" for c in fp)


def test_from_dict_roundtrips_via_asdict():
    cfg = PipelineConfig(
        transcribe=TranscribeConfig(no_fallback=False, suppress_nst=False),
        align=AlignConfig(enabled=False),
        llm_fix=LLMFixConfig(enabled=True),
    )
    reconstructed = PipelineConfig.from_dict(asdict(cfg))
    assert reconstructed == cfg


def test_dataclasses_are_frozen():
    cfg = PipelineConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.transcribe.model = "tiny"
