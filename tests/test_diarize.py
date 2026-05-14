import sys
import types
from pathlib import Path

import pytest

from transcript import diarize
from transcript.pipeline_config import DiarizeConfig


def test_parse_sortformer_segments_handles_rttm_lines():
    raw = diarize._parse_sortformer_segments([
        "0.50 2.66 speaker_0",
        "4.28 5.73 speaker_1",
        "junk",  # ignored
        "1.0 not-a-float speaker_0",  # ignored
    ])
    assert raw == [
        (0.50, 2.66, "speaker_0"),
        (4.28, 5.73, "speaker_1"),
    ]


@pytest.fixture
def reset_model_cache(monkeypatch):
    """Reset the diarize._model_cache module-level cache around each test."""
    monkeypatch.setattr(diarize, "_model_cache", {})


def _inject_fake_nemo(monkeypatch, sortformer_class):
    """Install a fake `nemo.collections.asr.models` in sys.modules so the lazy
    `from nemo... import SortformerEncLabelModel` resolves to our fake without
    ever importing real NeMo. Only the leaf is needed: `from X.Y.Z import …`
    short-circuits when the dotted name is already in sys.modules and never
    walks the parents."""
    fake_models = types.ModuleType("nemo.collections.asr.models")
    fake_models.SortformerEncLabelModel = sortformer_class
    monkeypatch.setitem(sys.modules, "nemo.collections.asr.models", fake_models)


def test_load_model_caches_across_calls(reset_model_cache, monkeypatch, mocker):
    fake_class = mocker.MagicMock()
    fake_class.from_pretrained.return_value = mocker.MagicMock()
    _inject_fake_nemo(monkeypatch, fake_class)

    m1 = diarize._load_model()
    m2 = diarize._load_model()

    assert m1 is m2
    assert fake_class.from_pretrained.call_count == 1


def test_load_model_raises_diarize_error_when_nemo_missing(reset_model_cache, monkeypatch):
    # Setting None in sys.modules makes `import X` raise ModuleNotFoundError,
    # which our handler catches and rewraps as DiarizeError with install advice.
    monkeypatch.setitem(sys.modules, "nemo.collections.asr.models", None)

    with pytest.raises(diarize.DiarizeError, match="scripts/install.sh"):
        diarize._load_model()


def test_streaming_params_for_very_high_lat_preset_match_nvidia_values():
    params = diarize._streaming_params("very_high_lat")
    assert params == {
        "chunk_len": 340,
        "chunk_right_context": 40,
        "fifo_len": 40,
        "spkcache_update_period": 340,
        "spkcache_len": 188,
    }


def test_streaming_params_for_low_lat_preset_match_nvidia_values():
    params = diarize._streaming_params("low_lat")
    assert params == {
        "chunk_len": 6,
        "chunk_right_context": 7,
        "fifo_len": 188,
        "spkcache_update_period": 144,
        "spkcache_len": 188,
    }


def test_run_filters_by_num_speakers(reset_model_cache, monkeypatch, mocker):
    fake_model = mocker.MagicMock()
    fake_model.diarize.return_value = [[
        "0.0 1.0 spk_a",
        "1.0 2.0 spk_b",
        "2.0 3.0 spk_c",
    ]]
    fake_class = mocker.MagicMock()
    fake_class.from_pretrained.return_value = fake_model
    _inject_fake_nemo(monkeypatch, fake_class)

    cfg = DiarizeConfig(num_speakers=2)
    turns = diarize.run(Path("/fake.wav"), config=cfg)
    assert {t.speaker for t in turns} == {"Speaker 1", "Speaker 2"}
