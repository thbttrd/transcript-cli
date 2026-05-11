import sys
import types

import pytest

from transcript import diarize
from transcript.models import Turn


def test_relabel_assigns_speaker_n_in_first_appearance_order():
    raw = [
        (0.0, 1.0, "spk_zzz"),
        (1.0, 2.0, "spk_aaa"),
        (2.0, 3.0, "spk_zzz"),
        (3.0, 4.0, "spk_aaa"),
        (4.0, 5.0, "spk_qqq"),
    ]
    turns = diarize._relabel(raw)
    assert turns == [
        Turn("Speaker 1", 0.0, 1.0),
        Turn("Speaker 2", 1.0, 2.0),
        Turn("Speaker 1", 2.0, 3.0),
        Turn("Speaker 2", 3.0, 4.0),
        Turn("Speaker 3", 4.0, 5.0),
    ]


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
    """Reset the diarize._model module-level cache around each test."""
    monkeypatch.setattr(diarize, "_model", None)


def _inject_fake_nemo(monkeypatch, sortformer_class):
    """Install a fake `nemo.collections.asr.models` module chain in sys.modules
    so the lazy `from nemo... import SortformerEncLabelModel` resolves to our
    fake class without ever importing real NeMo."""
    for name in ("nemo", "nemo.collections", "nemo.collections.asr"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
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
