from pathlib import Path

import pytest

from transcript import diarize
from transcript.models import Turn


class _FakeTrack:
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end


class _FakeAnnotation:
    def __init__(self, items: list[tuple[float, float, str]]):
        self._items = items

    def itertracks(self, yield_label: bool = False):
        for s, e, lbl in self._items:
            if yield_label:
                yield _FakeTrack(s, e), None, lbl
            else:
                yield _FakeTrack(s, e), None


def test_relabel_assigns_speaker_n_in_first_appearance_order():
    fake = _FakeAnnotation([
        (0.0, 1.0, "spk_zzz"),
        (1.0, 2.0, "spk_aaa"),
        (2.0, 3.0, "spk_zzz"),
        (3.0, 4.0, "spk_aaa"),
        (4.0, 5.0, "spk_qqq"),
    ])
    turns = diarize._to_turns(fake)
    assert turns == [
        Turn("Speaker 1", 0.0, 1.0),
        Turn("Speaker 2", 1.0, 2.0),
        Turn("Speaker 1", 2.0, 3.0),
        Turn("Speaker 2", 3.0, 4.0),
        Turn("Speaker 3", 4.0, 5.0),
    ]


def test_run_calls_pipeline_with_num_speakers(mocker, tmp_path):
    wav = tmp_path / "x.wav"; wav.write_bytes(b"")
    mocker.patch("transcript.diarize.config.hf_token", return_value="hf_xxx")
    mock_pipe = mocker.MagicMock()
    mock_pipe.return_value = _FakeAnnotation([(0.0, 1.0, "a")])
    mock_from = mocker.patch(
        "transcript.diarize.Pipeline.from_pretrained", return_value=mock_pipe
    )
    mocker.patch("transcript.diarize.torch.device", side_effect=lambda x: x)
    mocker.patch("transcript.diarize.torch.backends.mps.is_available", return_value=True)

    diarize.run(wav, num_speakers=2)
    mock_from.assert_called_once()
    mock_pipe.to.assert_called_once_with("mps")
    # When num_speakers is set, both min and max are pinned
    _, kwargs = mock_pipe.call_args
    assert kwargs == {"min_speakers": 2, "max_speakers": 2}


def test_run_omits_speaker_kwargs_when_unspecified(mocker, tmp_path):
    wav = tmp_path / "x.wav"; wav.write_bytes(b"")
    mocker.patch("transcript.diarize.config.hf_token", return_value="hf_xxx")
    mock_pipe = mocker.MagicMock()
    mock_pipe.return_value = _FakeAnnotation([])
    mocker.patch("transcript.diarize.Pipeline.from_pretrained", return_value=mock_pipe)
    mocker.patch("transcript.diarize.torch.device", side_effect=lambda x: x)
    mocker.patch("transcript.diarize.torch.backends.mps.is_available", return_value=True)

    diarize.run(wav, num_speakers=None)
    _, kwargs = mock_pipe.call_args
    assert kwargs == {}


def test_run_falls_back_to_cpu_when_mps_unavailable(mocker, tmp_path):
    wav = tmp_path / "x.wav"; wav.write_bytes(b"")
    mocker.patch("transcript.diarize.config.hf_token", return_value="hf_xxx")
    mock_pipe = mocker.MagicMock()
    mock_pipe.return_value = _FakeAnnotation([])
    mocker.patch("transcript.diarize.Pipeline.from_pretrained", return_value=mock_pipe)
    mocker.patch("transcript.diarize.torch.device", side_effect=lambda x: x)
    mocker.patch("transcript.diarize.torch.backends.mps.is_available", return_value=False)

    diarize.run(wav, num_speakers=None)
    mock_pipe.to.assert_called_once_with("cpu")


def test_run_401_raises_actionable_error(mocker, tmp_path):
    wav = tmp_path / "x.wav"; wav.write_bytes(b"")
    mocker.patch("transcript.diarize.config.hf_token", return_value="hf_xxx")

    class FakeHTTPError(Exception):
        def __init__(self):
            self.response = type("R", (), {"status_code": 401})()

    mocker.patch(
        "transcript.diarize.Pipeline.from_pretrained", side_effect=FakeHTTPError()
    )
    with pytest.raises(diarize.DiarizeError, match="license"):
        diarize.run(wav, num_speakers=None)
