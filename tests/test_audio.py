from pathlib import Path

import pytest

from transcript import audio


def test_prepare_passes_through_already_correct_wav(tmp_path, mocker):
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"")
    mock_probe = mocker.patch(
        "transcript.audio._probe", return_value={"sample_rate": 16000, "channels": 1, "duration": 5.0}
    )
    mock_run = mocker.patch("transcript.audio.subprocess.run")
    out_path, duration = audio.prepare(wav)
    assert out_path == wav
    assert duration == 5.0
    mock_run.assert_not_called()
    mock_probe.assert_called_once_with(wav)


def test_prepare_converts_m4a_via_ffmpeg(tmp_path, mocker):
    src = tmp_path / "voice.m4a"
    src.write_bytes(b"")
    mocker.patch(
        "transcript.audio._probe", return_value={"sample_rate": 44100, "channels": 2, "duration": 7.5}
    )
    mock_run = mocker.patch("transcript.audio.subprocess.run", return_value=mocker.Mock(returncode=0))
    out_path, duration = audio.prepare(src)
    assert out_path != src
    assert out_path.suffix == ".wav"
    assert duration == 7.5
    args, _ = mock_run.call_args
    cmd = args[0]
    assert cmd[0] == "ffmpeg"
    assert "-ar" in cmd and "16000" in cmd
    assert "-ac" in cmd and "1" in cmd
    assert str(src) in cmd
    assert str(out_path) in cmd


def test_prepare_missing_file_raises(tmp_path):
    with pytest.raises(audio.AudioError, match="not found"):
        audio.prepare(tmp_path / "missing.m4a")


def test_prepare_short_audio_raises(tmp_path, mocker):
    src = tmp_path / "short.wav"
    src.write_bytes(b"")
    mocker.patch(
        "transcript.audio._probe", return_value={"sample_rate": 16000, "channels": 1, "duration": 0.2}
    )
    with pytest.raises(audio.AudioError, match="too short"):
        audio.prepare(src)


def test_ffmpeg_missing_raises(tmp_path, mocker):
    src = tmp_path / "voice.m4a"
    src.write_bytes(b"")
    mocker.patch(
        "transcript.audio._probe", return_value={"sample_rate": 44100, "channels": 2, "duration": 5.0}
    )
    mocker.patch("transcript.audio.subprocess.run", side_effect=FileNotFoundError("ffmpeg"))
    with pytest.raises(audio.AudioError, match="ffmpeg"):
        audio.prepare(src)
