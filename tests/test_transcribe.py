import json
import shutil
from pathlib import Path

import pytest

from transcript import transcribe
from transcript.models import Word

FIXTURE = Path(__file__).parent / "fixtures" / "whisper_output_sample.json"


def test_parse_words_extracts_tokens_with_seconds():
    data = json.loads(FIXTURE.read_text())
    words = transcribe._parse_words(data)
    assert words == [
        Word(text=" Bonjour", start=0.0, end=1.0),
        Word(text=" monde", start=1.0, end=2.0),
    ]


def test_parse_words_skips_special_tokens():
    data = json.loads(FIXTURE.read_text())
    words = transcribe._parse_words(data)
    assert all("[_" not in w.text for w in words)


def test_run_invokes_whisper_with_correct_flags(tmp_path, mocker):
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"")
    mocker.patch("transcript.transcribe.config.whisper_binary", return_value=Path("/fake/main"))
    mocker.patch(
        "transcript.transcribe.config.whisper_model", return_value=Path("/fake/ggml-large-v3.bin")
    )
    out_json = tmp_path / "out.json"

    def fake_run(cmd, *args, **kwargs):
        # whisper.cpp writes <output_prefix>.json — emulate that
        json_file = Path(cmd[cmd.index("-of") + 1] + ".json")
        json_file.write_text(FIXTURE.read_text())
        return mocker.Mock(returncode=0)

    mock_run = mocker.patch("transcript.transcribe.subprocess.run", side_effect=fake_run)
    words = transcribe.run(wav, model="large-v3", language="fr")

    assert len(words) == 2
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "/fake/main"
    assert "-m" in cmd and "/fake/ggml-large-v3.bin" in cmd
    assert "-f" in cmd and str(wav) in cmd
    assert "-l" in cmd and "fr" in cmd
    assert "-ml" in cmd and "1" in cmd
    assert "--split-on-word" in cmd
    assert "-ojf" in cmd or "--output-json-full" in cmd


def test_run_auto_language_when_none(tmp_path, mocker):
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"")
    mocker.patch("transcript.transcribe.config.whisper_binary", return_value=Path("/fake/main"))
    mocker.patch(
        "transcript.transcribe.config.whisper_model", return_value=Path("/fake/ggml-large-v3.bin")
    )

    def fake_run(cmd, *args, **kwargs):
        json_file = Path(cmd[cmd.index("-of") + 1] + ".json")
        json_file.write_text(FIXTURE.read_text())
        return mocker.Mock(returncode=0)

    mock_run = mocker.patch("transcript.transcribe.subprocess.run", side_effect=fake_run)
    transcribe.run(wav, model="large-v3", language=None)
    cmd = mock_run.call_args[0][0]
    assert "-l" in cmd and "auto" in cmd


def test_run_missing_binary_raises(tmp_path, mocker):
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"")
    mocker.patch(
        "transcript.transcribe.config.whisper_binary", return_value=Path("/nope/main")
    )
    with pytest.raises(transcribe.TranscribeError, match="not found"):
        transcribe.run(wav, model="large-v3", language="fr")


def test_run_propagates_whisper_failure(tmp_path, mocker):
    import subprocess as sp
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"")
    mocker.patch("transcript.transcribe.config.whisper_binary", return_value=Path("/fake/main"))
    mocker.patch(
        "transcript.transcribe.config.whisper_model", return_value=Path("/fake/ggml-large-v3.bin")
    )
    err = sp.CalledProcessError(1, "main", stderr=b"failed badly")
    mocker.patch("transcript.transcribe.subprocess.run", side_effect=err)
    with pytest.raises(transcribe.TranscribeError, match="failed badly"):
        transcribe.run(wav, model="large-v3", language="fr")
