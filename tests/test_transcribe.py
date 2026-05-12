import json
import shutil
from pathlib import Path

import pytest

from transcript import transcribe
from transcript.models import Word
from transcript.pipeline_config import TranscribeConfig

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


def test_parse_words_keeps_multi_token_words_together():
    """Regression: whisper.cpp `-ml 1 --split-on-word` emits one *segment* per
    word, but a single word may comprise multiple BPE tokens (e.g. "Chouchou"
    → "Ch" + "ouch" + "ou"). We must consume segment text, not per-token text,
    or downstream diarization will scatter sub-word pieces across speakers.
    """
    data = {
        "transcription": [
            {
                "offsets": {"from": 300, "to": 930},
                "text": " Chouchou,",
                "tokens": [
                    {"text": " Ch", "offsets": {"from": 300, "to": 420}},
                    {"text": "ouch", "offsets": {"from": 420, "to": 620}},
                    {"text": "ou", "offsets": {"from": 690, "to": 780}},
                    {"text": ",", "offsets": {"from": 780, "to": 830}},
                ],
            },
        ],
    }
    words = transcribe._parse_words(data)
    assert words == [Word(text=" Chouchou,", start=0.30, end=0.93)]


def test_run_invokes_whisper_with_correct_flags(tmp_path, mocker):
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"")
    mocker.patch("transcript.transcribe.transcript_config.whisper_binary", return_value=Path("/fake/main"))
    mocker.patch(
        "transcript.transcribe.transcript_config.whisper_model", return_value=Path("/fake/ggml-large-v3.bin")
    )
    mocker.patch("transcript.transcribe.Path.exists", return_value=True)
    out_json = tmp_path / "out.json"

    def fake_run(cmd, *args, **kwargs):
        # whisper.cpp writes <output_prefix>.json — emulate that
        json_file = Path(cmd[cmd.index("-of") + 1] + ".json")
        json_file.write_text(FIXTURE.read_text())
        return mocker.Mock(returncode=0)

    mock_run = mocker.patch("transcript.transcribe.subprocess.run", side_effect=fake_run)
    words, detected_lang = transcribe.run(wav, config=TranscribeConfig(language="fr"))

    assert len(words) == 2
    assert detected_lang == "fr"
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "/fake/main"
    assert "-m" in cmd and "/fake/ggml-large-v3.bin" in cmd
    assert "-f" in cmd and str(wav) in cmd
    assert "-l" in cmd and "fr" in cmd
    assert "-ml" in cmd and "1" in cmd
    assert "--split-on-word" in cmd
    assert "--no-fallback" in cmd
    assert "--suppress-nst" in cmd
    assert "-ojf" in cmd or "--output-json-full" in cmd


def test_run_auto_language_when_none(tmp_path, mocker):
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"")
    mocker.patch("transcript.transcribe.transcript_config.whisper_binary", return_value=Path("/fake/main"))
    mocker.patch(
        "transcript.transcribe.transcript_config.whisper_model", return_value=Path("/fake/ggml-large-v3.bin")
    )
    mocker.patch("transcript.transcribe.Path.exists", return_value=True)

    def fake_run(cmd, *args, **kwargs):
        json_file = Path(cmd[cmd.index("-of") + 1] + ".json")
        json_file.write_text(FIXTURE.read_text())
        return mocker.Mock(returncode=0)

    mock_run = mocker.patch("transcript.transcribe.subprocess.run", side_effect=fake_run)
    transcribe.run(wav, config=TranscribeConfig(language=None))
    cmd = mock_run.call_args[0][0]
    assert "-l" in cmd and "auto" in cmd


def test_run_missing_binary_raises(tmp_path, mocker):
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"")
    mocker.patch(
        "transcript.transcribe.transcript_config.whisper_binary", return_value=Path("/nope/main")
    )
    with pytest.raises(transcribe.TranscribeError, match="not found"):
        transcribe.run(wav, config=TranscribeConfig(language="fr"))


def test_run_propagates_whisper_failure(tmp_path, mocker):
    import subprocess as sp
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"")
    mocker.patch("transcript.transcribe.transcript_config.whisper_binary", return_value=Path("/fake/main"))
    mocker.patch(
        "transcript.transcribe.transcript_config.whisper_model", return_value=Path("/fake/ggml-large-v3.bin")
    )
    mocker.patch("transcript.transcribe.Path.exists", return_value=True)
    err = sp.CalledProcessError(1, "main", stderr=b"failed badly")
    mocker.patch("transcript.transcribe.subprocess.run", side_effect=err)
    with pytest.raises(transcribe.TranscribeError, match="failed badly"):
        transcribe.run(wav, config=TranscribeConfig(language="fr"))


def test_run_respects_no_fallback_flag(tmp_path, mocker):
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"")
    mocker.patch("transcript.transcribe.transcript_config.whisper_binary", return_value=Path("/fake/main"))
    mocker.patch("transcript.transcribe.transcript_config.whisper_model", return_value=Path("/fake/m.bin"))
    mocker.patch("transcript.transcribe.Path.exists", return_value=True)

    def fake_run(cmd, *args, **kwargs):
        json_file = Path(cmd[cmd.index("-of") + 1] + ".json")
        json_file.write_text(FIXTURE.read_text())
        return mocker.Mock(returncode=0)

    mock_run = mocker.patch("transcript.transcribe.subprocess.run", side_effect=fake_run)
    cfg = TranscribeConfig(language="fr", no_fallback=False)
    transcribe.run(wav, config=cfg)
    cmd = mock_run.call_args[0][0]
    assert "--no-fallback" not in cmd


def test_run_respects_suppress_nst_flag(tmp_path, mocker):
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"")
    mocker.patch("transcript.transcribe.transcript_config.whisper_binary", return_value=Path("/fake/main"))
    mocker.patch("transcript.transcribe.transcript_config.whisper_model", return_value=Path("/fake/m.bin"))
    mocker.patch("transcript.transcribe.Path.exists", return_value=True)

    def fake_run(cmd, *args, **kwargs):
        json_file = Path(cmd[cmd.index("-of") + 1] + ".json")
        json_file.write_text(FIXTURE.read_text())
        return mocker.Mock(returncode=0)

    mock_run = mocker.patch("transcript.transcribe.subprocess.run", side_effect=fake_run)
    cfg = TranscribeConfig(language="fr", suppress_nst=False)
    transcribe.run(wav, config=cfg)
    cmd = mock_run.call_args[0][0]
    assert "--suppress-nst" not in cmd


def test_run_passes_temperature(tmp_path, mocker):
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"")
    mocker.patch("transcript.transcribe.transcript_config.whisper_binary", return_value=Path("/fake/main"))
    mocker.patch("transcript.transcribe.transcript_config.whisper_model", return_value=Path("/fake/m.bin"))
    mocker.patch("transcript.transcribe.Path.exists", return_value=True)

    def fake_run(cmd, *args, **kwargs):
        json_file = Path(cmd[cmd.index("-of") + 1] + ".json")
        json_file.write_text(FIXTURE.read_text())
        return mocker.Mock(returncode=0)

    mock_run = mocker.patch("transcript.transcribe.subprocess.run", side_effect=fake_run)
    cfg = TranscribeConfig(language="fr", temperature=0.3)
    transcribe.run(wav, config=cfg)
    cmd = mock_run.call_args[0][0]
    i = cmd.index("--temperature")
    assert cmd[i + 1] == "0.3"
