from pathlib import Path

import pytest

from transcript import cli


def test_main_no_args_prints_usage_and_exits_2(capsys):
    code = cli.main([])
    captured = capsys.readouterr()
    assert code == 2
    assert "usage" in captured.err.lower() or "usage" in captured.out.lower()


def test_main_version_prints_and_exits(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "0.1.0" in captured.out


def test_main_doctor_invokes_doctor_check(mocker, capsys):
    mocker.patch("transcript.cli.doctor.check", return_value=(0, "all good"))
    code = cli.main(["--doctor"])
    captured = capsys.readouterr()
    assert code == 0
    assert "all good" in captured.out


def test_main_dispatches_to_pipeline_with_defaults(tmp_path, mocker):
    f = tmp_path / "v.m4a"
    f.write_bytes(b"")
    spy = mocker.patch("transcript.cli.pipeline.run", return_value="# ok\n")
    code = cli.main([str(f)])
    assert code == 0
    _, kwargs = spy.call_args
    assert kwargs["audio_path"] == f
    assert kwargs["model"] == "large-v3"
    assert kwargs["language"] is None
    assert kwargs["diarize"] is True
    assert kwargs["num_speakers"] is None
    assert kwargs["format_name"] == "md"
    assert kwargs["with_timestamps"] is True


def test_main_no_diarize_flag(tmp_path, mocker):
    f = tmp_path / "v.m4a"; f.write_bytes(b"")
    spy = mocker.patch("transcript.cli.pipeline.run", return_value="ok")
    cli.main([str(f), "--no-diarize"])
    _, kwargs = spy.call_args
    assert kwargs["diarize"] is False


def test_main_speakers_flag(tmp_path, mocker):
    f = tmp_path / "v.m4a"; f.write_bytes(b"")
    spy = mocker.patch("transcript.cli.pipeline.run", return_value="ok")
    cli.main([str(f), "--speakers", "2"])
    _, kwargs = spy.call_args
    assert kwargs["num_speakers"] == 2


def test_main_writes_to_output_file_when_o_given(tmp_path, mocker):
    f = tmp_path / "v.m4a"; f.write_bytes(b"")
    out = tmp_path / "out.md"
    mocker.patch("transcript.cli.pipeline.run", return_value="# transcript\n")
    code = cli.main([str(f), "-o", str(out)])
    assert code == 0
    assert out.read_text() == "# transcript\n"


def test_main_audio_error_exit_10(tmp_path, mocker, capsys):
    from transcript.audio import AudioError

    f = tmp_path / "v.m4a"; f.write_bytes(b"")
    mocker.patch("transcript.cli.pipeline.run", side_effect=AudioError("file not found"))
    code = cli.main([str(f)])
    captured = capsys.readouterr()
    assert code == 10
    assert "file not found" in captured.err


def test_main_transcribe_error_exit_11(tmp_path, mocker, capsys):
    from transcript.transcribe import TranscribeError

    f = tmp_path / "v.m4a"; f.write_bytes(b"")
    mocker.patch("transcript.cli.pipeline.run", side_effect=TranscribeError("not built"))
    code = cli.main([str(f)])
    captured = capsys.readouterr()
    assert code == 11
    assert "not built" in captured.err


def test_main_diarize_error_exit_12(tmp_path, mocker, capsys):
    from transcript.diarize import DiarizeError

    f = tmp_path / "v.m4a"; f.write_bytes(b"")
    mocker.patch("transcript.cli.pipeline.run", side_effect=DiarizeError("license missing"))
    code = cli.main([str(f)])
    captured = capsys.readouterr()
    assert code == 12
    assert "license missing" in captured.err


def test_main_missing_token_error_exit_12(tmp_path, mocker, capsys):
    from transcript.config import MissingTokenError

    f = tmp_path / "v.m4a"; f.write_bytes(b"")
    mocker.patch("transcript.cli.pipeline.run", side_effect=MissingTokenError("no token"))
    code = cli.main([str(f)])
    captured = capsys.readouterr()
    assert code == 12
    assert "no token" in captured.err
