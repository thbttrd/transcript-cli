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
    from transcript.models import Meta, Utterance
    f = tmp_path / "v.m4a"
    f.write_bytes(b"")
    spy = mocker.patch(
        "transcript.cli.pipeline.run",
        return_value=(
            [Utterance("Speaker 1", 0.0, 1.0, "hello")],
            Meta(filename="v.m4a", duration=1.0, model="large-v3",
                 language="en", speaker_count=1, diarizer=None),
        ),
    )
    code = cli.main([str(f)])
    assert code == 0
    _, kwargs = spy.call_args
    assert kwargs["audio_path"] == f
    assert kwargs["with_diarization"] is True
    cfg = kwargs["config"]
    assert cfg.transcribe.model == "large-v3"
    assert cfg.transcribe.language is None
    assert cfg.diarize.num_speakers is None


def test_main_no_diarize_flag(tmp_path, mocker):
    from transcript.models import Meta, Utterance
    f = tmp_path / "v.m4a"; f.write_bytes(b"")
    spy = mocker.patch(
        "transcript.cli.pipeline.run",
        return_value=([], Meta("v.m4a", 1.0, "large-v3", "en", 0, None)),
    )
    cli.main([str(f), "--no-diarize"])
    _, kwargs = spy.call_args
    assert kwargs["with_diarization"] is False


def test_main_speakers_flag(tmp_path, mocker):
    from transcript.models import Meta, Utterance
    f = tmp_path / "v.m4a"; f.write_bytes(b"")
    spy = mocker.patch(
        "transcript.cli.pipeline.run",
        return_value=([], Meta("v.m4a", 1.0, "large-v3", "en", 0, None)),
    )
    cli.main([str(f), "--speakers", "2"])
    _, kwargs = spy.call_args
    assert kwargs["config"].diarize.num_speakers == 2


def test_main_writes_to_output_file_when_o_given(tmp_path, mocker):
    from transcript.models import Meta, Utterance
    f = tmp_path / "v.m4a"; f.write_bytes(b"")
    out = tmp_path / "out.md"
    mocker.patch(
        "transcript.cli.pipeline.run",
        return_value=(
            [Utterance("Speaker 1", 0.0, 1.0, "hello")],
            Meta("v.m4a", 1.0, "large-v3", "en", 1, None),
        ),
    )
    code = cli.main([str(f), "-o", str(out)])
    assert code == 0
    content = out.read_text()
    assert "hello" in content


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
    assert code == 11
    assert "license missing" in captured.err


def test_cli_builds_pipeline_config_from_args(tmp_path, mocker):
    """argparse to PipelineConfig wiring should pass model/language/speakers/align/llm_fix."""
    from transcript import cli
    from transcript.models import Meta, Utterance

    audio = tmp_path / "in.m4a"
    audio.write_bytes(b"")

    captured = {}

    def fake_run(*, audio_path, config, with_diarization, progress):
        captured["config"] = config
        captured["with_diarization"] = with_diarization
        return [Utterance("Speaker 1", 0.0, 1.0, "hi")], Meta(
            filename="in.m4a", duration=1.0, model="large-v3",
            language="fr", speaker_count=1, diarizer=None,
        )

    mocker.patch("transcript.cli.pipeline.run", side_effect=fake_run)
    rc = cli.main([str(audio), "-l", "fr", "--speakers", "2", "--no-align", "--llm-fix"])
    assert rc == 0
    cfg = captured["config"]
    assert cfg.transcribe.language == "fr"
    assert cfg.diarize.num_speakers == 2
    assert cfg.align.enabled is False
    assert cfg.llm_fix.enabled is True


def _stub_pipeline_run(mocker):
    """Return a spy on cli.pipeline.run that returns one trivial utterance.

    Most CLI flag tests just need to assert what cfg was constructed, not what
    came out of the pipeline.
    """
    from transcript.models import Meta, Utterance
    return mocker.patch(
        "transcript.cli.pipeline.run",
        return_value=(
            [Utterance("Speaker 1", 0.0, 1.0, "hi")],
            Meta("in.m4a", 1.0, "large-v3", "fr", 1, None),
        ),
    )


def test_main_diarizer_defaults_to_sortformer(tmp_path, mocker):
    f = tmp_path / "v.m4a"
    f.write_bytes(b"")
    spy = _stub_pipeline_run(mocker)
    cli.main([str(f)])
    assert spy.call_args.kwargs["config"].diarize.backend == "sortformer"


def test_main_diarizer_flag_propagates_to_diarize_config(tmp_path, mocker):
    f = tmp_path / "v.m4a"
    f.write_bytes(b"")
    spy = _stub_pipeline_run(mocker)
    cli.main([str(f), "--diarizer", "diarizen"])
    assert spy.call_args.kwargs["config"].diarize.backend == "diarizen"


def test_main_whisper_fallback_flag_disables_no_fallback(tmp_path, mocker):
    """--whisper-fallback lets Whisper retry → TranscribeConfig.no_fallback=False."""
    f = tmp_path / "v.m4a"
    f.write_bytes(b"")
    spy = _stub_pipeline_run(mocker)
    cli.main([str(f), "--whisper-fallback"])
    assert spy.call_args.kwargs["config"].transcribe.no_fallback is False


def test_main_no_whisper_fallback_flag_enables_no_fallback(tmp_path, mocker):
    """--no-whisper-fallback forbids retry → TranscribeConfig.no_fallback=True."""
    f = tmp_path / "v.m4a"
    f.write_bytes(b"")
    spy = _stub_pipeline_run(mocker)
    cli.main([str(f), "--no-whisper-fallback"])
    assert spy.call_args.kwargs["config"].transcribe.no_fallback is True


def test_main_whisper_fallback_unspecified_tracks_dataclass_default(tmp_path, mocker):
    """Neither --whisper-fallback nor --no-whisper-fallback → CLI doesn't override default."""
    from transcript.pipeline_config import TranscribeConfig

    f = tmp_path / "v.m4a"
    f.write_bytes(b"")
    spy = _stub_pipeline_run(mocker)
    cli.main([str(f)])
    cfg_value = spy.call_args.kwargs["config"].transcribe.no_fallback
    default_value = TranscribeConfig.__dataclass_fields__["no_fallback"].default
    assert cfg_value == default_value
