"""Unit tests for the DiariZen subprocess wrapper.

The wrapper itself does no ML work — DiariZen runs in an isolated PEP 723
script env (scripts/diarize_diarizen.py). These tests mock subprocess.run
and exercise the wrapper's contract: invoke uv correctly, parse JSON off
stdout, normalise speaker labels to first-appearance Speaker N order,
filter by num_speakers, surface errors as DiariZenError.
"""
import json
import logging
import subprocess
from pathlib import Path

import pytest

from transcript import diarize_diarizen
from transcript.diarize_common import DiarizeError
from transcript.models import Turn
from transcript.pipeline_config import DiarizeConfig


def _completed(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_run_parses_subprocess_json_into_turns(mocker):
    fake_json = json.dumps([
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.5},
        {"speaker": "SPEAKER_01", "start": 1.5, "end": 3.0},
    ])
    mocker.patch(
        "transcript.diarize_diarizen.subprocess.run",
        return_value=_completed(stdout=fake_json),
    )
    turns = diarize_diarizen.run(Path("/fake.wav"), config=DiarizeConfig())
    assert turns == [
        Turn("Speaker 1", 0.0, 1.5),
        Turn("Speaker 2", 1.5, 3.0),
    ]


def test_run_invokes_uv_run_script_with_wav_path(mocker):
    spy = mocker.patch(
        "transcript.diarize_diarizen.subprocess.run",
        return_value=_completed(stdout="[]"),
    )
    diarize_diarizen.run(Path("/some/audio.wav"), config=DiarizeConfig())
    argv = spy.call_args[0][0]
    assert argv[0] == "uv"
    assert argv[1:3] == ["run", "--script"]
    assert argv[3].endswith("scripts/diarize_diarizen.py")
    assert argv[4] == "/some/audio.wav"
    # Lock in the decode-safety + error-propagation kwargs.
    kwargs = spy.call_args.kwargs
    assert kwargs["capture_output"] is True
    assert kwargs["check"] is True
    assert kwargs["encoding"] == "utf-8"
    assert kwargs["errors"] == "replace"


def test_run_filters_by_num_speakers(mocker):
    fake_json = json.dumps([
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0},
        {"speaker": "SPEAKER_01", "start": 1.0, "end": 2.0},
        {"speaker": "SPEAKER_02", "start": 2.0, "end": 3.0},
    ])
    mocker.patch(
        "transcript.diarize_diarizen.subprocess.run",
        return_value=_completed(stdout=fake_json),
    )
    turns = diarize_diarizen.run(Path("/fake.wav"), config=DiarizeConfig(num_speakers=2))
    assert {t.speaker for t in turns} == {"Speaker 1", "Speaker 2"}


def test_run_warns_when_subprocess_returns_no_turns(mocker, caplog):
    mocker.patch(
        "transcript.diarize_diarizen.subprocess.run",
        return_value=_completed(stdout="[]"),
    )
    with caplog.at_level(logging.WARNING, logger="transcript.diarize_diarizen"):
        turns = diarize_diarizen.run(Path("/fake.wav"), config=DiarizeConfig())
    assert turns == []
    assert any("no turns" in r.getMessage().lower() for r in caplog.records)


def test_run_raises_on_subprocess_failure(mocker):
    err = subprocess.CalledProcessError(
        returncode=1, cmd=["uv", "run", "..."], output="", stderr="boom"
    )
    mocker.patch("transcript.diarize_diarizen.subprocess.run", side_effect=err)
    with pytest.raises(diarize_diarizen.DiariZenError, match="exit 1"):
        diarize_diarizen.run(Path("/fake.wav"), config=DiarizeConfig())


def test_run_raises_on_invalid_json_stdout(mocker):
    mocker.patch(
        "transcript.diarize_diarizen.subprocess.run",
        return_value=_completed(stdout="not json"),
    )
    with pytest.raises(diarize_diarizen.DiariZenError, match="non-JSON"):
        diarize_diarizen.run(Path("/fake.wav"), config=DiarizeConfig())


def test_run_raises_when_uv_not_on_path(mocker):
    mocker.patch(
        "transcript.diarize_diarizen.subprocess.run",
        side_effect=FileNotFoundError("uv"),
    )
    with pytest.raises(diarize_diarizen.DiariZenError, match="`uv` not found"):
        diarize_diarizen.run(Path("/fake.wav"), config=DiarizeConfig())


def test_run_raises_when_runner_script_missing(monkeypatch, mocker):
    # Defensive: also mock subprocess.run so a regression that removed the
    # early-exists check wouldn't silently shell out to uv during the test.
    spy = mocker.patch("transcript.diarize_diarizen.subprocess.run")
    monkeypatch.setattr(diarize_diarizen, "_SCRIPT", Path("/nope/diarize_diarizen.py"))
    with pytest.raises(diarize_diarizen.DiariZenError, match="not found"):
        diarize_diarizen.run(Path("/fake.wav"), config=DiarizeConfig())
    spy.assert_not_called()


def test_run_rejects_non_diarizeconfig(mocker):
    mocker.patch("transcript.diarize_diarizen.subprocess.run")
    with pytest.raises(TypeError, match="DiarizeConfig"):
        diarize_diarizen.run(Path("/fake.wav"), config="not a config")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_payload",
    [
        '[{"start": 0.0, "end": 1.0}]',                       # missing 'speaker'
        '[{"speaker": "A", "end": 1.0}]',                     # missing 'start'
        '[{"speaker": "A", "start": "oops", "end": 1.0}]',    # non-numeric start
        '{"turns": []}',                                      # wrong top-level shape
        "null",                                               # not iterable
    ],
)
def test_run_raises_on_malformed_turn_record(mocker, bad_payload):
    mocker.patch(
        "transcript.diarize_diarizen.subprocess.run",
        return_value=_completed(stdout=bad_payload),
    )
    with pytest.raises(diarize_diarizen.DiariZenError, match="malformed turn record"):
        diarize_diarizen.run(Path("/fake.wav"), config=DiarizeConfig())


def test_diarizen_error_is_a_diarize_error():
    """CLI catches DiarizeError; DiariZenError must be a subclass so DiariZen
    failures exit through the same code path with the same prefix."""
    assert issubclass(diarize_diarizen.DiariZenError, DiarizeError)
