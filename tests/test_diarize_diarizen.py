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
from transcript.models import Turn
from transcript.pipeline_config import DiarizeConfig


def test_relabel_by_first_appearance_orders_speakers_by_start_time():
    """Speaker 1 = first to talk, regardless of input list order or original label."""
    raw = [
        Turn("SPEAKER_05", 5.0, 6.0),
        Turn("SPEAKER_01", 0.0, 1.0),   # talks first
        Turn("SPEAKER_05", 7.0, 8.0),
        Turn("SPEAKER_03", 2.0, 3.0),   # talks second
        Turn("SPEAKER_01", 9.0, 10.0),  # same speaker as 0.0–1.0 → reuses Speaker 1
    ]
    turns = diarize_diarizen._relabel_by_first_appearance(raw)
    by_start = {t.start: t.speaker for t in turns}
    assert by_start[0.0] == "Speaker 1"   # SPEAKER_01 spoke first
    assert by_start[2.0] == "Speaker 2"   # SPEAKER_03 spoke second
    assert by_start[5.0] == "Speaker 3"   # SPEAKER_05 spoke third
    assert by_start[9.0] == "Speaker 1"   # SPEAKER_01 reappears
    # Original list order is preserved (only labels change).
    assert [t.start for t in turns] == [5.0, 0.0, 7.0, 2.0, 9.0]


def test_relabel_by_first_appearance_handles_empty_input():
    assert diarize_diarizen._relabel_by_first_appearance([]) == []


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


def test_run_raises_when_runner_script_missing(monkeypatch):
    monkeypatch.setattr(diarize_diarizen, "_SCRIPT", Path("/nope/diarize_diarizen.py"))
    with pytest.raises(diarize_diarizen.DiariZenError, match="not found"):
        diarize_diarizen.run(Path("/fake.wav"), config=DiarizeConfig())


def test_run_rejects_non_diarizeconfig(mocker):
    mocker.patch("transcript.diarize_diarizen.subprocess.run")
    with pytest.raises(TypeError, match="DiarizeConfig"):
        diarize_diarizen.run(Path("/fake.wav"), config="not a config")  # type: ignore[arg-type]
