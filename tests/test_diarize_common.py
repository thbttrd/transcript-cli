import logging
from pathlib import Path

import pytest

from transcript import diarize_common
from transcript.models import Turn


def test_relabel_assigns_speaker_n_in_first_appearance_order():
    raw = [
        Turn("spk_zzz", 0.0, 1.0),
        Turn("spk_aaa", 1.0, 2.0),
        Turn("spk_zzz", 2.0, 3.0),
        Turn("spk_aaa", 3.0, 4.0),
        Turn("spk_qqq", 4.0, 5.0),
    ]
    turns = diarize_common.relabel_by_first_appearance(raw)
    assert turns == [
        Turn("Speaker 1", 0.0, 1.0),
        Turn("Speaker 2", 1.0, 2.0),
        Turn("Speaker 1", 2.0, 3.0),
        Turn("Speaker 2", 3.0, 4.0),
        Turn("Speaker 3", 4.0, 5.0),
    ]


def test_relabel_orders_speakers_by_start_time_for_unsorted_input():
    """Sort-then-relabel finds first-appearance even when input isn't chronological."""
    raw = [
        Turn("SPEAKER_05", 5.0, 6.0),
        Turn("SPEAKER_01", 0.0, 1.0),   # talks first
        Turn("SPEAKER_05", 7.0, 8.0),
        Turn("SPEAKER_03", 2.0, 3.0),   # talks second
        Turn("SPEAKER_01", 9.0, 10.0),  # same speaker as 0.0–1.0 → reuses Speaker 1
    ]
    turns = diarize_common.relabel_by_first_appearance(raw)
    by_start = {t.start: t.speaker for t in turns}
    assert by_start[0.0] == "Speaker 1"
    assert by_start[2.0] == "Speaker 2"
    assert by_start[5.0] == "Speaker 3"
    assert by_start[9.0] == "Speaker 1"
    # Original list order is preserved (only labels change).
    assert [t.start for t in turns] == [5.0, 0.0, 7.0, 2.0, 9.0]


def test_relabel_handles_empty_input():
    assert diarize_common.relabel_by_first_appearance([]) == []


def test_filter_and_warn_caps_to_first_num_speakers():
    turns = [
        Turn("Speaker 1", 0.0, 1.0),
        Turn("Speaker 2", 1.0, 2.0),
        Turn("Speaker 3", 2.0, 3.0),
    ]
    log = logging.getLogger("test")
    out = diarize_common.filter_and_warn(
        turns, num_speakers=2, backend_label="X", wav_path=Path("/x.wav"), log=log
    )
    assert {t.speaker for t in out} == {"Speaker 1", "Speaker 2"}


def test_filter_and_warn_logs_when_num_speakers_drops_turns(caplog):
    """The user passed --speakers N < distinct speakers. Dropped turns' words
    get silently reassigned downstream — surface the silent narrowing as a
    warning so the user can re-run without the cap if it wasn't intended."""
    turns = [
        Turn("Speaker 1", 0.0, 1.0),
        Turn("Speaker 2", 1.0, 2.0),
        Turn("Speaker 3", 2.0, 3.0),
    ]
    log = logging.getLogger("transcript.test_diarize_common.drop")
    with caplog.at_level(logging.WARNING, logger=log.name):
        diarize_common.filter_and_warn(
            turns, num_speakers=2, backend_label="MyBackend",
            wav_path=Path("/x.wav"), log=log,
        )
    messages = [r.getMessage() for r in caplog.records]
    assert any("MyBackend" in m and "dropped" in m and "--speakers=2" in m for m in messages)


def test_filter_and_warn_silent_when_num_speakers_exceeds_distinct():
    """Cap >= distinct speakers is a true no-op: no dropped-warning log."""
    turns = [Turn("Speaker 1", 0.0, 1.0), Turn("Speaker 2", 1.0, 2.0)]
    log = logging.getLogger("transcript.test_diarize_common.noop")
    log.warning = lambda *a, **kw: pytest.fail(f"unexpected warning: {a} {kw}")
    out = diarize_common.filter_and_warn(
        turns, num_speakers=5, backend_label="X", wav_path=Path("/x.wav"), log=log
    )
    assert out == turns


def test_filter_and_warn_is_noop_when_num_speakers_none():
    turns = [Turn("Speaker 1", 0.0, 1.0), Turn("Speaker 2", 1.0, 2.0)]
    log = logging.getLogger("test")
    out = diarize_common.filter_and_warn(
        turns, num_speakers=None, backend_label="X", wav_path=Path("/x.wav"), log=log
    )
    assert out == turns


def test_filter_and_warn_logs_when_result_is_empty(caplog):
    log = logging.getLogger("transcript.test_diarize_common")
    with caplog.at_level(logging.WARNING, logger=log.name):
        out = diarize_common.filter_and_warn(
            [], num_speakers=None, backend_label="MyBackend", wav_path=Path("/x.wav"), log=log
        )
    assert out == []
    msg = caplog.records[0].getMessage()
    assert "MyBackend" in msg and "no turns" in msg.lower()
