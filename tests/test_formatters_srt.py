from transcript.formatters.srt import render
from transcript.models import Meta, Utterance


META = Meta(filename="v.m4a", duration=10.0, model="large-v3", language="fr", speaker_count=2)


def test_srt_basic_format():
    utterances = [
        Utterance(speaker="Speaker 1", start=0.0, end=2.5, text="bonjour"),
        Utterance(speaker="Speaker 2", start=2.5, end=5.0, text="salut"),
    ]
    out = render(utterances, META)
    expected = (
        "1\n"
        "00:00:00,000 --> 00:00:02,500\n"
        "Speaker 1: bonjour\n"
        "\n"
        "2\n"
        "00:00:02,500 --> 00:00:05,000\n"
        "Speaker 2: salut\n"
    )
    assert out == expected


def test_srt_long_timestamp():
    u = Utterance(speaker="X", start=3661.123, end=3662.456, text="a")
    out = render([u], META)
    assert "01:01:01,123 --> 01:01:02,456" in out


def test_srt_empty():
    assert render([], META) == ""
