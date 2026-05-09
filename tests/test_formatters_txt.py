from transcript.formatters.txt import render
from transcript.models import Meta, Utterance


META = Meta(filename="v.m4a", duration=5.0, model="large-v3", language="fr", speaker_count=2)


def test_txt_basic():
    utterances = [
        Utterance(speaker="Speaker 1", start=0.0, end=1.0, text="bonjour"),
        Utterance(speaker="Speaker 2", start=1.0, end=2.0, text="salut"),
    ]
    out = render(utterances, META)
    assert out == "Speaker 1: bonjour\nSpeaker 2: salut\n"


def test_txt_empty():
    assert render([], META) == ""
