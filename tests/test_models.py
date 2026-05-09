import pytest
from transcript.models import Meta, Turn, Utterance, Word


def test_word_is_frozen():
    w = Word(text="hello", start=0.0, end=0.5)
    with pytest.raises(AttributeError):
        w.text = "world"  # type: ignore[misc]


def test_turn_fields():
    t = Turn(speaker="Speaker 1", start=1.0, end=3.5)
    assert t.speaker == "Speaker 1"
    assert t.start == 1.0
    assert t.end == 3.5


def test_utterance_fields():
    u = Utterance(speaker="Speaker 2", start=0.0, end=1.0, text="bonjour")
    assert u.text == "bonjour"


def test_meta_fields():
    m = Meta(
        filename="voice.m4a",
        duration=754.0,
        model="large-v3",
        language="fr",
        speaker_count=2,
    )
    assert m.duration == 754.0
    assert m.speaker_count == 2
