import json

from transcript.formatters.json import render
from transcript.models import Meta, Utterance


def test_json_structure():
    utterances = [
        Utterance(speaker="Speaker 1", start=0.0, end=1.0, text="bonjour"),
        Utterance(speaker="Speaker 2", start=1.0, end=2.0, text="salut"),
    ]
    meta = Meta(filename="v.m4a", duration=2.0, model="large-v3", language="fr", speaker_count=2)
    out = render(utterances, meta)
    data = json.loads(out)

    assert data["meta"]["filename"] == "v.m4a"
    assert data["meta"]["duration"] == 2.0
    assert data["meta"]["model"] == "large-v3"
    assert data["meta"]["language"] == "fr"
    assert data["meta"]["speaker_count"] == 2

    assert len(data["utterances"]) == 2
    assert data["utterances"][0] == {
        "speaker": "Speaker 1",
        "start": 0.0,
        "end": 1.0,
        "text": "bonjour",
    }


def test_json_is_pretty_printed():
    out = render([], Meta(filename="x", duration=0.0, model="m", language="fr", speaker_count=0))
    assert "\n" in out  # 2-space indent uses newlines
