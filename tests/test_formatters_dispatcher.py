import pytest

from transcript.formatters import get


def test_get_returns_callable_per_format():
    for name in ("md", "json", "srt", "txt"):
        assert callable(get(name))


def test_get_unknown_format_raises():
    with pytest.raises(ValueError, match="unknown format"):
        get("xml")


def test_get_md_supports_with_timestamps_kwarg():
    from transcript.models import Meta, Utterance

    fn = get("md")
    meta = Meta(filename="v", duration=1.0, model="m", language="fr", speaker_count=1)
    out = fn([Utterance(speaker="A", start=0.0, end=1.0, text="hi")], meta, with_timestamps=False)
    assert "[00:00]" not in out
