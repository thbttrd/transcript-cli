from transcript.formatters.md import render
from transcript.models import Meta, Utterance


def _meta(speakers: int = 2) -> Meta:
    return Meta(
        filename="voice.m4a",
        duration=754.0,
        model="large-v3",
        language="fr",
        speaker_count=speakers,
    )


def test_md_with_timestamps():
    utterances = [
        Utterance(speaker="Speaker 1", start=0.0, end=14.2, text="bonjour"),
        Utterance(speaker="Speaker 2", start=14.2, end=38.5, text="oui"),
    ]
    out = render(utterances, _meta(), with_timestamps=True)
    assert "# voice.m4a" in out
    assert "## Speaker 1 [00:00]" in out
    assert "## Speaker 2 [00:14]" in out
    assert "bonjour" in out
    assert "oui" in out
    assert "12m34s" in out  # 754 seconds duration
    assert "2 speakers" in out
    assert "large-v3" in out
    assert "fr" in out


def test_md_without_timestamps():
    utterances = [Utterance(speaker="Speaker 1", start=0.0, end=1.0, text="bonjour")]
    out = render(utterances, _meta(speakers=1), with_timestamps=False)
    assert "## Speaker 1\n" in out
    assert "[00:00]" not in out
    assert "1 speaker" in out  # singular


def test_md_empty_utterances_still_has_header():
    out = render([], _meta(), with_timestamps=True)
    assert "# voice.m4a" in out
    assert "## Speaker" not in out


def test_md_timestamps_format_for_long_duration():
    # 1h 23m 45s = 5025s
    m = Meta(filename="long.m4a", duration=5025.0, model="large-v3", language="fr", speaker_count=1)
    out = render([], m, with_timestamps=True)
    assert "1h23m45s" in out
