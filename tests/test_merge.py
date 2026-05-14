from transcript import merge
from transcript.merge import assign
from transcript.models import Turn, Utterance, Word


def w(text: str, s: float, e: float) -> Word:
    return Word(text=text, start=s, end=e)


def t(speaker: str, s: float, e: float) -> Turn:
    return Turn(speaker=speaker, start=s, end=e)


def test_empty_inputs_returns_empty():
    assert assign([], []) == []


def test_words_with_no_turns_get_unknown_speaker():
    words = [w("hello", 0.0, 0.5)]
    result = assign(words, [])
    assert result == [Utterance(speaker="Unknown", start=0.0, end=0.5, text="hello")]


def test_single_speaker_collapses_into_one_utterance():
    words = [w(" bon", 0.0, 0.3), w("jour", 0.3, 0.6), w(" amis", 0.6, 1.0)]
    turns = [t("Speaker 1", 0.0, 1.0)]
    result = assign(words, turns)
    assert result == [
        Utterance(speaker="Speaker 1", start=0.0, end=1.0, text="bonjour amis")
    ]


def test_speaker_change_creates_two_utterances():
    words = [
        w(" hello", 0.0, 0.5),
        w(" world", 0.5, 1.0),
        w(" hi", 2.0, 2.3),
        w(" there", 2.3, 2.7),
    ]
    turns = [
        t("Speaker 1", 0.0, 1.5),
        t("Speaker 2", 1.5, 3.0),
    ]
    result = assign(words, turns)
    assert result == [
        Utterance(speaker="Speaker 1", start=0.0, end=1.0, text="hello world"),
        Utterance(speaker="Speaker 2", start=2.0, end=2.7, text="hi there"),
    ]


def test_three_speakers_alternating():
    words = [
        w(" a", 0.0, 0.2),
        w(" b", 1.0, 1.2),
        w(" c", 2.0, 2.2),
        w(" d", 3.0, 3.2),
    ]
    turns = [
        t("Speaker 1", 0.0, 0.5),
        t("Speaker 2", 0.9, 1.5),
        t("Speaker 3", 1.9, 2.5),
        t("Speaker 1", 2.9, 3.5),
    ]
    result = assign(words, turns)
    assert [u.speaker for u in result] == ["Speaker 1", "Speaker 2", "Speaker 3", "Speaker 1"]
    assert [u.text for u in result] == ["a", "b", "c", "d"]


def test_word_in_gap_snaps_to_upcoming_turn():
    # Word at 1.0–1.5 sits past S1's end with no S2 overlap. The word is the
    # first word of the next speaker, not the last word of the previous.
    words = [w(" lonely", 1.0, 1.5)]
    turns = [t("Speaker 1", 0.0, 1.0), t("Speaker 2", 2.0, 3.0)]
    result = assign(words, turns)
    assert result == [Utterance(speaker="Speaker 2", start=1.0, end=1.5, text="lonely")]


def test_word_overlapping_two_turns_picks_max_overlap():
    # Word 0.8–1.4 overlaps S1 by 0.2 and S2 by 0.4 → S2 wins.
    words = [w(" overlap", 0.8, 1.4)]
    turns = [t("Speaker 1", 0.0, 1.0), t("Speaker 2", 1.0, 2.0)]
    result = assign(words, turns)
    assert result == [Utterance(speaker="Speaker 2", start=0.8, end=1.4, text="overlap")]


def test_text_is_stripped_and_concatenated_in_order():
    words = [w("  he", 0.0, 0.1), w("llo", 0.1, 0.2), w("  world  ", 0.2, 0.3)]
    turns = [t("Speaker 1", 0.0, 0.5)]
    result = assign(words, turns)
    assert result[0].text == "hello  world"


# --- smooth_speaker_islands ----------------------------------------------------


def _wp(text: str, start: float, end: float, speaker: str) -> tuple[Word, str]:
    """Compact (Word, speaker) tuple for smoothing tests."""
    return (w(text, start, end), speaker)


def test_smooth_islands_flips_single_word_island_between_same_speakers():
    pairs = [
        _wp("a", 0.0, 0.5, "A"),
        _wp("b", 0.5, 1.0, "A"),
        _wp("c", 1.0, 1.2, "B"),   # 1-word B island
        _wp("d", 1.2, 1.7, "A"),
        _wp("e", 1.7, 2.0, "A"),
    ]
    result = merge.smooth_speaker_islands(pairs)
    assert [s for _, s in result] == ["A", "A", "A", "A", "A"]


def test_smooth_islands_flips_two_word_island_when_max_allows():
    pairs = [
        _wp("a", 0.0, 0.5, "A"),
        _wp("b", 0.5, 1.0, "B"),
        _wp("c", 1.0, 1.5, "B"),   # 2-word B island
        _wp("d", 1.5, 2.0, "A"),
    ]
    assert [s for _, s in merge.smooth_speaker_islands(pairs, max_island_words=2)] == [
        "A", "A", "A", "A",
    ]


def test_smooth_islands_preserves_three_word_island_when_max_is_two():
    pairs = [
        _wp("a", 0.0, 0.5, "A"),
        _wp("b", 0.5, 1.0, "B"),
        _wp("c", 1.0, 1.5, "B"),
        _wp("d", 1.5, 2.0, "B"),   # 3-word island — exceeds max
        _wp("e", 2.0, 2.5, "A"),
    ]
    assert [s for _, s in merge.smooth_speaker_islands(pairs, max_island_words=2)] == [
        "A", "B", "B", "B", "A",
    ]


def test_smooth_islands_preserves_when_surrounding_speakers_differ():
    """A → B → C → A: B and C are 1-word runs but with different neighbours, so neither flips."""
    pairs = [
        _wp("a", 0.0, 0.5, "A"),
        _wp("b", 0.5, 1.0, "B"),
        _wp("c", 1.0, 1.5, "C"),
        _wp("d", 1.5, 2.0, "A"),
    ]
    assert [s for _, s in merge.smooth_speaker_islands(pairs)] == ["A", "B", "C", "A"]


def test_smooth_islands_noop_on_empty_or_short_input():
    assert merge.smooth_speaker_islands([]) == []
    one = [_wp("a", 0.0, 0.5, "A")]
    assert merge.smooth_speaker_islands(one) == one
    two = [_wp("a", 0.0, 0.5, "A"), _wp("b", 0.5, 1.0, "B")]
    assert merge.smooth_speaker_islands(two) == two


def test_smooth_islands_handles_alternating_micro_runs_in_single_pass():
    """Two separate single-word B islands inside long A runs → both flip to A.

    The intervening A run is 3 words (above max_island_words=2), so it survives
    the pass intact — exactly what we want for "long A monologue with two stray
    B words" patterns. If the intervening A were itself shorter than max, the
    algorithm would (correctly) refuse to commit, since deciding the speaker
    of a 1-word run surrounded by 1-word other-speaker neighbours is genuinely
    ambiguous.
    """
    pairs = [
        _wp("alpha", 0.0, 0.5, "A"),
        _wp("beta", 0.5, 0.8, "A"),
        _wp("gamma", 0.8, 1.0, "A"),
        _wp("y", 1.0, 1.1, "B"),       # 1-word island
        _wp("delta", 1.1, 1.5, "A"),
        _wp("epsilon", 1.5, 1.8, "A"),
        _wp("zeta", 1.8, 2.0, "A"),
        _wp("y", 2.0, 2.1, "B"),       # 1-word island
        _wp("eta", 2.1, 2.5, "A"),
        _wp("theta", 2.5, 3.0, "A"),
    ]
    assert [s for _, s in merge.smooth_speaker_islands(pairs)] == ["A"] * 10


def test_smooth_islands_does_not_flip_when_max_island_words_is_zero():
    pairs = [
        _wp("a", 0.0, 0.5, "A"),
        _wp("b", 0.5, 1.0, "B"),
        _wp("c", 1.0, 1.5, "A"),
    ]
    assert merge.smooth_speaker_islands(pairs, max_island_words=0) == pairs
