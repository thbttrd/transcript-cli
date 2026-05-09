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


def test_word_midpoint_is_what_decides_assignment():
    # Word from 0.9 to 1.2; midpoint 1.05; turn boundary at 1.0.
    # Midpoint is in turn 2, so word is assigned to Speaker 2.
    words = [w(" overlap", 0.9, 1.2)]
    turns = [t("Speaker 1", 0.0, 1.0), t("Speaker 2", 1.0, 2.0)]
    result = assign(words, turns)
    assert result == [Utterance(speaker="Speaker 2", start=0.9, end=1.2, text="overlap")]


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


def test_word_in_gap_between_turns_is_unknown():
    words = [w(" lonely", 1.0, 1.5)]
    turns = [t("Speaker 1", 0.0, 0.5), t("Speaker 2", 2.0, 3.0)]
    result = assign(words, turns)
    assert result == [Utterance(speaker="Unknown", start=1.0, end=1.5, text="lonely")]


def test_text_is_stripped_and_concatenated_in_order():
    words = [w("  he", 0.0, 0.1), w("llo", 0.1, 0.2), w("  world  ", 0.2, 0.3)]
    turns = [t("Speaker 1", 0.0, 0.5)]
    result = assign(words, turns)
    assert result[0].text == "hello  world"
