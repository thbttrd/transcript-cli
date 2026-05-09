from transcript.models import Turn, Utterance, Word

UNKNOWN = "Unknown"


def _speaker_at(t: float, turns: list[Turn]) -> str:
    for turn in turns:
        if turn.start <= t <= turn.end:
            return turn.speaker
    return UNKNOWN


def assign(words: list[Word], turns: list[Turn]) -> list[Utterance]:
    """Assign each word to a speaker by timestamp midpoint, then collapse runs."""
    if not words:
        return []

    utterances: list[Utterance] = []
    current_speaker: str | None = None
    current_words: list[Word] = []

    def flush() -> None:
        if not current_words:
            return
        utterances.append(
            Utterance(
                speaker=current_speaker or UNKNOWN,
                start=current_words[0].start,
                end=current_words[-1].end,
                text="".join(w.text for w in current_words).strip(),
            )
        )

    for word in words:
        midpoint = (word.start + word.end) / 2
        speaker = _speaker_at(midpoint, turns)
        if speaker != current_speaker and current_words:
            flush()
            current_words = []
        current_speaker = speaker
        current_words.append(word)

    flush()
    return utterances
