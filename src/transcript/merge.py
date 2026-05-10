from transcript.models import Turn, Utterance, Word

UNKNOWN = "Unknown"


def _best_speaker(word: Word, turns: list[Turn]) -> str:
    if not turns:
        return UNKNOWN

    best_overlap = 0.0
    best_turn: Turn | None = None
    earliest_upcoming: Turn | None = None
    latest_turn: Turn | None = None
    for turn in turns:
        overlap = min(word.end, turn.end) - max(word.start, turn.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_turn = turn
        if turn.start >= word.start and (earliest_upcoming is None or turn.start < earliest_upcoming.start):
            earliest_upcoming = turn
        if latest_turn is None or turn.end > latest_turn.end:
            latest_turn = turn

    if best_turn is not None:
        return best_turn.speaker
    # No overlap: word sits in a gap. Prefer the upcoming turn — boundary words
    # are usually the first word of the new speaker, not the last of the previous.
    if earliest_upcoming is not None:
        return earliest_upcoming.speaker
    return latest_turn.speaker  # type: ignore[union-attr]


def assign(words: list[Word], turns: list[Turn]) -> list[Utterance]:
    """Assign each word to a speaker (max overlap, fall back to upcoming turn), then collapse runs."""
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
        speaker = _best_speaker(word, turns)
        if speaker != current_speaker and current_words:
            flush()
            current_words = []
        current_speaker = speaker
        current_words.append(word)

    flush()
    return utterances
