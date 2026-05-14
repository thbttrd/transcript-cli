from transcript.models import Turn, Utterance, Word

UNKNOWN = "Unknown"


def _best_speaker_hard_boundary(word: Word, turns: list[Turn]) -> str:
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
        if turn.start >= word.start and (
            earliest_upcoming is None or turn.start < earliest_upcoming.start
        ):
            earliest_upcoming = turn
        if latest_turn is None or turn.end > latest_turn.end:
            latest_turn = turn

    if best_turn is not None:
        return best_turn.speaker
    if earliest_upcoming is not None:
        return earliest_upcoming.speaker
    return latest_turn.speaker  # type: ignore[union-attr]


def assign_speakers(
    words: list[Word], turns: list[Turn]
) -> list[tuple[Word, str]]:
    """Per-word speaker assignment by max-overlap to turn ranges."""
    return [(w, _best_speaker_hard_boundary(w, turns)) for w in words]


def smooth_speaker_islands(
    word_speakers: list[tuple[Word, str]],
    *,
    max_island_words: int = 2,
) -> list[tuple[Word, str]]:
    """Flip same-speaker runs of length ≤ max_island_words sandwiched between
    two runs of the *same* other speaker.

    Mitigates the "boundary inside a phrase" artefact: when a turn boundary
    lands mid-word, the connective word ("y a", "tes") gets the marginal
    winner's label. If both neighbouring runs are the same speaker, the
    island is almost certainly mis-assigned and we flip it back.

    Genuine short interjections ("oui", "non") survive: those are typically
    followed by a *response* from the original speaker, not a continuation,
    so the same-speaker-on-both-sides guard preserves them.
    """
    n = len(word_speakers)
    if n < 3 or max_island_words < 1:
        return word_speakers

    runs: list[tuple[int, int, str]] = []  # (start_idx, end_idx_exclusive, speaker)
    run_start = 0
    for i in range(1, n):
        if word_speakers[i][1] != word_speakers[i - 1][1]:
            runs.append((run_start, i, word_speakers[run_start][1]))
            run_start = i
    runs.append((run_start, n, word_speakers[run_start][1]))

    if len(runs) < 3:
        return word_speakers

    # Decide flips against the ORIGINAL run list (no cascading): A,B,A,B,A
    # becomes all-A in one pass, not ping-ponging through interim states.
    flip_to: list[str | None] = [None] * len(runs)
    for idx in range(1, len(runs) - 1):
        prev_speaker = runs[idx - 1][2]
        if prev_speaker != runs[idx + 1][2]:
            continue
        if runs[idx][1] - runs[idx][0] > max_island_words:
            continue
        flip_to[idx] = prev_speaker

    if not any(flip_to):
        return word_speakers

    result = list(word_speakers)
    for idx, new_speaker in enumerate(flip_to):
        if new_speaker is None:
            continue
        for i in range(runs[idx][0], runs[idx][1]):
            result[i] = (result[i][0], new_speaker)
    return result


def collapse(word_speakers: list[tuple[Word, str]]) -> list[Utterance]:
    """Collapse consecutive same-speaker words into utterances."""
    if not word_speakers:
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

    for word, speaker in word_speakers:
        if speaker != current_speaker and current_words:
            flush()
            current_words = []
        current_speaker = speaker
        current_words.append(word)

    flush()
    return utterances


def assign(words: list[Word], turns: list[Turn]) -> list[Utterance]:
    """One-shot convenience: assign + collapse."""
    return collapse(assign_speakers(words, turns))
