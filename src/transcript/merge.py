from typing import Literal

import numpy as np

from transcript.models import Turn, Utterance, Word

UNKNOWN = "Unknown"
FRAME_S = 0.08  # Sortformer frame size — 80 ms


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


def _best_speaker_prob_based(word: Word, probs: np.ndarray) -> str:
    """Average per-frame probabilities over the word's frame window; argmax to speaker."""
    n_frames = probs.shape[0]
    start_frame = int(word.start / FRAME_S)
    end_frame = max(start_frame + 1, int(np.ceil(word.end / FRAME_S)))
    if start_frame >= n_frames:
        return UNKNOWN
    end_frame = min(end_frame, n_frames)
    window = probs[start_frame:end_frame]
    if window.size == 0:
        return UNKNOWN
    mean = window.mean(axis=0)
    idx = int(mean.argmax())
    return f"Speaker {idx + 1}"


def assign_speakers(
    words: list[Word],
    turns: list[Turn],
    *,
    strategy: Literal["hard_boundary", "prob_based"] = "hard_boundary",
    probs: np.ndarray | None = None,
) -> list[tuple[Word, str]]:
    """Per-word speaker assignment.

    - strategy="hard_boundary": max-overlap to turn ranges (existing logic).
    - strategy="prob_based": average per-frame probabilities over word.start..end,
      argmax over the 4 speaker columns. Requires `probs` (a [T x 4] array).
      Falls back to hard_boundary silently if probs is None.
    """
    if strategy == "prob_based" and probs is not None:
        return [(w, _best_speaker_prob_based(w, probs)) for w in words]
    return [(w, _best_speaker_hard_boundary(w, turns)) for w in words]


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
    """One-shot convenience: hard_boundary assign + collapse."""
    return collapse(assign_speakers(words, turns))
