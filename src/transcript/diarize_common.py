"""Helpers shared across diarization backends."""
import logging
from pathlib import Path

from transcript.models import Turn


def relabel_by_first_appearance(turns: list[Turn]) -> list[Turn]:
    """Renumber speakers so 'Speaker 1' is whoever talks first.

    Sorts by start to determine label order, then returns turns in original
    list order with new labels. Works whether the input is already chronological
    (Sortformer) or arbitrarily ordered by a clustering step (DiariZen).
    """
    if not turns:
        return turns
    label_map: dict[str, str] = {}
    for t in sorted(turns, key=lambda t: t.start):
        if t.speaker not in label_map:
            label_map[t.speaker] = f"Speaker {len(label_map) + 1}"
    return [Turn(speaker=label_map[t.speaker], start=t.start, end=t.end) for t in turns]


def filter_and_warn(
    turns: list[Turn],
    *,
    num_speakers: int | None,
    backend_label: str,
    wav_path: Path,
    log: logging.Logger,
) -> list[Turn]:
    """Cap to first `num_speakers` and warn if the result is empty."""
    if num_speakers is not None:
        keep = {f"Speaker {i + 1}" for i in range(num_speakers)}
        turns = [t for t in turns if t.speaker in keep]
    if not turns:
        log.warning(
            "%s returned no turns for %s — every word will be labelled Unknown",
            backend_label,
            wav_path,
        )
    return turns
