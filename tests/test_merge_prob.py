import numpy as np

from transcript.merge import assign_speakers
from transcript.models import Turn, Word

FRAME_S = 0.08  # 80 ms — Sortformer's frame size


def test_prob_based_assigns_argmax_speaker_over_word_frames():
    # Word spans frames 3..7 (0.24..0.56s). probs spike on speaker B in those frames.
    words = [Word(text=" foo", start=0.24, end=0.56)]
    turns = []  # turns ignored in prob mode
    probs = np.zeros((20, 4), dtype=np.float32)
    probs[3:8, 1] = 0.9   # speaker B (index 1)

    pairs = assign_speakers(words, turns, strategy="prob_based", probs=probs)
    assert len(pairs) == 1
    _, speaker = pairs[0]
    assert speaker == "Speaker 2"


def test_prob_based_handles_word_outside_tensor_range():
    # Word at 5.0s but tensor only covers 10 frames (0..0.8s). Fall back to "Unknown".
    words = [Word(text=" foo", start=5.0, end=5.5)]
    probs = np.zeros((10, 4), dtype=np.float32)
    pairs = assign_speakers(words, [], strategy="prob_based", probs=probs)
    _, speaker = pairs[0]
    assert speaker == "Unknown"


def test_prob_based_with_no_probs_falls_back_to_hard_boundary():
    """If strategy=prob_based but probs is None, fall back gracefully."""
    words = [Word(text=" foo", start=0.0, end=0.5)]
    turns = [Turn("Speaker 1", 0.0, 1.0)]
    pairs = assign_speakers(words, turns, strategy="prob_based", probs=None)
    _, speaker = pairs[0]
    assert speaker == "Speaker 1"


def test_hard_boundary_strategy_unchanged():
    """The pre-existing strategy still works when explicitly requested."""
    words = [Word(text=" foo", start=0.0, end=0.5)]
    turns = [Turn("Speaker 1", 0.0, 1.0)]
    pairs = assign_speakers(words, turns, strategy="hard_boundary", probs=None)
    _, speaker = pairs[0]
    assert speaker == "Speaker 1"
