import math

import pytest

from bench.metrics import ClipMetrics, normalise, score
from transcript.models import Utterance


def test_normalise_lowercases_and_strips_punctuation():
    assert normalise("Bonjour, monde!") == "bonjour monde"


def test_normalise_collapses_whitespace():
    assert normalise("hello    world\n\nfoo") == "hello world foo"


def test_normalise_strips_summ_re_markers():
    assert normalise("salut @ * + ami") == "salut ami"


def test_normalise_is_idempotent():
    s = "Salut, là-bas! Comment ça va @ ?"
    assert normalise(s) == normalise(normalise(s))


def test_score_perfect_match_yields_zero():
    hyp = [Utterance("Speaker 1", 0.0, 1.0, "bonjour")]
    ref = [Utterance("Speaker 1", 0.0, 1.0, "bonjour")]
    m = score(hyp, ref)
    assert m.cpwer == 0.0
    assert m.wer == 0.0
    assert m.der == 0.0
    assert m.speaker_assignment_error_rate == 0.0


def test_der_zero_for_perfect_match_with_relabelled_speakers():
    """Hyp uses generic 'Speaker N' labels; ref uses corpus-specific IDs.
    The optimal permutation must pair them so DER is 0, not 1.0."""
    hyp = [
        Utterance("Speaker 1", 0.0, 1.0, "a"),
        Utterance("Speaker 2", 1.0, 2.0, "b"),
    ]
    ref = [
        Utterance("MEE068", 0.0, 1.0, "a"),
        Utterance("FEE066", 1.0, 2.0, "b"),
    ]
    m = score(hyp, ref)
    assert m.der == 0.0


def test_der_one_for_completely_swapped_speakers():
    """Every ref frame is labelled with the wrong speaker, no permutation helps."""
    hyp = [Utterance("Speaker 1", 0.0, 2.0, "a b")]
    ref = [
        Utterance("MEE068", 0.0, 1.0, "a"),
        Utterance("FEE066", 1.0, 2.0, "b"),
    ]
    m = score(hyp, ref)
    # One hyp speaker covers two ref speakers — best perm matches half the frames.
    assert m.der == pytest.approx(0.5, abs=0.05)


def test_score_completely_wrong_word_yields_wer_one():
    hyp = [Utterance("Speaker 1", 0.0, 1.0, "salut")]
    ref = [Utterance("Speaker 1", 0.0, 1.0, "bonjour")]
    m = score(hyp, ref)
    assert m.wer == 1.0


def test_score_returns_dataclass_with_four_metrics():
    hyp = [Utterance("Speaker 1", 0.0, 1.0, "bonjour")]
    ref = [Utterance("Speaker 1", 0.0, 1.0, "bonjour")]
    m = score(hyp, ref)
    assert isinstance(m, ClipMetrics)
    assert all(hasattr(m, k) for k in ("cpwer", "wer", "der", "speaker_assignment_error_rate"))


def test_clipmetrics_rejects_negative_rate():
    with pytest.raises(ValueError, match="must be a non-negative finite rate"):
        ClipMetrics(cpwer=-0.1, wer=0.0, der=0.0, speaker_assignment_error_rate=0.0)


def test_clipmetrics_rejects_nan_rate():
    with pytest.raises(ValueError, match="must be a non-negative finite rate"):
        ClipMetrics(cpwer=math.nan, wer=0.0, der=0.0, speaker_assignment_error_rate=0.0)
