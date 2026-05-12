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
    assert m.speaker_assignment_error_rate == 0.0


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
