"""Metric computation for the bench harness.

Primary: cpWER (concatenated minimum-permutation WER) via meeteval.
Secondary diagnostics: speaker-agnostic WER, DER, and the cpWER-WER
decomposition (the "speaker-assignment error rate" — right word, wrong
speaker — which is the failure mode the prob-based merge targets).
"""
import math
import re
import unicodedata
from dataclasses import dataclass
from itertools import permutations

import numpy as np

from transcript.models import Utterance

_PUNCT_RE   = re.compile(r"[^\w\s\-']", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")
_SUMM_RE_MARKERS = re.compile(r"[@*+]")

_SESSION_ID = "bench"


@dataclass(frozen=True)
class ClipMetrics:
    cpwer: float
    wer:   float
    der:   float
    speaker_assignment_error_rate: float

    def __post_init__(self) -> None:
        for name in ("cpwer", "wer", "der", "speaker_assignment_error_rate"):
            v = getattr(self, name)
            if math.isnan(v) or v < 0.0:
                raise ValueError(f"{name} must be a non-negative finite rate, got {v}")


def normalise(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = _SUMM_RE_MARKERS.sub(" ", s)
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WHITESPACE.sub(" ", s).strip()
    return s


def _utterances_to_seglst(utterances: list[Utterance]) -> list[dict]:
    """Build a meeteval SegLST payload (one segment per non-empty speaker)."""
    by_speaker: dict[str, list[str]] = {}
    for u in utterances:
        text = normalise(u.text)
        if not text:
            continue
        by_speaker.setdefault(u.speaker, []).append(text)
    return [
        {"session_id": _SESSION_ID, "speaker": spk, "words": " ".join(words)}
        for spk, words in by_speaker.items()
    ]


def _speaker_agnostic_wer(hyp: list[Utterance], ref: list[Utterance]) -> float:
    """Concatenate everything ignoring speakers; compute plain WER via meeteval."""
    from meeteval.wer import siso_word_error_rate
    hyp_text = normalise(" ".join(u.text for u in hyp))
    ref_text = normalise(" ".join(u.text for u in ref))
    result = siso_word_error_rate(reference=ref_text, hypothesis=hyp_text)
    return float(result.error_rate)


def _cpwer(hyp: list[Utterance], ref: list[Utterance]) -> float:
    from meeteval.wer import cpwer as meet_cpwer
    hyp_seg = _utterances_to_seglst(hyp)
    ref_seg = _utterances_to_seglst(ref)
    if not ref_seg:
        # No reference words to score against: define cpWER as 0 if hyp also
        # empty, else 1 (every hypothesis word is an insertion).
        return 0.0 if not hyp_seg else 1.0
    if not hyp_seg:
        # Reference has words, hypothesis doesn't — every ref word is a deletion.
        return 1.0
    # meeteval.cpwer with SegLST inputs returns dict[session_id, CPErrorRate].
    result = meet_cpwer(reference=ref_seg, hypothesis=hyp_seg)
    per_session = result[_SESSION_ID] if isinstance(result, dict) else result
    return float(per_session.error_rate)


def _der(hyp: list[Utterance], ref: list[Utterance]) -> float:
    """Speaker-only DER approximation: fraction of reference frames labelled with
    the wrong (post-permutation) hypothesis speaker, at 10 ms resolution.

    The permutation maps hyp labels onto ref labels (not onto hyp labels), so
    a perfectly-diarized clip with `hyp={"Speaker 1","Speaker 2"}` and
    `ref={"MEE068","FEE066"}` scores DER=0 after the optimal pairing.
    """
    if not ref:
        return 0.0
    end = max(u.end for u in ref)
    step = 0.01
    n = int(end / step) + 1
    ref_arr = np.full(n, "", dtype=object)
    hyp_arr = np.full(n, "", dtype=object)
    for u in ref:
        ref_arr[int(u.start / step):int(u.end / step)] = u.speaker
    for u in hyp:
        hyp_arr[int(u.start / step):int(min(u.end, end) / step)] = u.speaker

    ref_spks = sorted({s for s in ref_arr if s})
    hyp_spks = sorted({s for s in hyp_arr if s})
    if not ref_spks or not hyp_spks:
        return 1.0

    total = float(np.sum(ref_arr != ""))
    if total == 0.0:
        return 1.0

    # Brute-force optimal hyp→ref label permutation (cap: 4 speakers → 24 perms).
    # When hyp/ref speaker counts differ, the smaller set permutes against subsets
    # of the larger; unmatched labels carry through unchanged and always mismatch.
    best_err = math.inf
    if len(hyp_spks) <= len(ref_spks):
        for ref_perm in permutations(ref_spks, len(hyp_spks)):
            mapping = dict(zip(hyp_spks, ref_perm, strict=True))
            remapped = np.array([mapping.get(s, s) for s in hyp_arr], dtype=object)
            err = float(np.sum(remapped != ref_arr)) / total
            best_err = min(best_err, err)
    else:
        for hyp_perm in permutations(hyp_spks, len(ref_spks)):
            mapping = dict(zip(hyp_perm, ref_spks, strict=True))
            remapped = np.array([mapping.get(s, s) for s in hyp_arr], dtype=object)
            err = float(np.sum(remapped != ref_arr)) / total
            best_err = min(best_err, err)
    return best_err if best_err != math.inf else 1.0


def score(hypothesis: list[Utterance], reference: list[Utterance]) -> ClipMetrics:
    w = _speaker_agnostic_wer(hypothesis, reference)
    c = _cpwer(hypothesis, reference)
    d = _der(hypothesis, reference)
    return ClipMetrics(
        cpwer = c,
        wer   = w,
        der   = d,
        speaker_assignment_error_rate = max(0.0, c - w),
    )
