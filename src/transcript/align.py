"""Forced word-level alignment via wav2vec2/MMS CTC.

Refines Whisper's word timestamps against the raw audio using
`MahmoudAshraf/mms-300m-1130-forced-aligner` — Meta's MMS-300m model, covering
1100+ languages with one set of weights. Replaces the imprecise DTW-derived
word boundaries from whisper.cpp that cause downstream diarization to assign
words to the wrong speaker. Falls through gracefully on any error.

Implementation note: we bypass ctc-forced-aligner's `load_alignment_model`
helper because it hardcodes `.to(device)` with no MPS branch — we want the
model on Apple Silicon's GPU when available. Otherwise the helper would be a
drop-in replacement.
"""
import sys
from pathlib import Path

from transcript import _debug
from transcript.models import Word

_MODEL_PATH = "MahmoudAshraf/mms-300m-1130-forced-aligner"
_DEBUG_LOG = _debug.log_path("align")

# Whisper uses ISO 639-1 (2-letter); the MMS aligner expects ISO 639-3 (3-letter).
# Covers the top ~30 Whisper-supported languages; unknown codes pass through and
# the aligner errors visibly so the user knows to add a mapping.
_ISO_639_1_TO_3 = {
    "en": "eng", "fr": "fra", "es": "spa", "de": "deu", "it": "ita",
    "pt": "por", "nl": "nld", "ru": "rus", "pl": "pol", "tr": "tur",
    "ja": "jpn", "ko": "kor", "zh": "cmn", "ar": "arb", "he": "heb",
    "cs": "ces", "ca": "cat", "sv": "swe", "no": "nor", "da": "dan",
    "fi": "fin", "uk": "ukr", "el": "ell", "ro": "ron", "hu": "hun",
    "vi": "vie", "th": "tha", "id": "ind", "hi": "hin", "fa": "fas",
}

_model = None
_tokenizer = None


def is_available() -> bool:
    """True if torch + transformers + ctc-forced-aligner can all be imported."""
    try:
        import torch  # noqa: F401
        from ctc_forced_aligner import load_audio  # noqa: F401
        from transformers import AutoModelForCTC  # noqa: F401
        return True
    except ImportError:
        return False


def run(audio_path: Path, words: list[Word], *, language: str) -> list[Word]:
    """Refine word timestamps in `words` by force-aligning their text against `audio_path`.

    Returns a list of Word objects with the same text and same indexing as the
    input, but with corrected start/end times. Punctuation-only entries (no
    letters — e.g. " ?", " .") keep their original Whisper timestamps because
    the aligner has nothing audible to anchor them to.

    On any failure (model load error, length mismatch, alignment exception),
    returns the input unchanged and writes a one-line diagnostic to
    `$TMPDIR/transcript-align.log`.
    """
    if not words:
        return words

    iso3 = _ISO_639_1_TO_3.get(language, language)

    alignable_indices = [i for i, w in enumerate(words) if _has_letters(w.text)]
    if not alignable_indices:
        _log("no alignable words; keeping Whisper timestamps")
        return words

    try:
        from ctc_forced_aligner import (
            generate_emissions,
            get_alignments,
            get_spans,
            load_audio,
            postprocess_results,
            preprocess_text,
        )
        model, tokenizer = _load_model()
        first_param = next(iter(model.parameters()))
        device = str(first_param.device)
        dtype = first_param.dtype

        clean_texts = [_strip_punct_edges(words[i].text) for i in alignable_indices]
        full_text = " ".join(clean_texts)

        waveform = load_audio(str(audio_path), dtype, device)
        emissions, stride = generate_emissions(model, waveform, batch_size=1)
        tokens_starred, text_starred = preprocess_text(
            full_text, romanize=True, language=iso3, split_size="word",
        )
        segments, scores, blank = get_alignments(emissions, tokens_starred, tokenizer)
        spans = get_spans(tokens_starred, segments, blank)
        aligned = postprocess_results(text_starred, spans, stride, scores)
    except Exception as e:
        _log(f"alignment failed: {type(e).__name__}: {e}")
        print(f"⚠ Forced alignment failed ({type(e).__name__}); keeping Whisper timestamps. See {_DEBUG_LOG}.", file=sys.stderr)
        return words

    if len(aligned) != len(alignable_indices):
        _log(f"length mismatch: aligner returned {len(aligned)} words, expected {len(alignable_indices)}")
        return words

    result = list(words)
    for src_idx, a in zip(alignable_indices, aligned, strict=True):
        original = words[src_idx]
        result[src_idx] = Word(text=original.text, start=float(a["start"]), end=float(a["end"]))

    _log(f"aligned {len(aligned)}/{len(words)} words (lang={iso3})")
    return result


def _load_model():
    """Lazy-load the aligner model once. Cached across calls within a process."""
    global _model, _tokenizer
    if _model is None:
        import torch
        from transformers import AutoModelForCTC, AutoTokenizer
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        m = AutoModelForCTC.from_pretrained(_MODEL_PATH, dtype=torch.float32)
        m = m.to(device)
        m.train(False)  # inference mode (project convention; matches diarize.py)
        _model = m
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_PATH)
    return _model, _tokenizer


def _has_letters(s: str) -> bool:
    return any(c.isalpha() for c in s)


def _strip_punct_edges(s: str) -> str:
    """Strip leading/trailing punctuation and whitespace; preserve internal apostrophes and hyphens."""
    return s.strip(" \t,.!?;:\"'()[]…«»")


def _log(msg: str) -> None:
    _debug.write(_DEBUG_LOG, f"{msg}\n", append=True)
