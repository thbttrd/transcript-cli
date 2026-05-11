from pathlib import Path

from transcript import align
from transcript.models import Word


def _mock_aligner(mocker, *, aligned_words):
    """Patch the full ctc-forced-aligner pipeline + model loader.
    `aligned_words` is what `postprocess_results` should return."""
    mock_param = mocker.MagicMock(device="cpu", dtype="float32")
    mock_model = mocker.MagicMock()
    mock_model.parameters.return_value = iter([mock_param])
    mocker.patch("transcript.align._load_model", return_value=(mock_model, mocker.MagicMock()))
    mocker.patch("ctc_forced_aligner.load_audio", return_value=mocker.MagicMock())
    mocker.patch("ctc_forced_aligner.generate_emissions", return_value=(mocker.MagicMock(), 0.02))
    mocker.patch("ctc_forced_aligner.preprocess_text", return_value=([], []))
    mocker.patch("ctc_forced_aligner.get_alignments", return_value=([], [], None))
    mocker.patch("ctc_forced_aligner.get_spans", return_value=[])
    return mocker.patch("ctc_forced_aligner.postprocess_results", return_value=aligned_words)


def test_has_letters():
    assert align._has_letters(" Salut")
    assert align._has_letters("c'est")
    assert not align._has_letters(" ?")
    assert not align._has_letters(",")
    assert not align._has_letters("…")


def test_strip_punct_edges_preserves_internal_apostrophes_and_hyphens():
    assert align._strip_punct_edges(" Chouchou,") == "Chouchou"
    assert align._strip_punct_edges("c'est") == "c'est"
    assert align._strip_punct_edges(" vas-y.") == "vas-y"
    assert align._strip_punct_edges(" ?") == ""


def test_iso_639_1_to_3_covers_french_and_common_languages():
    """French is the primary target; the mapping must include it for the user's case."""
    assert align._ISO_639_1_TO_3["fr"] == "fra"
    assert align._ISO_639_1_TO_3["en"] == "eng"
    assert align._ISO_639_1_TO_3["es"] == "spa"


def test_run_returns_input_unchanged_when_empty():
    assert align.run(Path("/x"), [], language="fr") == []


def test_run_falls_through_when_only_punctuation(mocker, tmp_path):
    """Pipeline of punctuation-only Words has nothing for the aligner to anchor — pass through."""
    words = [Word(" ?", 1.0, 1.1), Word(" .", 2.0, 2.05)]
    spy = mocker.patch("transcript.align._load_model")
    out = align.run(tmp_path / "fake.wav", words, language="fr")
    assert out == words
    spy.assert_not_called()


def test_run_falls_through_on_alignment_exception(mocker, tmp_path):
    words = [Word(" hi", 0.0, 0.5), Word(" there", 0.5, 1.0)]
    mocker.patch("transcript.align._load_model", side_effect=RuntimeError("boom"))
    out = align.run(tmp_path / "fake.wav", words, language="fr")
    assert out == words


def test_run_falls_through_on_length_mismatch(mocker, tmp_path):
    """If the aligner emits a different number of words than we expected, keep originals."""
    words = [Word(" hi", 0.0, 0.5), Word(" there", 0.5, 1.0)]
    _mock_aligner(mocker, aligned_words=[
        {"start": 0.1, "end": 0.4, "text": "hi", "score": -0.5},
    ])
    out = align.run(tmp_path / "fake.wav", words, language="fr")
    assert out == words


def test_run_replaces_timestamps_on_success(mocker, tmp_path):
    """Aligned timestamps must replace Whisper's, but text and order are preserved."""
    words = [Word(" hi", 0.0, 0.5), Word(" there", 0.5, 1.0)]
    _mock_aligner(mocker, aligned_words=[
        {"start": 0.10, "end": 0.35, "text": "hi", "score": -0.5},
        {"start": 0.55, "end": 0.95, "text": "there", "score": -0.3},
    ])
    out = align.run(tmp_path / "fake.wav", words, language="fr")
    assert out[0].text == " hi" and out[0].start == 0.10 and out[0].end == 0.35
    assert out[1].text == " there" and out[1].start == 0.55 and out[1].end == 0.95


def test_run_keeps_punctuation_only_timestamps_unchanged(mocker, tmp_path):
    """Punctuation-only Words are skipped during alignment; their original timestamps survive."""
    words = [
        Word(" hi", 0.0, 0.5),
        Word(" ?", 0.5, 0.55),
        Word(" there", 0.6, 1.0),
    ]
    _mock_aligner(mocker, aligned_words=[
        {"start": 0.10, "end": 0.35, "text": "hi", "score": -0.5},
        {"start": 0.55, "end": 0.95, "text": "there", "score": -0.3},
    ])
    out = align.run(tmp_path / "fake.wav", words, language="fr")
    assert out[1].text == " ?" and out[1].start == 0.5 and out[1].end == 0.55
    assert out[0].start == 0.10  # alignable, refined
    assert out[2].start == 0.55  # alignable, refined
