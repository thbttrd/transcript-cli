import json

from bench.artefacts import save_diff, save_transcript
from transcript.models import Utterance


def test_save_transcript_writes_hypothesis_and_reference(tmp_path):
    hypothesis = [Utterance("Speaker 1", 0.0, 1.0, "bonjour"),
                  Utterance("Speaker 2", 1.0, 2.0, "salut")]
    reference = [Utterance("Speaker A", 0.0, 1.0, "bonjour"),
                 Utterance("Speaker B", 1.0, 2.0, "salut")]
    path = save_transcript(
        results_dir=tmp_path, tier=1, clip_id="AMI:EN2002a",
        config_fingerprint="abc123",
        hypothesis=hypothesis, reference=reference,
    )
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["clip_id"] == "AMI:EN2002a"
    assert data["config_fingerprint"] == "abc123"
    assert len(data["hypothesis"]) == 2
    assert len(data["reference"]) == 2
    assert data["hypothesis"][0]["text"] == "bonjour"


def test_save_diff_writes_word_ops_and_totals(tmp_path):
    speaker_permutation = {"Speaker 1": "Speaker A", "Speaker 2": "Speaker B"}
    word_ops = [
        {"op": "equal", "ref_word": "bonjour", "hyp_word": "bonjour",
         "ref_speaker": "A", "hyp_speaker": "A"},
        {"op": "sub", "ref_word": "allons", "hyp_word": "allon",
         "ref_speaker": "A", "hyp_speaker": "A"},
        {"op": "speaker_swap", "ref_word": "oui", "hyp_word": "oui",
         "ref_speaker": "B", "hyp_speaker": "A"},
    ]
    path = save_diff(
        results_dir=tmp_path, tier=2, clip_id="SUMM-RE:001",
        config_fingerprint="xyz789",
        speaker_permutation=speaker_permutation, word_ops=word_ops,
    )
    data = json.loads(path.read_text())
    assert data["speaker_permutation"] == speaker_permutation
    assert len(data["word_ops"]) == 3
    assert data["totals"] == {"sub": 1, "ins": 0, "del": 0, "speaker_swap": 1}


def test_paths_follow_layout_spec(tmp_path):
    p = save_transcript(
        results_dir=tmp_path, tier=3, clip_id="AMI:EN2002a",
        config_fingerprint="deadbeef0001",
        hypothesis=[], reference=[],
    )
    assert p == tmp_path / "transcripts" / "tier-3" / "AMI_EN2002a" / "deadbeef0001.json"
