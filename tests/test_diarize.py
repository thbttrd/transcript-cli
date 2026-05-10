from transcript import diarize
from transcript.models import Turn


def test_relabel_assigns_speaker_n_in_first_appearance_order():
    raw = [
        (0.0, 1.0, "spk_zzz"),
        (1.0, 2.0, "spk_aaa"),
        (2.0, 3.0, "spk_zzz"),
        (3.0, 4.0, "spk_aaa"),
        (4.0, 5.0, "spk_qqq"),
    ]
    turns = diarize._relabel(raw)
    assert turns == [
        Turn("Speaker 1", 0.0, 1.0),
        Turn("Speaker 2", 1.0, 2.0),
        Turn("Speaker 1", 2.0, 3.0),
        Turn("Speaker 2", 3.0, 4.0),
        Turn("Speaker 3", 4.0, 5.0),
    ]


def test_parse_sortformer_segments_handles_rttm_lines():
    raw = diarize._parse_sortformer_segments([
        "0.50 2.66 speaker_0",
        "4.28 5.73 speaker_1",
        "junk",  # ignored
        "1.0 not-a-float speaker_0",  # ignored
    ])
    assert raw == [
        (0.50, 2.66, "speaker_0"),
        (4.28, 5.73, "speaker_1"),
    ]
