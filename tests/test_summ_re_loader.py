import numpy as np
import soundfile as sf

from bench.datasets.summ_re import _mix_tracks, _synthesise_rttm, _synthesise_stm


def test_synthesise_rttm_emits_one_line_per_segment(tmp_path):
    tracks = [
        {"speaker_id": "001", "segments": [
            {"start": 0.0, "end": 1.0}, {"start": 1.5, "end": 2.0},
        ]},
        {"speaker_id": "002", "segments": [
            {"start": 1.0, "end": 1.5},
        ]},
    ]
    out = tmp_path / "ref.rttm"
    _synthesise_rttm(tracks, meeting_id="m1", out_path=out)
    lines = out.read_text().splitlines()
    assert len(lines) == 3
    for line in lines:
        parts = line.split()
        assert parts[0] == "SPEAKER"
        assert parts[1] == "m1"


def test_synthesise_stm_concatenates_words_per_speaker(tmp_path):
    tracks = [
        {"speaker_id": "001", "segments": [
            {"start": 0.0, "end": 1.0, "words": [
                {"word": "bonjour", "start": 0.0, "end": 0.5},
                {"word": "toi",     "start": 0.5, "end": 1.0},
            ]},
        ]},
    ]
    out = tmp_path / "ref.stm"
    _synthesise_stm(tracks, meeting_id="m1", out_path=out)
    line = out.read_text().strip()
    assert "bonjour" in line
    assert "toi" in line


def test_mix_tracks_writes_16khz_mono_wav(tmp_path):
    track_a = tmp_path / "a.wav"
    track_b = tmp_path / "b.wav"
    sf.write(track_a, np.zeros(32000, dtype=np.float32), 32000)
    sf.write(track_b, np.ones(32000, dtype=np.float32) * 0.1, 32000)

    out = tmp_path / "mixed.wav"
    _mix_tracks([track_a, track_b], out_path=out)
    data, sr = sf.read(out)
    assert sr == 16000
    assert data.ndim == 1
    assert abs(len(data) - 16000) < 100
