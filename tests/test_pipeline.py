from transcript import pipeline
from transcript.models import Meta, Turn, Word


def _setup_mocks(mocker, tmp_path):
    wav = tmp_path / "in.m4a"
    wav.write_bytes(b"")
    prepared = tmp_path / "prepared.wav"
    prepared.write_bytes(b"")

    mocker.patch("transcript.pipeline.audio.prepare", return_value=(prepared, 5.0))
    mocker.patch(
        "transcript.pipeline.transcribe.run",
        return_value=[Word(" hi", 0.0, 1.0), Word(" there", 2.0, 3.0)],
    )
    mocker.patch(
        "transcript.pipeline.diarize.run",
        return_value=[Turn("Speaker 1", 0.0, 1.5), Turn("Speaker 2", 1.5, 3.0)],
    )
    return wav


def test_pipeline_returns_rendered_text(tmp_path, mocker):
    wav = _setup_mocks(mocker, tmp_path)
    out = pipeline.run(
        audio_path=wav,
        model="large-v3",
        language="fr",
        diarize=True,
        num_speakers=None,
        format_name="md",
        with_timestamps=True,
    )
    assert "## Speaker 1" in out
    assert "## Speaker 2" in out
    assert "# in.m4a" in out


def test_pipeline_no_diarize_assigns_single_speaker(tmp_path, mocker):
    wav = _setup_mocks(mocker, tmp_path)
    diarize_spy = mocker.patch("transcript.pipeline.diarize.run")
    out = pipeline.run(
        audio_path=wav,
        model="large-v3",
        language="fr",
        diarize=False,
        num_speakers=None,
        format_name="md",
        with_timestamps=True,
    )
    diarize_spy.assert_not_called()
    assert "## Speaker 1" in out
    assert "## Speaker 2" not in out


def test_pipeline_passes_num_speakers_to_diarize(tmp_path, mocker):
    wav = _setup_mocks(mocker, tmp_path)
    diarize_spy = mocker.patch(
        "transcript.pipeline.diarize.run",
        return_value=[Turn("Speaker 1", 0.0, 1.0)],
    )
    pipeline.run(
        audio_path=wav,
        model="large-v3",
        language="fr",
        diarize=True,
        num_speakers=2,
        format_name="md",
        with_timestamps=True,
    )
    _, kwargs = diarize_spy.call_args
    assert kwargs == {"num_speakers": 2}


def test_pipeline_meta_reflects_inputs(tmp_path, mocker):
    wav = _setup_mocks(mocker, tmp_path)
    out = pipeline.run(
        audio_path=wav,
        model="large-v3",
        language="fr",
        diarize=True,
        num_speakers=None,
        format_name="json",
        with_timestamps=True,
    )
    import json
    data = json.loads(out)
    assert data["meta"]["filename"] == "in.m4a"
    assert data["meta"]["model"] == "large-v3"
    assert data["meta"]["language"] == "fr"
    assert data["meta"]["duration"] == 5.0
    assert data["meta"]["speaker_count"] == 2  # two distinct speakers in fixture turns


def test_pipeline_cleans_up_temp_wav(tmp_path, mocker):
    wav = _setup_mocks(mocker, tmp_path)
    prepared = tmp_path / "prepared.wav"
    assert prepared.exists()  # set up by _setup_mocks
    pipeline.run(
        audio_path=wav,
        model="large-v3",
        language="fr",
        diarize=True,
        num_speakers=None,
        format_name="md",
        with_timestamps=True,
    )
    assert not prepared.exists(), "Pipeline should have unlinked the temp WAV"
