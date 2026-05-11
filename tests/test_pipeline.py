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
        return_value=([Word(" hi", 0.0, 1.0), Word(" there", 2.0, 3.0)], "fr"),
    )
    mocker.patch(
        "transcript.pipeline.diarize.run",
        return_value=[Turn("Speaker 1", 0.0, 1.5), Turn("Speaker 2", 1.5, 3.0)],
    )
    mocker.patch("transcript.pipeline.llm_fix.is_available", return_value=False)
    # Default: don't load the real aligner in tests. Tests that need it can override.
    mocker.patch("transcript.pipeline.align.is_available", return_value=False)
    return wav


def test_pipeline_returns_rendered_text(tmp_path, mocker):
    wav = _setup_mocks(mocker, tmp_path)
    out = pipeline.run(
        audio_path=wav,
        model="large-v3",
        language="fr",
        with_diarization=True,
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
        with_diarization=False,
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
        with_diarization=True,
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
        with_diarization=True,
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


def test_pipeline_calls_llm_fix_when_enabled_and_available(tmp_path, mocker):
    wav = _setup_mocks(mocker, tmp_path)
    mocker.patch("transcript.pipeline.llm_fix.is_available", return_value=True)
    spy = mocker.patch(
        "transcript.pipeline.llm_fix.apply",
        side_effect=lambda pairs, **_: pairs,
    )
    pipeline.run(
        audio_path=wav,
        model="large-v3",
        language="fr",
        with_diarization=True,
        num_speakers=2,
        format_name="md",
        with_timestamps=True,
        with_llm_fix=True,
    )
    spy.assert_called_once()


def test_pipeline_skips_llm_fix_by_default(tmp_path, mocker):
    """Regression: LLM cleanup is opt-in. Default pipeline must not call it."""
    wav = _setup_mocks(mocker, tmp_path)
    mocker.patch("transcript.pipeline.llm_fix.is_available", return_value=True)
    spy = mocker.patch("transcript.pipeline.llm_fix.apply")
    pipeline.run(
        audio_path=wav,
        model="large-v3",
        language="fr",
        with_diarization=True,
        num_speakers=2,
        format_name="md",
        with_timestamps=True,
    )
    spy.assert_not_called()


def test_pipeline_skips_llm_fix_when_unavailable(tmp_path, mocker):
    wav = _setup_mocks(mocker, tmp_path)
    spy = mocker.patch("transcript.pipeline.llm_fix.apply")
    pipeline.run(
        audio_path=wav,
        model="large-v3",
        language="fr",
        with_diarization=True,
        num_speakers=None,
        format_name="md",
        with_timestamps=True,
        with_llm_fix=True,
    )
    spy.assert_not_called()


def test_pipeline_calls_align_when_available(tmp_path, mocker):
    wav = _setup_mocks(mocker, tmp_path)
    mocker.patch("transcript.pipeline.align.is_available", return_value=True)
    spy = mocker.patch(
        "transcript.pipeline.align.run",
        side_effect=lambda _wav, words, **_: words,
    )
    pipeline.run(
        audio_path=wav,
        model="large-v3",
        language="fr",
        with_diarization=True,
        num_speakers=None,
        format_name="md",
        with_timestamps=True,
    )
    spy.assert_called_once()
    _, kwargs = spy.call_args
    assert kwargs["language"] == "fr"  # detected_lang flows through


def test_pipeline_respects_no_align(tmp_path, mocker):
    wav = _setup_mocks(mocker, tmp_path)
    mocker.patch("transcript.pipeline.align.is_available", return_value=True)
    spy = mocker.patch("transcript.pipeline.align.run")
    pipeline.run(
        audio_path=wav,
        model="large-v3",
        language="fr",
        with_diarization=True,
        num_speakers=None,
        format_name="md",
        with_timestamps=True,
        with_align=False,
    )
    spy.assert_not_called()


def test_pipeline_skips_llm_fix_when_no_diarize(tmp_path, mocker):
    wav = _setup_mocks(mocker, tmp_path)
    mocker.patch("transcript.pipeline.llm_fix.is_available", return_value=True)
    spy = mocker.patch("transcript.pipeline.llm_fix.apply")
    pipeline.run(
        audio_path=wav,
        model="large-v3",
        language="fr",
        with_diarization=False,
        num_speakers=None,
        format_name="md",
        with_timestamps=True,
        with_llm_fix=True,
    )
    spy.assert_not_called()


def test_pipeline_cleans_up_temp_wav(tmp_path, mocker):
    wav = _setup_mocks(mocker, tmp_path)
    prepared = tmp_path / "prepared.wav"
    assert prepared.exists()  # set up by _setup_mocks
    pipeline.run(
        audio_path=wav,
        model="large-v3",
        language="fr",
        with_diarization=True,
        num_speakers=None,
        format_name="md",
        with_timestamps=True,
    )
    assert not prepared.exists(), "Pipeline should have unlinked the temp WAV"
