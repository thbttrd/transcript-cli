import pytest

from transcript import pipeline
from transcript.models import Turn, Word
from transcript.pipeline_config import (
    AlignConfig,
    DiarizeConfig,
    LLMFixConfig,
    PipelineConfig,
    TranscribeConfig,
)


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
    mocker.patch("transcript.pipeline.align.is_available", return_value=False)
    return wav


def test_pipeline_returns_utterances_and_meta(tmp_path, mocker):
    wav = _setup_mocks(mocker, tmp_path)
    cfg = PipelineConfig(transcribe=TranscribeConfig(language="fr"))
    utterances, meta = pipeline.run(audio_path=wav, config=cfg, with_diarization=True)
    assert len(utterances) == 2
    assert {u.speaker for u in utterances} == {"Speaker 1", "Speaker 2"}
    assert meta.filename == "in.m4a"
    assert meta.duration == 5.0
    assert meta.language == "fr"
    assert meta.model == "large-v3"
    assert meta.speaker_count == 2


def test_pipeline_no_diarize_assigns_single_speaker(tmp_path, mocker):
    wav = _setup_mocks(mocker, tmp_path)
    diarize_spy = mocker.patch("transcript.pipeline.diarize.run")
    utterances, _ = pipeline.run(
        audio_path=wav, config=PipelineConfig(), with_diarization=False
    )
    diarize_spy.assert_not_called()
    assert {u.speaker for u in utterances} == {"Speaker 1"}


def test_pipeline_passes_num_speakers_to_diarize(tmp_path, mocker):
    wav = _setup_mocks(mocker, tmp_path)
    diarize_spy = mocker.patch(
        "transcript.pipeline.diarize.run",
        return_value=[Turn("Speaker 1", 0.0, 1.0)],
    )
    cfg = PipelineConfig(diarize=DiarizeConfig(num_speakers=2))
    pipeline.run(audio_path=wav, config=cfg, with_diarization=True)
    _, kwargs = diarize_spy.call_args
    assert kwargs["config"].num_speakers == 2


def test_pipeline_calls_llm_fix_when_enabled_and_available(tmp_path, mocker):
    wav = _setup_mocks(mocker, tmp_path)
    mocker.patch("transcript.pipeline.llm_fix.is_available", return_value=True)
    spy = mocker.patch(
        "transcript.pipeline.llm_fix.apply",
        side_effect=lambda pairs, **_: pairs,
    )
    cfg = PipelineConfig(llm_fix=LLMFixConfig(enabled=True))
    pipeline.run(audio_path=wav, config=cfg, with_diarization=True)
    spy.assert_called_once()


def test_pipeline_skips_llm_fix_by_default(tmp_path, mocker):
    wav = _setup_mocks(mocker, tmp_path)
    mocker.patch("transcript.pipeline.llm_fix.is_available", return_value=True)
    spy = mocker.patch("transcript.pipeline.llm_fix.apply")
    pipeline.run(audio_path=wav, config=PipelineConfig(), with_diarization=True)
    spy.assert_not_called()


def test_pipeline_skips_llm_fix_when_unavailable(tmp_path, mocker):
    wav = _setup_mocks(mocker, tmp_path)
    spy = mocker.patch("transcript.pipeline.llm_fix.apply")
    cfg = PipelineConfig(llm_fix=LLMFixConfig(enabled=True))
    pipeline.run(audio_path=wav, config=cfg, with_diarization=True)
    spy.assert_not_called()


def test_pipeline_calls_align_when_enabled_and_available(tmp_path, mocker):
    wav = _setup_mocks(mocker, tmp_path)
    mocker.patch("transcript.pipeline.align.is_available", return_value=True)
    spy = mocker.patch(
        "transcript.pipeline.align.run",
        side_effect=lambda _wav, words, **_: words,
    )
    pipeline.run(audio_path=wav, config=PipelineConfig(), with_diarization=True)
    spy.assert_called_once()
    _, kwargs = spy.call_args
    assert kwargs["language"] == "fr"


def test_pipeline_respects_align_disabled(tmp_path, mocker):
    wav = _setup_mocks(mocker, tmp_path)
    mocker.patch("transcript.pipeline.align.is_available", return_value=True)
    spy = mocker.patch("transcript.pipeline.align.run")
    cfg = PipelineConfig(align=AlignConfig(enabled=False))
    pipeline.run(audio_path=wav, config=cfg, with_diarization=True)
    spy.assert_not_called()


def test_pipeline_cleans_up_temp_wav(tmp_path, mocker):
    wav = _setup_mocks(mocker, tmp_path)
    prepared = tmp_path / "prepared.wav"
    assert prepared.exists()
    pipeline.run(audio_path=wav, config=PipelineConfig(), with_diarization=True)
    assert not prepared.exists()


def test_pipeline_dispatches_to_sortformer_by_default(tmp_path, mocker):
    wav = _setup_mocks(mocker, tmp_path)
    diarizen_spy = mocker.patch("transcript.pipeline.diarize_diarizen.run")
    _, meta = pipeline.run(audio_path=wav, config=PipelineConfig(), with_diarization=True)
    diarizen_spy.assert_not_called()
    assert "Sortformer" in meta.diarizer


def test_pipeline_dispatches_to_diarizen_when_backend_configured(tmp_path, mocker):
    wav = _setup_mocks(mocker, tmp_path)
    diarizen_spy = mocker.patch(
        "transcript.pipeline.diarize_diarizen.run",
        return_value=[Turn("Speaker 1", 0.0, 1.5), Turn("Speaker 2", 1.5, 3.0)],
    )
    sortformer_spy = mocker.patch("transcript.pipeline.diarize.run")
    cfg = PipelineConfig(diarize=DiarizeConfig(backend="diarizen"))
    _, meta = pipeline.run(audio_path=wav, config=cfg, with_diarization=True)
    diarizen_spy.assert_called_once()
    sortformer_spy.assert_not_called()
    assert "DiariZen" in meta.diarizer


def test_pipeline_raises_keyerror_for_unknown_backend(tmp_path, mocker):
    wav = _setup_mocks(mocker, tmp_path)
    cfg = PipelineConfig(diarize=DiarizeConfig(backend="bogus"))  # type: ignore[arg-type]
    with pytest.raises(KeyError):
        pipeline.run(audio_path=wav, config=cfg, with_diarization=True)
