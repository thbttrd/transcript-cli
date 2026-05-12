import json
from pathlib import Path

import pytest

from transcript import config, formatters, pipeline
from transcript.pipeline_config import (
    DiarizeConfig,
    PipelineConfig,
    TranscribeConfig,
)

FIXTURE = Path(__file__).parent / "fixtures" / "tiny.wav"


def _setup_ready() -> tuple[bool, str]:
    if not FIXTURE.exists():
        return False, "tiny.wav fixture not generated (run scripts/generate_tiny_wav.sh)"
    if not config.whisper_binary().exists():
        return False, "whisper.cpp binary not built (run scripts/install.sh)"
    base_model = config.whisper_model("base")
    if not base_model.exists():
        return False, (
            f"whisper 'base' model missing at {base_model} "
            "(download via whisper.cpp's download-ggml-model.sh)"
        )
    return True, ""


@pytest.mark.integration
def test_full_pipeline_against_tiny_wav():
    ok, reason = _setup_ready()
    if not ok:
        pytest.skip(reason)

    cfg = PipelineConfig(
        transcribe=TranscribeConfig(model="base", language="fr"),
        diarize=DiarizeConfig(num_speakers=2),
    )
    utterances, meta = pipeline.run(
        audio_path=FIXTURE,
        config=cfg,
        with_diarization=True,
    )
    data = json.loads(formatters.get("json")(utterances, meta))
    # Two French voices — Sortformer should detect 2 speakers
    assert data["meta"]["speaker_count"] == 2
    # Some recognisable French content should be present
    text = " ".join(u["text"] for u in data["utterances"]).lower()
    assert any(word in text for word in ("bonjour", "merci", "très", "bien"))


@pytest.mark.integration
def test_no_diarize_returns_single_speaker():
    ok, reason = _setup_ready()
    if not ok:
        pytest.skip(reason)

    cfg = PipelineConfig(transcribe=TranscribeConfig(model="base", language="fr"))
    utterances, meta = pipeline.run(
        audio_path=FIXTURE,
        config=cfg,
        with_diarization=False,
    )
    data = json.loads(formatters.get("json")(utterances, meta))
    speakers = {u["speaker"] for u in data["utterances"]}
    assert speakers == {"Speaker 1"}
