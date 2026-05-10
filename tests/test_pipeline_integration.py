import json
import shutil
import sys
from pathlib import Path

import pytest

from transcript import config, pipeline

FIXTURE = Path(__file__).parent / "fixtures" / "tiny.wav"


def _setup_ready() -> tuple[bool, str]:
    if not FIXTURE.exists():
        return False, "tiny.wav fixture not generated (run scripts/generate_tiny_wav.sh)"
    if not config.whisper_binary().exists():
        return False, "whisper.cpp binary not built (run scripts/install.sh)"
    base_model = config.whisper_model("base")
    if not base_model.exists():
        return False, f"whisper 'base' model missing at {base_model} (download via whisper.cpp's download-ggml-model.sh)"
    return True, ""


@pytest.mark.integration
def test_full_pipeline_against_tiny_wav():
    ok, reason = _setup_ready()
    if not ok:
        pytest.skip(reason)

    out = pipeline.run(
        audio_path=FIXTURE,
        model="base",            # use the small model to keep this test fast
        language="fr",
        with_diarization=True,
        num_speakers=2,
        format_name="json",
        with_timestamps=True,
    )
    data = json.loads(out)
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

    out = pipeline.run(
        audio_path=FIXTURE,
        model="base",
        language="fr",
        with_diarization=False,
        num_speakers=None,
        format_name="json",
        with_timestamps=True,
    )
    data = json.loads(out)
    speakers = {u["speaker"] for u in data["utterances"]}
    assert speakers == {"Speaker 1"}
