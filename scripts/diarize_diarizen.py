#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     # DiariZen ships a *bundled fork* of pyannote.audio at version 3.1.1 with
#     # private patches (config=, seg_duration=, device= kwargs on SpeakerDiarization).
#     # The fork lives at DiariZen/pyannote-audio/. Installing stock PyPI pyannote
#     # silently breaks DiariZen's super().__init__ call. Don't pin a stock version
#     # here — let the fork install via [tool.uv.sources."pyannote.audio"] below.
#     "pyannote.audio",
#     # The fork's requirements.txt pins these transitive pyannote packages exactly.
#     # They're incompatible with the pyannote 4.x line (which needs core==6.x etc).
#     "pyannote.core==5.0.0",
#     "pyannote.database==5.1.3",
#     "pyannote.metrics==3.2.1",
#     "pyannote.pipeline==3.0.1",
#     # Fork requires torch>=2.0.0. Stay at 2.5.x — torch 2.6 changed the default
#     # of `torch.load(weights_only=...)` from False to True, which breaks the fork's
#     # checkpoint loader (uses an older lightning_fabric that doesn't pass it).
#     "torch==2.5.1",
#     "torchaudio==2.5.1",
#     # Fork uses `np.NaN` (removed in numpy 2.0). Pin <2 to match main-project numpy.
#     "numpy==1.26.4",
#     "diarizen",
#     # DiariZen's pyproject.toml omits runtime deps — pulled from their requirements.txt.
#     "toml==0.10.2",
#     "einops==0.8.1",
#     "librosa==0.10.2.post1",
#     "soundfile==0.12.1",
#     "pyyaml==6.0.2",
#     "accelerate==1.6.0",
#     "tqdm==4.67.1",
# ]
#
# [tool.uv.sources.diarizen]
# git = "https://github.com/BUTSpeechFIT/DiariZen.git"
# rev = "d52b8d5e3d96632b1a8a0dc34762bf811471e441"
#
# [tool.uv.sources."pyannote.audio"]
# git = "https://github.com/BUTSpeechFIT/DiariZen.git"
# rev = "d52b8d5e3d96632b1a8a0dc34762bf811471e441"
# subdirectory = "pyannote-audio"
# ///
"""Isolated DiariZen runner — invoked as a subprocess from the main project.

The main project pins torch==2.6.0 (NeMo compat); DiariZen needs pyannote.audio 4.x
which in turn needs torch>=2.7. We can't satisfy both in one env, so this script
declares its own env via PEP 723 inline-deps and uv caches it globally.

I/O contract (parent: src/transcript/diarize_diarizen.py):
  argv[1]: absolute path to a wav file
  stdout:  one line of JSON — a list of {speaker, start, end} dicts (NOTHING else)
  stderr:  everything else (DiariZen progress prints, model-load chatter, errors)
"""
import contextlib
import json
import sys

from diarizen.pipelines.inference import DiariZenPipeline

_HF_MODEL = "BUT-FIT/diarizen-wavlm-large-s80-md"


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <wav_path>", file=sys.stderr)
        return 2

    wav_path = sys.argv[1]

    # DiariZen scatters progress prints to stdout; redirect to stderr so our
    # final stdout payload stays parseable as a single JSON line.
    with contextlib.redirect_stdout(sys.stderr):
        pipeline = DiariZenPipeline.from_pretrained(_HF_MODEL)
        annotation = pipeline(wav_path)

    turns = [
        {"speaker": str(speaker), "start": float(seg.start), "end": float(seg.end)}
        for seg, _, speaker in annotation.itertracks(yield_label=True)
    ]
    json.dump(turns, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
