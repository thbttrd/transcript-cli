#!/usr/bin/env python
"""Dump raw whisper words + raw Sortformer turns side-by-side for debugging.

Usage:
    ~/.local/share/uv/tools/transcript-app/bin/python scripts/dump_pipeline.py \
        "/path/to/audio.m4a" [--speakers N]
"""
import argparse
import sys

from transcript import audio, diarize, transcribe


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("audio_file")
    p.add_argument("--speakers", type=int, default=None)
    p.add_argument("-l", "--language", default=None)
    p.add_argument("-m", "--model", default="large-v3")
    args = p.parse_args()

    from pathlib import Path
    wav, dur = audio.prepare(Path(args.audio_file))
    print(f"# audio: {args.audio_file}  ({dur:.2f}s, wav={wav})\n")

    print("## raw NeMo Sortformer turns")
    turns = diarize.run(wav, num_speakers=args.speakers)
    for t in turns:
        print(f"  {t.speaker:9s}  {t.start:6.2f} → {t.end:6.2f}  ({t.end - t.start:.2f}s)")

    print("\n## raw whisper words")
    words = transcribe.run(wav, model=args.model, language=args.language)
    for w in words:
        print(f"  {w.start:6.2f} → {w.end:6.2f}  '{w.text}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
