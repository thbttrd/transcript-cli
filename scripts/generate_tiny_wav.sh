#!/usr/bin/env bash
# Generates an 8-second test fixture with two macOS voices alternating.
# Run from project root: bash scripts/generate_tiny_wav.sh
set -euo pipefail

OUT="tests/fixtures/tiny.wav"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Two distinct voices — Thomas (fr) and Amelie (fr) — alternating short phrases
say -v "Thomas"  -o "$TMP/a.aiff" "Bonjour, comment allez-vous aujourd'hui?"
say -v "Amelie"  -o "$TMP/b.aiff" "Très bien merci, et vous?"
say -v "Thomas"  -o "$TMP/c.aiff" "Je vais bien aussi."
say -v "Amelie"  -o "$TMP/d.aiff" "Parfait."

# Concatenate with short gaps and convert to 16 kHz mono WAV
ffmpeg -y \
  -i "$TMP/a.aiff" -i "$TMP/b.aiff" -i "$TMP/c.aiff" -i "$TMP/d.aiff" \
  -filter_complex "[0:a][1:a][2:a][3:a]concat=n=4:v=0:a=1[a]" \
  -map "[a]" -ar 16000 -ac 1 -c:a pcm_s16le \
  "$OUT"

echo "Wrote $OUT ($(du -h "$OUT" | cut -f1))"
