#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="$HOME/.local/share/transcript"
WHISPER_DIR="$DATA_DIR/whisper.cpp"
MODELS_DIR="$DATA_DIR/models"
MODEL="large-v3"

# 1. Sanity checks
[[ "$(uname -s)" == "Darwin" ]] || { echo "macOS only"; exit 1; }
[[ "$(uname -m)" == "arm64" ]] || echo "⚠️  Not Apple Silicon — Metal/ANE unavailable"

# 2. System deps via Homebrew (skipped if present)
command -v brew    >/dev/null || { echo "Install Homebrew first: https://brew.sh"; exit 1; }
command -v ffmpeg  >/dev/null || brew install ffmpeg
command -v cmake   >/dev/null || brew install cmake
command -v uv      >/dev/null || brew install uv

# 3. whisper.cpp — clone + build with CoreML + Metal
mkdir -p "$DATA_DIR"
[[ -d "$WHISPER_DIR" ]] || git clone https://github.com/ggerganov/whisper.cpp "$WHISPER_DIR"
( cd "$WHISPER_DIR" && git pull --ff-only )
( cd "$WHISPER_DIR" && WHISPER_COREML=1 WHISPER_COREML_ALLOW_FALLBACK=1 make -j )

# 4. Models — GGML weights + CoreML encoder
mkdir -p "$MODELS_DIR"
if [[ ! -f "$MODELS_DIR/ggml-${MODEL}.bin" ]]; then
  ( cd "$WHISPER_DIR" && bash ./models/download-ggml-model.sh "$MODEL" )
  mv "$WHISPER_DIR/models/ggml-${MODEL}.bin" "$MODELS_DIR/"
fi
if [[ ! -d "$MODELS_DIR/ggml-${MODEL}-encoder.mlmodelc" ]]; then
  ( cd "$WHISPER_DIR" && bash ./models/generate-coreml-model.sh "$MODEL" )
  mv "$WHISPER_DIR/models/ggml-${MODEL}-encoder.mlmodelc" "$MODELS_DIR/"
fi

# 5. HuggingFace token → Keychain
if ! security find-generic-password -s transcript -a huggingface >/dev/null 2>&1; then
  echo
  echo "→ Need a HuggingFace token for pyannote (free)."
  echo "  1. Create one at https://huggingface.co/settings/tokens"
  echo "  2. Accept license at https://huggingface.co/pyannote/speaker-diarization-3.1"
  echo "  3. Accept license at https://huggingface.co/pyannote/segmentation-3.0"
  echo
  read -rsp "Paste your HF token (input hidden): " HF_TOKEN; echo
  security add-generic-password -s transcript -a huggingface -w "$HF_TOKEN"
fi

# 6. Install the CLI globally — clone repo if not invoked from inside it
if [[ -f "$(pwd)/pyproject.toml" ]] && grep -q '^name = "transcript-app"' "$(pwd)/pyproject.toml" 2>/dev/null; then
  REPO_DIR="$(pwd)"
else
  REPO_DIR="$DATA_DIR/transcript-app"
  [[ -d "$REPO_DIR" ]] || git clone https://github.com/<you>/transcript-app "$REPO_DIR"
  ( cd "$REPO_DIR" && git pull --ff-only )
fi
uv tool install --from "$REPO_DIR" transcript-app

# 7. Smoke check
echo
transcript --doctor || true   # report-only; doctor exit code != 0 is informative, not fatal here
echo
echo "✅  Done. Try: transcript path/to/voice-memo.m4a"
