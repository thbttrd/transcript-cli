#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="$HOME/.local/share/transcript"
WHISPER_DIR="$DATA_DIR/whisper.cpp"
MODELS_DIR="$DATA_DIR/models"
COREML_VENV="$DATA_DIR/coreml-venv"
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
  # CoreML conversion needs torch + coremltools + openai-whisper + ane_transformers,
  # pinned to versions that disagree with the runtime's torch<2.5. Keep them in an
  # isolated, install-time-only venv so they don't leak into the user's Python env.
  [[ -d "$COREML_VENV" ]] || uv venv --python 3.11 "$COREML_VENV"
  uv pip install --python "$COREML_VENV/bin/python" \
    -r "$WHISPER_DIR/models/requirements-coreml.txt"
  ( cd "$WHISPER_DIR" \
    && PATH="$COREML_VENV/bin:$PATH" bash ./models/generate-coreml-model.sh "$MODEL" )
  mv "$WHISPER_DIR/models/ggml-${MODEL}-encoder.mlmodelc" "$MODELS_DIR/"
fi

# 5. Install the CLI globally — clone repo if not invoked from inside it
if [[ -f "$(pwd)/pyproject.toml" ]] && grep -q '^name = "transcript-app"' "$(pwd)/pyproject.toml" 2>/dev/null; then
  REPO_DIR="$(pwd)"
else
  REPO_DIR="$DATA_DIR/transcript-app"
  [[ -d "$REPO_DIR" ]] || git clone https://github.com/<you>/transcript-app "$REPO_DIR"
  ( cd "$REPO_DIR" && git pull --ff-only )
fi
# --python 3.11 is mandatory: torch 2.4.1 only ships cp310/cp311/cp312 wheels,
# and older uv versions can pick 3.13/3.14 even when requires-python forbids it.
# --force re-installs the tool; --reinstall forces a fresh wheel build from source
# (without it, uv reuses the cached 0.1.0 wheel and source-only changes don't ship).
uv tool install --python 3.11 --force --reinstall --from "$REPO_DIR" transcript-app

# 6. Smoke check
echo
transcript --doctor || true   # report-only; doctor exit code != 0 is informative, not fatal here
echo
echo "✅  Done. Try: transcript path/to/voice-memo.m4a"
