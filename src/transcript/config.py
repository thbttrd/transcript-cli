import os
from pathlib import Path

import keyring


class MissingTokenError(RuntimeError):
    """Raised when no HuggingFace token is available."""


_KEYRING_SERVICE = "transcript"
_KEYRING_USER = "huggingface"


def data_dir() -> Path:
    return Path.home() / ".local" / "share" / "transcript"


def whisper_dir() -> Path:
    return data_dir() / "whisper.cpp"


def whisper_binary() -> Path:
    return whisper_dir() / "main"


def models_dir() -> Path:
    return data_dir() / "models"


def whisper_model(name: str) -> Path:
    return models_dir() / f"ggml-{name}.bin"


def whisper_coreml_encoder(name: str) -> Path:
    return models_dir() / f"ggml-{name}-encoder.mlmodelc"


def hf_token() -> str:
    """Return the HuggingFace token. Env var wins; Keychain is fallback."""
    if env := os.environ.get("HF_TOKEN"):
        return env
    if kc := keyring.get_password(_KEYRING_SERVICE, _KEYRING_USER):
        return kc
    raise MissingTokenError(
        "No HuggingFace token found. Set $HF_TOKEN or run scripts/install.sh."
    )
