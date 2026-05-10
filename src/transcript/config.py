from pathlib import Path


def data_dir() -> Path:
    return Path.home() / ".local" / "share" / "transcript"


def whisper_dir() -> Path:
    return data_dir() / "whisper.cpp"


def whisper_binary() -> Path:
    # Modern whisper.cpp ships the CLI at build/bin/whisper-cli; the legacy
    # `main` target is a deprecation-warning stub.
    return whisper_dir() / "build" / "bin" / "whisper-cli"


def models_dir() -> Path:
    return data_dir() / "models"


def whisper_model(name: str) -> Path:
    return models_dir() / f"ggml-{name}.bin"


def whisper_coreml_encoder(name: str) -> Path:
    return models_dir() / f"ggml-{name}-encoder.mlmodelc"
