from pathlib import Path

import pytest

from transcript import config


def test_data_dir_under_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert config.data_dir() == tmp_path / ".local" / "share" / "transcript"


def test_whisper_binary_path(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert config.whisper_binary() == tmp_path / ".local" / "share" / "transcript" / "whisper.cpp" / "main"


def test_whisper_model_path(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert config.whisper_model("large-v3") == (
        tmp_path / ".local" / "share" / "transcript" / "models" / "ggml-large-v3.bin"
    )


def test_hf_token_env_var_takes_precedence(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "from-env")
    assert config.hf_token() == "from-env"


def test_hf_token_falls_back_to_keyring(monkeypatch, mocker):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    mock_get = mocker.patch("transcript.config.keyring.get_password", return_value="from-keychain")
    assert config.hf_token() == "from-keychain"
    mock_get.assert_called_once_with("transcript", "huggingface")


def test_hf_token_missing_raises(monkeypatch, mocker):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    mocker.patch("transcript.config.keyring.get_password", return_value=None)
    with pytest.raises(config.MissingTokenError):
        config.hf_token()
