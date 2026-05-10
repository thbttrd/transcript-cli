from transcript import config


def test_data_dir_under_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert config.data_dir() == tmp_path / ".local" / "share" / "transcript"


def test_whisper_binary_path(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert config.whisper_binary() == (
        tmp_path / ".local" / "share" / "transcript" / "whisper.cpp" / "build" / "bin" / "whisper-cli"
    )


def test_whisper_model_path(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert config.whisper_model("large-v3") == (
        tmp_path / ".local" / "share" / "transcript" / "models" / "ggml-large-v3.bin"
    )
