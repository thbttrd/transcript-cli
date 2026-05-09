import io

from transcript.progress import Progress


def test_progress_silent_when_quiet():
    buf = io.StringIO()
    p = Progress(verbose=False, quiet=True, stream=buf)
    p.step("doing thing")
    assert buf.getvalue() == ""


def test_progress_compact_when_default():
    buf = io.StringIO()
    p = Progress(verbose=False, quiet=False, stream=buf)
    p.step("preparing audio")
    p.step("transcribing")
    out = buf.getvalue()
    assert "preparing audio" in out
    assert "transcribing" in out


def test_progress_verbose_includes_timing():
    buf = io.StringIO()
    p = Progress(verbose=True, quiet=False, stream=buf)
    p.step("preparing audio")
    p.done("preparing audio")
    out = buf.getvalue()
    assert "preparing audio" in out
    # Verbose mode should print "ok" or similar marker on done()
    assert "ok" in out.lower() or "✓" in out
