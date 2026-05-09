import argparse
import sys
from pathlib import Path

from transcript import __version__, doctor, pipeline
from transcript.audio import AudioError
from transcript.config import MissingTokenError
from transcript.diarize import DiarizeError
from transcript.progress import Progress
from transcript.transcribe import TranscribeError

EXIT_OK = 0
EXIT_ERR = 1
EXIT_USAGE = 2
EXIT_AUDIO = 10
EXIT_SETUP = 11
EXIT_AUTH = 12
EXIT_RESOURCE = 13


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="transcript",
        description="Local voice-memo transcription with speaker diarization.",
    )
    p.add_argument(
        "audio_file",
        nargs="?",
        type=Path,
        help="Audio file (.m4a, .mp3, .wav, .mp4, …)",
    )
    p.add_argument("-o", "--output", type=Path, help="Write transcript to file (default: stdout)")
    p.add_argument(
        "-f", "--format",
        choices=["md", "json", "srt", "txt"],
        default="md",
        help="Output format (default: md)",
    )
    p.add_argument("--no-timestamps", action="store_true", help="Omit [mm:ss] markers in markdown")
    p.add_argument("-l", "--language", default=None, help="Language code (default: auto-detect)")
    p.add_argument(
        "-m", "--model",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        default="large-v3",
        help="Whisper model (default: large-v3)",
    )
    p.add_argument("--no-diarize", action="store_true", help="Skip speaker labelling")
    p.add_argument("--speakers", type=int, default=None, help="Fix speaker count when known")
    p.add_argument("-v", "--verbose", action="store_true", help="Show step-by-step progress")
    p.add_argument("-q", "--quiet", action="store_true", help="Suppress all progress")
    p.add_argument("--version", action="version", version=f"transcript {__version__}")
    p.add_argument("--doctor", action="store_true", help="Check setup and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.doctor:
        code, report = doctor.check()
        print(report)
        return code

    if args.audio_file is None:
        parser.print_usage(sys.stderr)
        return EXIT_USAGE

    progress = Progress(verbose=args.verbose, quiet=args.quiet)

    try:
        out = pipeline.run(
            audio_path=args.audio_file,
            model=args.model,
            language=args.language,
            diarize=not args.no_diarize,
            num_speakers=args.speakers,
            format_name=args.format,
            with_timestamps=not args.no_timestamps,
            progress=progress,
        )
    except AudioError as e:
        print(f"✗ {e}", file=sys.stderr)
        return EXIT_AUDIO
    except TranscribeError as e:
        print(f"✗ {e}", file=sys.stderr)
        return EXIT_SETUP
    except (DiarizeError, MissingTokenError) as e:
        print(f"✗ {e}", file=sys.stderr)
        return EXIT_AUTH
    except Exception as e:  # last-resort
        if args.verbose:
            raise
        print(f"✗ unexpected error: {e}", file=sys.stderr)
        return EXIT_ERR

    if args.output:
        args.output.write_text(out)
    else:
        sys.stdout.write(out)
    return EXIT_OK
