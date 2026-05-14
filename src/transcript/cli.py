import argparse
import sys
from pathlib import Path

from transcript import __version__, doctor, pipeline
from transcript.audio import AudioError
from transcript.diarize import DiarizeError
from transcript.progress import Progress
from transcript.transcribe import TranscribeError

EXIT_OK = 0
EXIT_ERR = 1
EXIT_USAGE = 2
EXIT_AUDIO = 10
EXIT_SETUP = 11
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
    p.add_argument(
        "--diarizer",
        choices=["sortformer", "diarizen"],
        default="sortformer",
        help=(
            "Diarization backend (default: sortformer / NeMo Streaming Sortformer 4spk-v2.1; "
            "diarizen = BUT-FIT WavLM-Large s80-md, requires --extra diarizen)."
        ),
    )
    p.add_argument("--speakers", type=int, default=None, help="Fix speaker count when known")
    p.add_argument(
        "--no-align",
        action="store_true",
        help="Skip forced word-alignment. On by default — ctc-forced-aligner + MMS-300m refines whisper.cpp word timestamps to sub-100 ms.",
    )
    p.add_argument(
        "--llm-fix",
        action="store_true",
        help="Opt in to local-LLM speaker-label cleanup via Ollama (gemma4:e4b). Off by default — alignment usually obsoletes it.",
    )
    p.add_argument(
        "--whisper-fallback",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override TranscribeConfig.no_fallback. --whisper-fallback lets Whisper retry "
            "low-confidence segments at higher temperature (no_fallback=False); "
            "--no-whisper-fallback disables retries (no_fallback=True). "
            "Default: track the pipeline_config default."
        ),
    )
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
        from transcript import formatters
        from transcript.pipeline_config import (
            AlignConfig,
            DiarizeConfig,
            LLMFixConfig,
            PipelineConfig,
            TranscribeConfig,
        )

        tx_kwargs = {"model": args.model, "language": args.language}
        if args.whisper_fallback is not None:
            tx_kwargs["no_fallback"] = not args.whisper_fallback
        cfg = PipelineConfig(
            transcribe=TranscribeConfig(**tx_kwargs),
            diarize=DiarizeConfig(num_speakers=args.speakers, backend=args.diarizer),
            align=AlignConfig(enabled=not args.no_align),
            llm_fix=LLMFixConfig(enabled=args.llm_fix),
        )
        utterances, meta = pipeline.run(
            audio_path=args.audio_file,
            config=cfg,
            with_diarization=not args.no_diarize,
            progress=progress,
        )
        render = formatters.get(args.format)
        if args.format == "md":
            out = render(utterances, meta, with_timestamps=not args.no_timestamps)
        else:
            out = render(utterances, meta)
    except AudioError as e:
        print(f"✗ {e}", file=sys.stderr)
        return EXIT_AUDIO
    except TranscribeError as e:
        print(f"✗ {e}", file=sys.stderr)
        return EXIT_SETUP
    except DiarizeError as e:
        print(f"✗ {e}", file=sys.stderr)
        return EXIT_SETUP
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
