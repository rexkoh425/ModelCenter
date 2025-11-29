#!/usr/bin/env python3
"""
Simple FFmpeg-based MP4 upscaler to 4K (3840x2160) or any custom resolution
configured via YAML.

Example:
    # Use the default video_upscale_config.yaml next to this script
    python video_upscale.py

    # Override the config path and some CLI options
    python video_upscale.py input.mp4 --config custom_settings.yaml --width 2560

Dependencies:
    ffmpeg (installed separately and available on PATH)
    PyYAML (pip install pyyaml)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_WIDTH = 3840
DEFAULT_HEIGHT = 2160
DEFAULT_SCALE_FILTER = "lanczos"
DEFAULT_VIDEO_CODEC = "libx264"
DEFAULT_PRESET = "slow"
DEFAULT_CRF = 18
DEFAULT_AUDIO_CODEC = "copy"
DEFAULT_CONFIG_FILENAME = "video_upscale_config.yaml"
DEFAULT_CONFIG_PATH = Path(__file__).with_name(DEFAULT_CONFIG_FILENAME)

ConfigDict = Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upscale an MP4 video to 4K (or another resolution) using FFmpeg with optional YAML config overrides."
    )
    parser.add_argument(
        "input_video",
        nargs="?",
        type=Path,
        help="Path to the source MP4 video to upscale. Optional when provided in the YAML config.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Path to a YAML config file. Defaults to video_upscale_config.yaml next to this script when omitted."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the upscaled video. Defaults to <name>_<width>x<height>.mp4.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help=f"Target width in pixels (default: {DEFAULT_WIDTH}).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help=f"Target height in pixels (default: {DEFAULT_HEIGHT}).",
    )
    parser.add_argument(
        "--scale-filter",
        default=None,
        help=f"Resampling filter passed to FFmpeg scale filter (default: {DEFAULT_SCALE_FILTER}).",
    )
    parser.add_argument(
        "--video-codec",
        default=None,
        help=f"FFmpeg video codec to use (default: {DEFAULT_VIDEO_CODEC}).",
    )
    parser.add_argument(
        "--preset",
        default=None,
        help=f"FFmpeg encoder preset controlling speed/quality trade-off (default: {DEFAULT_PRESET}).",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=None,
        help=f"Constant Rate Factor for quality (smaller is better, default: {DEFAULT_CRF}).",
    )
    parser.add_argument(
        "--bitrate",
        default=None,
        help="Optional target video bitrate (e.g., 30M). Overrides CRF if provided.",
    )
    parser.add_argument(
        "--audio-codec",
        default=None,
        help=f"Audio codec to use (default: {DEFAULT_AUDIO_CODEC}).",
    )
    parser.add_argument(
        "--audio-bitrate",
        default=None,
        help="Optional target audio bitrate (e.g., 320k). Ignored when audio codec is 'copy'.",
    )

    overwrite_group = parser.add_mutually_exclusive_group()
    overwrite_group.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
        default=None,
    )
    overwrite_group.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Prevent overwriting even if the config enables it.",
    )

    dry_run_group = parser.add_mutually_exclusive_group()
    dry_run_group.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Print the FFmpeg command without running it.",
        default=None,
    )
    dry_run_group.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="Force execution even if the config is set to dry_run.",
    )

    return parser.parse_args()


def load_yaml_config(config_path: Path | None, *, require_exists: bool = False) -> ConfigDict:
    if config_path is None:
        return {}

    expanded = Path(config_path).expanduser()
    if not expanded.exists():
        if require_exists:
            raise FileNotFoundError(f"Config file not found: {expanded}")
        return {}

    with expanded.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping of option names to values.")

    return data


def merge_args_with_config(args: argparse.Namespace, config: ConfigDict) -> argparse.Namespace:
    merged = {
        "input_video": _resolve_path_option(args.input_video, config, "input_video"),
        "output": _resolve_path_option(args.output, config, "output"),
        "width": _resolve_int_option(args.width, config, "width", DEFAULT_WIDTH),
        "height": _resolve_int_option(args.height, config, "height", DEFAULT_HEIGHT),
        "scale_filter": _resolve_str_option(
            args.scale_filter, config, "scale_filter", DEFAULT_SCALE_FILTER
        ),
        "video_codec": _resolve_str_option(
            args.video_codec, config, "video_codec", DEFAULT_VIDEO_CODEC
        ),
        "preset": _resolve_str_option(args.preset, config, "preset", DEFAULT_PRESET),
        "crf": _resolve_int_option(args.crf, config, "crf", DEFAULT_CRF, allow_none=True),
        "bitrate": _resolve_str_option(
            args.bitrate, config, "bitrate", None, allow_none=True
        ),
        "audio_codec": _resolve_str_option(
            args.audio_codec, config, "audio_codec", DEFAULT_AUDIO_CODEC
        ),
        "audio_bitrate": _resolve_str_option(
            args.audio_bitrate, config, "audio_bitrate", None, allow_none=True
        ),
        "overwrite": _resolve_bool_option(args.overwrite, config, "overwrite", False),
        "dry_run": _resolve_bool_option(args.dry_run, config, "dry_run", False),
    }
    return argparse.Namespace(**merged)


def _resolve_path_option(
    cli_value: Path | None, config: ConfigDict, key: str
) -> Path | None:
    if cli_value is not None:
        return cli_value.expanduser()
    if key in config:
        value = config[key]
        if value in {None, ""}:
            return None
        return Path(str(value)).expanduser()
    return None


def _resolve_int_option(
    cli_value: int | None,
    config: ConfigDict,
    key: str,
    default: int | None,
    *,
    allow_none: bool = False,
) -> int | None:
    if cli_value is not None:
        return cli_value
    if key in config:
        value = config[key]
        if value is None:
            return None if allow_none else default
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Config field '{key}' must be an integer.") from exc
    return default


def _resolve_str_option(
    cli_value: str | None,
    config: ConfigDict,
    key: str,
    default: str | None,
    *,
    allow_none: bool = False,
) -> str | None:
    if cli_value is not None:
        return cli_value
    if key in config:
        value = config[key]
        if value is None:
            return None if allow_none else default
        return str(value)
    return default


def _resolve_bool_option(
    cli_value: bool | None, config: ConfigDict, key: str, default: bool
) -> bool:
    if cli_value is not None:
        return cli_value
    if key in config:
        value = config[key]
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y"}:
                return True
            if lowered in {"false", "0", "no", "n"}:
                return False
        raise ValueError(
            f"Config field '{key}' must be a boolean (true/false, yes/no, 1/0)."
        )
    return default


def ensure_ffmpeg_exists() -> None:
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg executable not found. Please install FFmpeg and ensure it is on PATH.")


def build_command(args: argparse.Namespace, output_path: Path) -> list[str]:
    command = ["ffmpeg"]
    command.append("-y" if args.overwrite else "-n")
    command += ["-i", str(args.input_video)]

    scale_expr = f"scale={args.width}:{args.height}:flags={args.scale_filter}"
    command += ["-vf", scale_expr, "-c:v", args.video_codec, "-preset", args.preset]

    if args.bitrate:
        command += ["-b:v", args.bitrate]
    elif args.crf is not None:
        command += ["-crf", str(args.crf)]

    command += ["-c:a", args.audio_codec]
    if args.audio_codec != "copy" and args.audio_bitrate:
        command += ["-b:a", args.audio_bitrate]

    command.append(str(output_path))
    return command


def main() -> None:
    args = parse_args()
    config_path = args.config or DEFAULT_CONFIG_PATH
    config = load_yaml_config(config_path, require_exists=args.config is not None)
    merged_args = merge_args_with_config(args, config)

    if merged_args.input_video is None:
        sys.exit(
            "Input video must be provided either as a positional argument or in the YAML config under 'input_video'."
        )

    ensure_ffmpeg_exists()

    if not merged_args.input_video.exists():
        sys.exit(f"Input video not found: {merged_args.input_video}")

    output_path = merged_args.output
    if output_path is None:
        suffix = f"_{merged_args.width}x{merged_args.height}"
        output_path = merged_args.input_video.with_name(
            f"{merged_args.input_video.stem}{suffix}{merged_args.input_video.suffix}"
        )

    ffmpeg_cmd = build_command(merged_args, output_path)

    if merged_args.dry_run:
        print(" ".join(ffmpeg_cmd))
        return

    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Upscaled video saved to: {output_path}")
    except subprocess.CalledProcessError as exc:
        sys.exit(
            f"FFmpeg failed with exit code {exc.returncode}. Check the console output for details."
        )


if __name__ == "__main__":
    main()
