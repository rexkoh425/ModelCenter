"""Process a video with Depth Anything V2 and export a depth-map video.

This script reads an input video, estimates per-frame depth using the
`LiheYoung/depth-anything-v2-large-hf` model (via the 🤗 `transformers`
pipeline), and writes the resulting depth maps to an output video file.

Usage:
    # Edit DepthAnything/depth_anything_v2_video_config.yaml, then run:
    python depth_anything_v2_video.py

Dependencies (install beforehand):
    pip install torch torchvision transformers pillow opencv-python numpy

The script will automatically download the model weights on the first run.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline
import yaml


DEFAULT_MODEL_ID = "LiheYoung/depth-anything-v2-large-hf"


COLORMAP_LOOKUP = {
    "inferno": cv2.COLORMAP_INFERNO,
    "turbo": cv2.COLORMAP_TURBO,
    "magma": cv2.COLORMAP_MAGMA,
    "plasma": cv2.COLORMAP_PLASMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "cividis": cv2.COLORMAP_CIVIDIS,
    "jet": cv2.COLORMAP_JET,
    "gray": None,
}


DEFAULT_CONFIG_FILENAME = "depth_anything_v2_video_config.yaml"
DEFAULT_CONFIG_PATH = Path(__file__).with_name(DEFAULT_CONFIG_FILENAME)


@dataclass(slots=True)
class DepthVideoConfig:
    input_video: Path
    output: Optional[Path] = None
    model: str = DEFAULT_MODEL_ID
    device: str = "auto"
    fps: Optional[float] = None
    colormap: str = "inferno"
    min_percentile: float = 2.0
    max_percentile: float = 98.0
    side_by_side: bool = False
    max_frames: Optional[int] = None


def load_config(config_path: Path) -> DepthVideoConfig:
    expanded = Path(config_path).expanduser()
    if not expanded.is_file():
        raise FileNotFoundError(f"Config file not found: {expanded}")

    with expanded.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")

    def require_path(key: str) -> Path:
        value = data.get(key)
        if value is None:
            raise ValueError(f"Config field '{key}' is required.")
        return Path(str(value)).expanduser()

    def optional_path(key: str) -> Optional[Path]:
        value = data.get(key)
        if value is None:
            return None
        return Path(str(value)).expanduser()

    def coerce_float(key: str, default: Optional[float] = None) -> Optional[float]:
        value = data.get(key, default)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Config field '{key}' must be a float.") from exc

    def coerce_bool(key: str, default: bool = False) -> bool:
        value = data.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y", "on"}:
                return True
            if lowered in {"false", "0", "no", "n", "off"}:
                return False
        return bool(value)

    def coerce_int(key: str) -> Optional[int]:
        value = data.get(key)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Config field '{key}' must be an integer.") from exc

    colormap = str(data.get("colormap", "inferno"))
    if colormap not in COLORMAP_LOOKUP:
        raise ValueError(
            f"Unsupported colormap '{colormap}'. Expected one of: {', '.join(COLORMAP_LOOKUP.keys())}."
        )

    min_percentile = float(data.get("min_percentile", 2.0))
    max_percentile = float(data.get("max_percentile", 98.0))
    if max_percentile <= min_percentile:
        raise ValueError("max_percentile must be greater than min_percentile.")

    return DepthVideoConfig(
        input_video=require_path("input_video"),
        output=optional_path("output"),
        model=str(data.get("model", DEFAULT_MODEL_ID)),
        device=str(data.get("device", "auto")),
        fps=coerce_float("fps"),
        colormap=colormap,
        min_percentile=min_percentile,
        max_percentile=max_percentile,
        side_by_side=coerce_bool("side_by_side", False),
        max_frames=coerce_int("max_frames"),
    )


def resolve_device(requested: str) -> int | str:
    if requested == "auto":
        if torch.cuda.is_available():
            return 0
        if torch.backends.mps.is_available():  # pragma: no cover - Mac specific
            return "mps"
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but not available.")
        return 0
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS device requested but not available.")
        return "mps"
    return "cpu"


def create_video_writer(
    output_path: Path, width: int, height: int, fps: float
) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer at {output_path}.")
    return writer


def normalise_depth(
    depth_map: np.ndarray,
    min_percentile: float,
    max_percentile: float,
) -> np.ndarray:
    safe_depth = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
    lo = np.percentile(safe_depth, min_percentile)
    hi = np.percentile(safe_depth, max_percentile)
    if hi <= lo:
        hi = safe_depth.max()
        lo = safe_depth.min()
    if hi <= lo:
        hi = lo + 1.0
    clipped = np.clip(safe_depth, lo, hi)
    norm = (clipped - lo) / (hi - lo)
    return np.clip(norm * 255.0, 0, 255).astype(np.uint8)


def depth_to_visual(
    depth_map: np.ndarray,
    colormap_name: str,
) -> np.ndarray:
    colormap_id = COLORMAP_LOOKUP[colormap_name]
    if colormap_id is None:
        return cv2.merge([depth_map, depth_map, depth_map])
    return cv2.applyColorMap(depth_map, colormap_id)


def infer_depth(
    estimator,
    frame_bgr: np.ndarray,
) -> np.ndarray:
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    result = estimator(pil_image)
    depth_pil = result["depth"]
    depth_np = np.array(depth_pil, dtype=np.float32)
    return depth_np


def process_video(config: DepthVideoConfig) -> Path:
    input_path = config.input_video
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    output_path = config.output if config.output else input_path.with_name(
        f"{input_path.stem}_depth.mp4"
    )

    device = resolve_device(config.device)
    dtype = torch.float16 if device != "cpu" else torch.float32
    estimator = pipeline(
        task="depth-estimation",
        model=config.model,
        device=device,
        torch_dtype=dtype,
    )

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    fps_out = config.fps if config.fps else (fps if fps > 0 else 24.0)

    output_width = width * 2 if config.side_by_side else width
    output_height = height
    writer = create_video_writer(output_path, output_width, output_height, fps_out)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = (
        min(frame_count, config.max_frames)
        if config.max_frames and frame_count > 0
        else (config.max_frames or frame_count)
    )

    processed = 0
    try:
        while True:
            if config.max_frames is not None and processed >= config.max_frames:
                break

            success, frame = capture.read()
            if not success:
                break

            depth_map = infer_depth(estimator, frame)
            depth_map_resized = cv2.resize(
                depth_map,
                (width, height),
                interpolation=cv2.INTER_CUBIC,
            )
            depth_8bit = normalise_depth(
                depth_map_resized,
                config.min_percentile,
                config.max_percentile,
            )
            depth_visual = depth_to_visual(depth_8bit, config.colormap)

            if config.side_by_side:
                composite = np.hstack((frame, depth_visual))
                writer.write(composite)
            else:
                writer.write(depth_visual)

            processed += 1
            if processed % 10 == 0:
                total_display = total_frames if total_frames else "?"
                print(f"Processed {processed}/{total_display} frames", end="\r", file=sys.stderr)
    finally:
        capture.release()
        writer.release()

    print(f"\nDepth video saved to: {output_path}")
    return output_path


class DepthAnythingVideoApp:
    """Class wrapper to load YAML config and run the depth processor."""

    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
        self.config: Optional[DepthVideoConfig] = None

    def load(self) -> None:
        self.config = load_config(self.config_path)

    def run(self) -> Path:
        if self.config is None:
            self.load()
        assert self.config is not None
        return process_video(self.config)


def main() -> None:
    app = DepthAnythingVideoApp()
    try:
        app.run()
    except KeyboardInterrupt:  # pragma: no cover - user interrupt
        print("\nInterrupted by user.")
    except Exception as exc:  # pragma: no cover - runtime errors
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
