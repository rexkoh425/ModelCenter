"""Panoptic lane segmentation on near-first-person driving footage with Mask2Former.

This tool runs the `facebook/mask2former-swin-large-cityscapes-panoptic` model
on every frame of an input video, extracts road- and lane-related segments, and
writes back an annotated video that highlights the drivable area from a
near-first-person viewpoint (e.g. dashcam footage).

Usage:
    1. Update `Mask2Former/mask2former_lane_segmenter_config.yaml`.
    2. Run `python mask2former_lane_segmenter.py`.

Dependencies:
    pip install torch torchvision transformers pillow opencv-python numpy pyyaml
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


DEFAULT_MODEL_ID = "facebook/mask2former-swin-large-cityscapes-panoptic"
DEFAULT_CONFIG_FILENAME = "mask2former_lane_segmenter_config.yaml"
DEFAULT_CONFIG_PATH = Path(__file__).with_name(DEFAULT_CONFIG_FILENAME)

# Road-related labels that tend to correspond to vehicle lanes
# in first-person driving datasets such as Cityscapes.
DEFAULT_LANE_LABELS = (
    "road",
    "lane-marking",  # some checkpoints include this finer-grained class
    "ground",
    "parking",
)


@dataclass(slots=True)
class Mask2FormerLaneConfig:
    input_video: Path
    output: Optional[Path] = None
    model: str = DEFAULT_MODEL_ID
    device: str = "auto"
    fps: Optional[float] = None
    highlight_labels: Sequence[str] = field(default_factory=lambda: DEFAULT_LANE_LABELS)
    overlay_color: Tuple[int, int, int] = (64, 255, 112)  # BGR for cv2
    overlay_alpha: float = 0.6
    min_score: float = 0.4
    max_long_edge: Optional[int] = 1280
    dilate_kernel: int = 5
    side_by_side: bool = True
    output_mask_only: bool = False
    max_frames: Optional[int] = None


def resolve_device(preference: str) -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        return torch.device(preference)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported device specifier: {preference}") from exc


def load_config(config_path: Path) -> Mask2FormerLaneConfig:
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

    def optional_float(key: str) -> Optional[float]:
        value = data.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Config field '{key}' must be a float.") from exc

    def optional_int(key: str) -> Optional[int]:
        value = data.get(key)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Config field '{key}' must be an integer.") from exc

    def as_label_list(key: str) -> Sequence[str]:
        value = data.get(key, DEFAULT_LANE_LABELS)
        if isinstance(value, str):
            return tuple(part.strip() for part in value.split(",") if part.strip())
        if isinstance(value, Iterable):
            cleaned = [str(item).strip() for item in value if str(item).strip()]
            return tuple(cleaned)
        raise ValueError(f"Config field '{key}' must be a sequence or comma-delimited string.")

    def as_color(key: str) -> Tuple[int, int, int]:
        value = data.get(key, (64, 255, 112))
        if isinstance(value, str):
            stripped = value.strip().lstrip("#")
            if len(stripped) == 6:
                r = int(stripped[0:2], 16)
                g = int(stripped[2:4], 16)
                b = int(stripped[4:6], 16)
                return (b, g, r)
        if isinstance(value, Iterable):
            ints: List[int] = []
            for item in value:
                try:
                    ints.append(int(item))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Color channel '{item}' is not an integer.") from exc
            if len(ints) != 3:
                raise ValueError("overlay_color must list exactly three integers (BGR).")
            if not all(0 <= c <= 255 for c in ints):
                raise ValueError("overlay_color values must be between 0 and 255.")
            return (ints[0], ints[1], ints[2])
        raise ValueError(f"Config field '{key}' must be a #RRGGBB string or 3-element sequence.")

    overlay_alpha = float(data.get("overlay_alpha", 0.6))
    if not 0.0 <= overlay_alpha <= 1.0:
        raise ValueError("overlay_alpha must be between 0 and 1.")

    min_score = float(data.get("min_score", 0.4))
    if not 0.0 <= min_score <= 1.0:
        raise ValueError("min_score must be between 0 and 1.")

    dilate_kernel = int(data.get("dilate_kernel", 5))
    if dilate_kernel < 1:
        raise ValueError("dilate_kernel must be >= 1.")

    return Mask2FormerLaneConfig(
        input_video=require_path("input_video"),
        output=optional_path("output"),
        model=str(data.get("model", DEFAULT_MODEL_ID)),
        device=str(data.get("device", "auto")),
        fps=optional_float("fps"),
        highlight_labels=as_label_list("highlight_labels"),
        overlay_color=as_color("overlay_color"),
        overlay_alpha=overlay_alpha,
        min_score=min_score,
        max_long_edge=optional_int("max_long_edge"),
        dilate_kernel=dilate_kernel,
        side_by_side=bool(data.get("side_by_side", True)),
        output_mask_only=bool(data.get("output_mask_only", False)),
        max_frames=optional_int("max_frames"),
    )


def create_video_writer(path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    return writer


def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def build_lane_mask(
    segmentation: torch.Tensor,
    segments_info: Sequence[Dict],
    label_lookup: Dict[int, str],
    target_labels: Sequence[str],
    min_score: float,
) -> np.ndarray:
    desired = {label.lower() for label in target_labels}
    if not desired:
        return np.zeros(tuple(segmentation.shape), dtype=bool)

    eligible_ids: List[int] = []
    for segment in segments_info:
        label_id = int(segment.get("label_id", -1))
        label_name = label_lookup.get(label_id, str(label_id))
        if label_name.lower() not in desired:
            continue
        score = float(segment.get("score", 1.0))
        if score < min_score:
            continue
        eligible_ids.append(int(segment["id"]))

    if not eligible_ids:
        return np.zeros(tuple(segmentation.shape), dtype=bool)

    segmentation_np = segmentation.cpu().numpy()
    mask = np.isin(segmentation_np, eligible_ids)
    return mask


def upscale_mask(mask: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_size
    if mask.shape[0] == target_h and mask.shape[1] == target_w:
        return mask
    resized = cv2.resize(
        mask.astype(np.uint8),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool)


def apply_overlay(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    color_bgr: Tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    if not mask.any():
        return frame_bgr

    overlay = frame_bgr.copy()
    overlay[mask] = (
        overlay[mask].astype(np.float32) * (1.0 - alpha)
        + np.array(color_bgr, dtype=np.float32) * alpha
    )
    overlay[mask] = np.clip(overlay[mask], 0, 255)
    return overlay.astype(np.uint8)


def dilate_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return mask
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    return dilated.astype(bool)


def maybe_resize_for_model(frame_bgr: np.ndarray, max_long_edge: Optional[int]) -> Tuple[np.ndarray, float]:
    if not max_long_edge:
        return frame_bgr, 1.0
    height, width = frame_bgr.shape[:2]
    long_edge = max(height, width)
    if long_edge <= max_long_edge:
        return frame_bgr, 1.0
    scale = max_long_edge / float(long_edge)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(frame_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, scale


def format_progress(current: int, total: Optional[int]) -> str:
    if total:
        return f"{current}/{total}"
    return str(current)


def process_video(config: Mask2FormerLaneConfig) -> Path:
    input_path = config.input_video
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    output_path = config.output or input_path.with_name(f"{input_path.stem}_panoptic_lanes.mp4")

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    original_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = capture.get(cv2.CAP_PROP_FPS) or 0.0
    fps_out = config.fps if config.fps else (fps_in if fps_in > 0 else 24.0)

    device = resolve_device(config.device)
    processor = AutoImageProcessor.from_pretrained(config.model)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(config.model)
    model.to(device)
    model.eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}

    output_width = original_width * 2 if config.side_by_side else original_width
    writer = create_video_writer(output_path, output_width, original_height, fps_out)

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    processed = 0

    try:
        while True:
            if config.max_frames is not None and processed >= config.max_frames:
                break

            success, frame = capture.read()
            if not success:
                break

            resized_frame, scale = maybe_resize_for_model(frame, config.max_long_edge)
            rgb_frame = to_rgb(resized_frame)

            inputs = processor(images=Image.fromarray(rgb_frame), return_tensors="pt")
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

            with torch.inference_mode():
                outputs = model(**inputs)
                processed_panoptic = processor.post_process_panoptic_segmentation(
                    outputs,
                    target_sizes=[rgb_frame.shape[:2]],
                )[0]

            segmentation = processed_panoptic["segmentation"]
            segments_info = processed_panoptic["segments_info"]
            mask = build_lane_mask(
                segmentation=segmentation,
                segments_info=segments_info,
                label_lookup=id2label,
                target_labels=config.highlight_labels,
                min_score=config.min_score,
            )

            mask_full_res = upscale_mask(mask, (resized_frame.shape[0], resized_frame.shape[1]))
            if scale != 1.0:
                mask_full_res = upscale_mask(mask_full_res, (frame.shape[0], frame.shape[1]))
            mask_full_res = dilate_mask(mask_full_res, config.dilate_kernel)

            if config.output_mask_only:
                overlay_frame = (mask_full_res.astype(np.uint8) * 255)
                overlay_frame = cv2.cvtColor(overlay_frame, cv2.COLOR_GRAY2BGR)
            else:
                overlay_frame = apply_overlay(frame, mask_full_res, config.overlay_color, config.overlay_alpha)

            if config.side_by_side:
                combined = np.hstack((frame, overlay_frame))
            else:
                combined = overlay_frame

            writer.write(combined)
            processed += 1
            if processed % 10 == 0:
                prog = format_progress(processed, config.max_frames or total_frames)
                print(f"Processed {prog} frames", end="\r", file=sys.stderr)
    finally:
        capture.release()
        writer.release()

    print(f"\nPanoptic lane video saved to: {output_path}")
    return output_path


class Mask2FormerLaneApp:
    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
        self.config: Optional[Mask2FormerLaneConfig] = None

    def load(self) -> None:
        self.config = load_config(self.config_path)

    def run(self) -> Path:
        if self.config is None:
            self.load()
        assert self.config is not None
        return process_video(self.config)


def main() -> None:
    app = Mask2FormerLaneApp()
    try:
        app.run()
    except KeyboardInterrupt:  # pragma: no cover - user interrupt
        print("\nInterrupted by user.")
    except Exception as exc:  # pragma: no cover - runtime errors
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
