"""Video lane detection using the pip `laneatt` package.

Update laneatt_inference_config.yaml, then run:
    python LaneATT/laneatt_video_inference.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional

import cv2
import numpy as np
import torch
import yaml
from laneatt import LaneATT


DEFAULT_MODEL_CONFIG = Path(__file__).with_name("laneatt_model_config.yaml")
DEFAULT_RUN_CONFIG = Path(__file__).with_name("laneatt_inference_config.yaml")


@dataclass(slots=True)
class InferenceConfig:
    input_video: Path
    model_config: Path
    weights: Path
    output: Path
    frame_stride: int = 1
    apply_nms: bool = True
    nms_threshold: float = 40.0
    max_frames: Optional[int] = None
    show_preview: bool = False


def load_run_config(path: Path) -> InferenceConfig:
    raw = yaml.safe_load(path.read_text()) or {}

    def require(key: str) -> str:
        value = raw.get(key)
        if value is None:
            raise ValueError(f"Config field '{key}' is required.")
        return str(value)

    input_video = Path(require("input_video")).expanduser()
    model_config = Path(raw.get("model_config", DEFAULT_MODEL_CONFIG)).expanduser()
    weights = Path(require("weights")).expanduser()

    output_raw = raw.get("output")
    if output_raw:
        output = Path(str(output_raw)).expanduser()
    else:
        output = input_video.with_name(f"{input_video.stem}_laneatt.mp4")

    frame_stride = max(1, int(raw.get("frame_stride", 1)))
    apply_nms = bool(raw.get("apply_nms", True))
    nms_threshold = float(raw.get("nms_threshold", 40.0))
    show_preview = bool(raw.get("show_preview", False))

    max_frames_raw = raw.get("max_frames")
    max_frames = int(max_frames_raw) if max_frames_raw is not None else None
    if max_frames is not None and max_frames < 1:
        max_frames = None

    return InferenceConfig(
        input_video=input_video,
        model_config=model_config,
        weights=weights,
        output=output,
        frame_stride=frame_stride,
        apply_nms=apply_nms,
        nms_threshold=nms_threshold,
        max_frames=max_frames,
        show_preview=show_preview,
    )


def create_writer(path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    return writer


def render_lanes(
    proposals: torch.Tensor,
    frame_bgr: np.ndarray,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    if proposals is None or proposals.numel() == 0:
        return cv2.resize(frame_bgr, (img_w, img_h))

    lanes = proposals.detach().cpu()
    if lanes.ndim == 1:
        lanes = lanes.unsqueeze(0)

    steps = lanes.shape[1] - 5
    if steps <= 0:
        return cv2.resize(frame_bgr, (img_w, img_h))

    ys = np.linspace(img_h, 0, steps)
    overlay = cv2.resize(frame_bgr, (img_w, img_h))
    colors = [
        (88, 214, 141),
        (72, 149, 239),
        (255, 196, 86),
        (199, 108, 230),
        (255, 111, 145),
    ]

    for lane_idx, lane in enumerate(lanes):
        length = int(max(0, min(float(lane[4].item()), steps)))
        if length == 0:
            continue

        xs = lane[5:].numpy()
        color = colors[lane_idx % len(colors)]
        prev_x, prev_y = xs[0], ys[0]
        for idx in range(length):
            x, y = xs[idx], ys[idx]
            cv2.line(overlay, (int(prev_x), int(prev_y)), (int(x), int(y)), color, 2)
            prev_x, prev_y = x, y

    return overlay


def process_video(cfg: InferenceConfig) -> Path:
    if not cfg.input_video.is_file():
        raise FileNotFoundError(f"Input video not found: {cfg.input_video}")
    if not cfg.model_config.is_file():
        raise FileNotFoundError(f"Model config not found: {cfg.model_config}")
    if not cfg.weights.is_file():
        raise FileNotFoundError(f"Weights not found: {cfg.weights}")

    model = LaneATT(str(cfg.model_config))
    model.load(str(cfg.weights))
    model.eval()

    capture = cv2.VideoCapture(str(cfg.input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {cfg.input_video}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    writer = create_writer(cfg.output, model.img_w, model.img_h, float(fps))

    processed = 0
    saved = 0

    try:
        with torch.inference_mode():
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                processed += 1
                if processed % cfg.frame_stride != 0:
                    continue

                output = model.cv2_inference(frame)
                if cfg.apply_nms:
                    output = model.nms(output, cfg.nms_threshold)

                rendered = render_lanes(output, frame, model.img_w, model.img_h)
                writer.write(rendered)
                saved += 1

                if cfg.show_preview:
                    cv2.imshow("LaneATT", rendered)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if cfg.max_frames is not None and saved >= cfg.max_frames:
                    break
    finally:
        capture.release()
        writer.release()
        if cfg.show_preview:
            cv2.destroyAllWindows()

    print(f"Saved {saved} frames to {cfg.output}")
    return cfg.output


def main() -> None:
    config_path = DEFAULT_RUN_CONFIG
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1]).expanduser()
    cfg = load_run_config(config_path)
    process_video(cfg)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - runtime helper
        print(f"Error: {exc}")
