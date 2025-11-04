"""Translate a video between visual domains using a pre-trained CycleGAN model.

This script loads one of the standard CycleGAN generators (e.g. horse2zebra),
applies it frame-by-frame to an input video, and writes the translated frames
to a new video file. Optional flags let you export a side-by-side comparison
with the original frames or limit processing to a subset of frames.

Usage:
    # Edit CycleGAN/cyclegan_video_config.yaml, then run:
    python cyclegan_video.py

Dependencies:
    pip install torch torchvision pillow opencv-python numpy

The first invocation will download the selected CycleGAN weights
from https://download.pytorch.org/models/cyclegan/.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import yaml


# Canonical CycleGAN checkpoints released with the original paper.
MODEL_URLS = {
    "apple2orange": "https://download.pytorch.org/models/cyclegan/apple2orange.pth",
    "orange2apple": "https://download.pytorch.org/models/cyclegan/orange2apple.pth",
    "horse2zebra": "https://download.pytorch.org/models/cyclegan/horse2zebra.pth",
    "zebra2horse": "https://download.pytorch.org/models/cyclegan/zebra2horse.pth",
    "summer2winter_yosemite": "https://download.pytorch.org/models/cyclegan/summer2winter_yosemite.pth",
    "winter2summer_yosemite": "https://download.pytorch.org/models/cyclegan/winter2summer_yosemite.pth",
    "monet2photo": "https://download.pytorch.org/models/cyclegan/monet2photo.pth",
    "photo2monet": "https://download.pytorch.org/models/cyclegan/photo2monet.pth",
    "cezanne2photo": "https://download.pytorch.org/models/cyclegan/cezanne2photo.pth",
    "photo2cezanne": "https://download.pytorch.org/models/cyclegan/photo2cezanne.pth",
    "ukiyoe2photo": "https://download.pytorch.org/models/cyclegan/ukiyoe2photo.pth",
    "photo2ukiyoe": "https://download.pytorch.org/models/cyclegan/photo2ukiyoe.pth",
    "vangogh2photo": "https://download.pytorch.org/models/cyclegan/vangogh2photo.pth",
    "photo2vangogh": "https://download.pytorch.org/models/cyclegan/photo2vangogh.pth",
}


DEFAULT_CONFIG_FILENAME = "cyclegan_video_config.yaml"
DEFAULT_CONFIG_PATH = Path(__file__).with_name(DEFAULT_CONFIG_FILENAME)


@dataclass(slots=True)
class CycleGANVideoConfig:
    input_video: Path
    output: Optional[Path] = None
    mapping: str = "horse2zebra"
    device: str = "auto"
    fp16: bool = False
    fps: Optional[float] = None
    side_by_side: bool = False
    max_frames: Optional[int] = None
    verbose: bool = False


def load_config(config_path: Path) -> CycleGANVideoConfig:
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

    def coerce_float(key: str) -> Optional[float]:
        value = data.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Config field '{key}' must be a float.") from exc

    def coerce_int(key: str) -> Optional[int]:
        value = data.get(key)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Config field '{key}' must be an integer.") from exc

    mapping = str(data.get("mapping", "horse2zebra"))
    if mapping not in MODEL_URLS:
        raise ValueError(f"Unsupported mapping '{mapping}'. Expected one of: {', '.join(MODEL_URLS)}.")

    device = str(data.get("device", "auto"))

    return CycleGANVideoConfig(
        input_video=require_path("input_video"),
        output=optional_path("output"),
        mapping=mapping,
        device=device,
        fp16=coerce_bool("fp16", False),
        fps=coerce_float("fps"),
        side_by_side=coerce_bool("side_by_side", False),
        max_frames=coerce_int("max_frames"),
        verbose=coerce_bool("verbose", False),
    )


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():  # pragma: no cover - macOS specific
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available.")
        return torch.device("cuda")

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not supported on this system.")
        return torch.device("mps")

    return torch.device("cpu")


class ResnetBlock(nn.Module):
    """Standard ResNet block used by CycleGAN."""

    def __init__(self, dim: int, use_dropout: bool = False) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=True),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers.extend(
            [
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim, dim, kernel_size=3, bias=True),
                nn.InstanceNorm2d(dim),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    """Minimal CycleGAN generator (9 residual blocks)."""

    def __init__(self, input_nc: int = 3, output_nc: int = 3, ngf: int = 64) -> None:
        super().__init__()
        n_blocks = 9
        n_downsampling = 2

        model: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
        ]

        for i in range(n_downsampling):
            mult = 2**i
            model.extend(
                [
                    nn.Conv2d(
                        ngf * mult,
                        ngf * mult * 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=True,
                    ),
                    nn.InstanceNorm2d(ngf * mult * 2),
                    nn.ReLU(True),
                ]
            )

        mult = 2**n_downsampling
        for _ in range(n_blocks):
            model.append(ResnetBlock(ngf * mult))

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model.extend(
                [
                    nn.ConvTranspose2d(
                        ngf * mult,
                        int(ngf * mult / 2),
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                        bias=True,
                    ),
                    nn.InstanceNorm2d(int(ngf * mult / 2)),
                    nn.ReLU(True),
                ]
            )

        model.extend(
            [
                nn.ReflectionPad2d(3),
                nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                nn.Tanh(),
            ]
        )

        self.main = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple
        return self.main(x)


def load_generator(mapping: str, device: torch.device, use_fp16: bool) -> ResnetGenerator:
    if mapping not in MODEL_URLS:
        raise ValueError(f"Unknown mapping '{mapping}'.")

    state_dict = load_state_dict_from_url(MODEL_URLS[mapping], map_location=device, progress=True)
    if isinstance(state_dict, dict):
        for key_option in ("state_dict", "params", "model", "netG_A", "netG"):
            if key_option in state_dict:
                state_dict = state_dict[key_option]
                break

    # Strip possible "module." prefixes from DataParallel checkpoints.
    if isinstance(state_dict, dict) and any(key.startswith("module.") for key in state_dict):
        state_dict = {key[len("module.") :]: value for key, value in state_dict.items()}

    net = ResnetGenerator()
    if isinstance(state_dict, dict):
        missing, unexpected = net.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Failed to load generator weights. Missing: {missing}, Unexpected: {unexpected}"
            )

    net.to(device)
    net.eval()
    if use_fp16 and device.type == "cuda":
        net.half()
    return net


def prepare_frame(frame_bgr: np.ndarray, device: torch.device, use_fp16: bool) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0)
    tensor = tensor * 2.0 - 1.0  # scale to [-1, 1]
    dtype = torch.float16 if use_fp16 and device.type == "cuda" else torch.float32
    return tensor.to(device=device, dtype=dtype)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().clamp(-1, 1)
    tensor = (tensor + 1.0) / 2.0
    array = tensor.squeeze(0).permute(1, 2, 0).numpy()
    array = (array * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def create_video_writer(output_path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to initialise writer for {output_path}.")
    return writer


def translate_video(config: CycleGANVideoConfig) -> Path:
    input_path = config.input_video
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    output_path = (
        config.output
        if config.output
        else input_path.with_name(f"{input_path.stem}_{config.mapping}.mp4")
    )

    device = resolve_device(config.device)
    generator = load_generator(config.mapping, device, config.fp16)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    fps_out = config.fps if config.fps else (source_fps if source_fps > 0 else 24.0)

    output_width = width * 2 if config.side_by_side else width
    output_height = height
    writer = create_video_writer(output_path, output_width, output_height, fps_out)

    frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = config.max_frames if config.max_frames else frame_total

    processed = 0
    try:
        while True:
            if max_frames and processed >= max_frames:
                break

            success, frame = capture.read()
            if not success:
                break

            input_tensor = prepare_frame(frame, device, config.fp16)
            with torch.no_grad():
                translated = generator(input_tensor)
            translated_img = tensor_to_image(translated)

            if config.side_by_side:
                combined = np.hstack((frame, translated_img))
                writer.write(combined)
            else:
                writer.write(translated_img)

            processed += 1
            if config.verbose and processed % 10 == 0:
                total_disp = max_frames if max_frames > 0 else "?"
                print(f"Processed {processed}/{total_disp} frames", file=sys.stderr)
    finally:
        capture.release()
        writer.release()

    print(f"CycleGAN video saved to: {output_path}")
    return output_path


class CycleGANVideoApp:
    """Class wrapper that loads config and runs the translator."""

    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
        self.config: Optional[CycleGANVideoConfig] = None

    def load(self) -> None:
        self.config = load_config(self.config_path)

    def run(self) -> Path:
        if self.config is None:
            self.load()
        assert self.config is not None
        return translate_video(self.config)


def main() -> None:
    app = CycleGANVideoApp()
    try:
        app.run()
    except KeyboardInterrupt:  # pragma: no cover - user interruption
        print("\nInterrupted by user.")
    except Exception as exc:  # pragma: no cover - runtime errors
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
