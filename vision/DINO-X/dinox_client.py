"""Convenience wrapper for running DINO-X Cloud API on local images.

This script batches one or more images through the DINO-X API, saves the
raw JSON response, and (optionally) produces annotated previews when the
visualisation dependencies are available.

Usage:
    # Edit DINO-X/dinox_client_config.yaml, then run:
    python dinox_client.py
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import yaml

try:
    from dds_cloudapi_sdk import Client, Config
    from dds_cloudapi_sdk.image_resizer import image_to_base64
    from dds_cloudapi_sdk.tasks.v2_task import V2Task
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "Missing dependency 'dds-cloudapi-sdk'. "
        "Install it first with `pip install dds-cloudapi-sdk`."
    ) from exc

# Optional dependencies for visualisation (box overlays).
try:  # pragma: no cover - optional dependency
    import cv2
    import numpy as np
    import supervision as sv

    VISUAL_LIBS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    VISUAL_LIBS_AVAILABLE = False
    cv2 = None
    np = None
    sv = None

# pycocotools is only required when masks are requested.
try:  # pragma: no cover - optional dependency
    from pycocotools import mask as mask_utils
except ImportError:  # pragma: no cover - optional dependency
    mask_utils = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class DinoXError(Exception):
    """Custom exception for DINO-X client errors."""


DEFAULT_CONFIG_FILENAME = "dinox_client_config.yaml"
DEFAULT_CONFIG_PATH = Path(__file__).with_name(DEFAULT_CONFIG_FILENAME)


@dataclass(slots=True)
class DinoXClientConfig:
    input_path: Path
    api_token: Optional[str] = None
    model: str = "DINO-X-1.0"
    prompt: Optional[str] = None
    prompt_free: bool = False
    targets: Sequence[str] = field(default_factory=lambda: ["bbox", "mask"])
    bbox_threshold: float = 0.25
    iou_threshold: float = 0.8
    output_dir: Path = Path("outputs/dinox")
    save_visuals: bool = False
    force: bool = False


def load_config(config_path: Path) -> DinoXClientConfig:
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

    def optional_str(key: str) -> Optional[str]:
        value = data.get(key)
        if value is None:
            return None
        return str(value)

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

    def coerce_float(key: str, default: float) -> float:
        value = data.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Config field '{key}' must be a float.") from exc

    def coerce_targets(value) -> Sequence[str]:
        if value is None:
            return ["bbox", "mask"]
        if isinstance(value, str):
            tokens = [token.strip() for token in value.split(",")]
            return [token for token in tokens if token]
        if isinstance(value, (list, tuple)):
            cleaned = []
            for item in value:
                item_str = str(item).strip()
                if item_str:
                    cleaned.append(item_str)
            return cleaned
        raise ValueError("Config field 'targets' must be a list or comma-separated string.")

    targets = coerce_targets(data.get("targets"))
    if not targets:
        raise ValueError("Config field 'targets' must include at least one entry.")

    return DinoXClientConfig(
        input_path=require_path("input_path"),
        api_token=optional_str("api_token"),
        model=str(data.get("model", "DINO-X-1.0")),
        prompt=optional_str("prompt"),
        prompt_free=coerce_bool("prompt_free", False),
        targets=targets,
        bbox_threshold=coerce_float("bbox_threshold", 0.25),
        iou_threshold=coerce_float("iou_threshold", 0.8),
        output_dir=Path(str(data.get("output_dir", "outputs/dinox"))).expanduser(),
        save_visuals=coerce_bool("save_visuals", False),
        force=coerce_bool("force", False),
    )




def resolve_api_token(explicit_token: Optional[str]) -> str:
    """Resolve the API token from CLI flag, env var, or .env file."""
    if explicit_token:
        return explicit_token.strip()

    env_token = os.getenv("DDS_API_TOKEN")
    if env_token:
        return env_token.strip()

    env_file = Path(".env")
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "DDS_API_TOKEN":
                return value.strip().strip('"').strip("'")

    raise DinoXError(
        "No API token provided. Pass --api-token, set DDS_API_TOKEN in your environment, "
        "or add DDS_API_TOKEN=<token> to a local .env file."
    )


def collect_images(input_path: str) -> List[Path]:
    """Collect all image paths from the provided input path."""
    path = Path(input_path)
    if not path.exists():
        raise DinoXError(f"Input path does not exist: {path}")

    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise DinoXError(f"Unsupported image extension: {path.suffix}")
        return [path]

    if not path.is_dir():
        raise DinoXError(f"Input path is neither file nor directory: {path}")

    images = sorted(
        p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise DinoXError(f"No images found in directory: {path}")
    return images


def build_prompt_payload(config: DinoXClientConfig) -> dict:
    """Create the prompt payload for the API request."""
    if config.prompt_free:
        return {"type": "universal"}

    if config.prompt:
        return {"type": "text", "text": config.prompt}

    raise DinoXError("Either supply 'prompt' or enable 'prompt_free' in the config.")


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def request_dinox(
    client: Client,
    image_path: Path,
    config: DinoXClientConfig,
) -> dict:
    """Send the image to DINO-X Cloud API and return the response payload."""
    image_b64 = image_to_base64(str(image_path))

    api_body = {
        "model": config.model,
        "image": image_b64,
        "prompt": build_prompt_payload(config),
        "targets": list(config.targets),
        "mask_format": "coco_rle" if "mask" in config.targets else None,
        "bbox_threshold": config.bbox_threshold,
        "iou_threshold": config.iou_threshold,
    }

    # Remove None entries so the API receives a clean payload.
    api_body = {k: v for k, v in api_body.items() if v is not None}

    task = V2Task(
        api_path="/v2/task/dinox/detection",
        api_body=api_body,
    )

    client.run_task(task)
    return task.result


def save_json(payload: dict, path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise DinoXError(f"{path} already exists. Use --force to overwrite.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def annotate_image(
    image_path: Path,
    payload: dict,
    output_dir: Path,
    force: bool,
) -> Optional[Path]:
    """Generate annotated overlays if optional dependencies are installed."""
    if not VISUAL_LIBS_AVAILABLE:
        print(
            "Skipping visualisation (missing OpenCV/numpy/supervision). "
            "Install them and rerun with --save-visuals. "
            "You can still access the raw JSON outputs.",
            file=sys.stderr,
        )
        return None

    objects = payload.get("objects", [])
    if not objects:
        print(f"No objects returned for {image_path.name}; nothing to visualise.")
        return None

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not read image {image_path}; skipping visualisation.", file=sys.stderr)
        return None

    boxes = []
    masks = []
    scores = []
    class_names: List[str] = []

    for obj in objects:
        bbox = obj.get("bbox")
        mask = obj.get("mask")
        score = obj.get("score", 0.0)
        cls_name = (obj.get("category") or "object").strip().lower()

        if bbox is None:
            continue
        boxes.append(bbox)
        scores.append(float(score))
        class_names.append(cls_name or "object")

        if mask is not None and mask_utils is not None:
            decoded = mask_utils.decode(mask)
            if decoded.ndim == 3:
                decoded = decoded[..., 0]
            masks.append(decoded.astype(bool))

    if not boxes:
        print(f"No valid detections to draw for {image_path.name}.")
        return None

    boxes_arr = np.array(boxes, dtype=np.float32)

    class_map: dict[str, int] = {}
    class_id_arr = []
    for name in class_names:
        if name not in class_map:
            class_map[name] = len(class_map)
        class_id_arr.append(class_map[name])
    class_id_arr = np.array(class_id_arr, dtype=np.int32)
    labels = [f"{name} {score:.2f}" for name, score in zip(class_names, scores)]

    mask_array = np.asarray(masks, dtype=bool) if masks else None

    detections = sv.Detections(
        xyxy=boxes_arr,
        class_id=class_id_arr,
        mask=mask_array,
    )

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    mask_annotator = sv.MaskAnnotator() if masks else None

    annotated = box_annotator.annotate(scene=img.copy(), detections=detections)
    annotated = label_annotator.annotate(
        scene=annotated,
        detections=detections,
        labels=labels,
    )
    if mask_annotator:
        annotated = mask_annotator.annotate(scene=annotated, detections=detections)

    annotated_path = output_dir / "annotated.jpg"
    if annotated_path.exists() and not force:
        raise DinoXError(f"{annotated_path} already exists. Use --force to overwrite.")

    cv2.imwrite(str(annotated_path), annotated)
    return annotated_path


def process_images(config_data: DinoXClientConfig) -> None:
    token = resolve_api_token(config_data.api_token)
    images = collect_images(config_data.input_path)
    output_root = Path(config_data.output_dir)
    ensure_output_dir(output_root)

    sdk_config = Config(token)
    client = Client(sdk_config)

    for image_path in images:
        print(f"[DINO-X] Processing {image_path} ...")
        try:
            payload = request_dinox(client, image_path, config_data)
        except Exception as exc:
            raise DinoXError(f"API request failed for {image_path.name}: {exc}") from exc

        image_output_dir = output_root / image_path.stem
        result_path = image_output_dir / "dinox_result.json"

        save_json(payload, result_path, force=config_data.force)
        print(f"  • Saved raw response to {result_path}")

        if config_data.save_visuals:
            try:
                annotated_path = annotate_image(
                    image_path=image_path,
                    payload=payload,
                    output_dir=image_output_dir,
                    force=config_data.force,
                )
                if annotated_path:
                    print(f"  • Saved annotated preview to {annotated_path}")
            except DinoXError as exc:
                print(f"  ! Visualisation skipped: {exc}", file=sys.stderr)


class DinoXClientApp:
    """Class wrapper to load YAML config and process images via DINO-X."""

    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
        self.config: Optional[DinoXClientConfig] = None

    def load(self) -> None:
        self.config = load_config(self.config_path)

    def run(self) -> None:
        if self.config is None:
            self.load()
        assert self.config is not None
        process_images(self.config)


def main() -> None:
    app = DinoXClientApp()
    try:
        app.run()
    except DinoXError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
