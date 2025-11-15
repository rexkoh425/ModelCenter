"""YOLOE bounding-box detection pipeline.

This utility downloads a YOLOE checkpoint from Hugging Face, runs zero-shot
text-prompted detection on one or more images, and writes the raw results to
disk. Optionally, it can render annotated previews when OpenCV, NumPy, and
Supervision are installed.

Usage:
    # Edit YOLOE/yoloe_bbox_pipeline_config.yaml, then run:
    python yoloe_bbox_pipeline.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

try:
    import torch
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "Missing dependency 'torch'. Install it with `pip install torch`."
    ) from exc

try:
    from ultralytics import YOLOE
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "Missing dependency 'ultralytics'. Install it with `pip install ultralytics` "
        "and ensure your version is recent enough to expose YOLOE."
    ) from exc

try:
from huggingface_hub import hf_hub_download
import yaml
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "Missing dependency 'huggingface-hub'. Install it with `pip install huggingface-hub`."
    ) from exc

# Optional visualisation helpers.
try:  # pragma: no cover - optional dependency
    import cv2
    import numpy as np
    import supervision as sv

    VISUAL_DEPS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    VISUAL_DEPS_AVAILABLE = False
    cv2 = None
    np = None
    sv = None


MODEL_CHOICES = [
    "yoloe-v8s",
    "yoloe-v8m",
    "yoloe-v8l",
    "yoloe-11s",
    "yoloe-11m",
    "yoloe-11l",
]


DEFAULT_CONFIG_FILENAME = "yoloe_bbox_pipeline_config.yaml"
DEFAULT_CONFIG_PATH = Path(__file__).with_name(DEFAULT_CONFIG_FILENAME)


@dataclass(slots=True)
class YOLOEConfig:
    input_path: Path
    prompt: str = "person, car"
    model: str = "yoloe-v8l"
    image_size: int = 640
    confidence: float = 0.25
    iou: float = 0.5
    device: str = "auto"
    output_dir: Path = Path("outputs/yoloe")
    save_visuals: bool = False
    force: bool = False


def load_config(config_path: Path) -> YOLOEConfig:
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

    def coerce_int(key: str, default: int) -> int:
        value = data.get(key, default)
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Config field '{key}' must be an integer.") from exc

    def coerce_float(key: str, default: float) -> float:
        value = data.get(key, default)
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

    model_name = str(data.get("model", "yoloe-v8l"))
    if model_name not in MODEL_CHOICES:
        raise ValueError(f"Unsupported model '{model_name}'. Expected one of: {', '.join(MODEL_CHOICES)}.")

    return YOLOEConfig(
        input_path=require_path("input_path"),
        prompt=str(data.get("prompt", "person, car")),
        model=model_name,
        image_size=coerce_int("image_size", 640),
        confidence=coerce_float("confidence", 0.25),
        iou=coerce_float("iou", 0.5),
        device=str(data.get("device", "auto")),
        output_dir=Path(str(data.get("output_dir", "outputs/yoloe"))).expanduser(),
        save_visuals=coerce_bool("save_visuals", False),
        force=coerce_bool("force", False),
    )
@dataclass
class DetectionRecord:
    bbox: List[float]
    confidence: float
    class_id: int
    class_name: str


def resolve_device(preference: str) -> str:
    if preference == "cpu":
        return "cpu"
    if preference == "cuda" and torch.cuda.is_available():
        return "cuda"
    if preference == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if preference == "cuda":
        print("CUDA requested but not available; falling back to CPU.", file=sys.stderr)
        return "cpu"
    return "cpu"


def collect_images(path_str: Path | str) -> List[Path]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if path.is_file():
        return [path]
    images = sorted(
        p
        for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    )
    if not images:
        raise ValueError(f"No supported images found in directory: {path}")
    return images


def load_model(model_id: str, device: str) -> YOLOE:
    weight_name = f"{model_id}-seg.pt"
    weight_path = hf_hub_download(repo_id="jameslahm/yoloe", filename=weight_name)
    model = YOLOE(weight_path)
    model.eval()
    model.to(device)
    return model


def prepare_classes(model: YOLOE, prompt: str) -> List[str]:
    prompts = [token.strip() for token in prompt.split(",") if token.strip()]
    if not prompts:
        raise ValueError("Prompt must contain at least one category.")
    embeddings = model.get_text_pe(prompts)
    model.set_classes(prompts, embeddings)
    return prompts


def run_inference(
    model: YOLOE,
    image_path: Path,
    image_size: int,
    confidence: float,
    iou: float,
    class_lookup: Dict[int, str],
) -> List[DetectionRecord]:
    results = model.predict(
        source=str(image_path),
        imgsz=image_size,
        conf=confidence,
        iou=iou,
        verbose=False,
    )
    if not results:
        return []
    result = results[0]
    names = result.names or class_lookup

    boxes = result.boxes.xyxy.cpu().tolist() if result.boxes is not None else []
    scores = result.boxes.conf.cpu().tolist() if result.boxes is not None else []
    classes = result.boxes.cls.cpu().tolist() if result.boxes is not None else []

    detections: List[DetectionRecord] = []
    for bbox, score, class_idx in zip(boxes, scores, classes):
        class_id = int(class_idx)
        class_name = names.get(class_id, class_lookup.get(class_id, str(class_id)))
        detections.append(
            DetectionRecord(
                bbox=[float(x) for x in bbox],
                confidence=float(score),
                class_id=class_id,
                class_name=class_name,
            )
        )
    return detections


def save_json(
    detections: List[DetectionRecord],
    output_path: Path,
    image_path: Path,
    model_id: str,
    prompts: List[str],
    force: bool,
) -> None:
    if output_path.exists() and not force:
        raise FileExistsError(f"{output_path} already exists. Use --force to overwrite.")

    payload = {
        "image": str(image_path),
        "model": model_id,
        "prompts": prompts,
        "detections": [
            {
                "bbox": det.bbox,
                "confidence": det.confidence,
                "class_id": det.class_id,
                "class_name": det.class_name,
            }
            for det in detections
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def annotate(
    image_path: Path,
    detections: List[DetectionRecord],
    output_dir: Path,
    force: bool,
) -> Optional[Path]:
    if not VISUAL_DEPS_AVAILABLE:
        print(
            "Skipping visualisation (requires opencv-python, numpy, supervision).",
            file=sys.stderr,
        )
        return None

    if not detections:
        print(f"No detections produced for {image_path.name}; skipping preview.")
        return None

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read image {image_path}; skipping preview.", file=sys.stderr)
        return None

    boxes = np.array([det.bbox for det in detections], dtype=np.float32)
    class_ids: Dict[str, int] = {}
    class_index = []
    confidences = []
    labels = []

    for det in detections:
        if det.class_name not in class_ids:
            class_ids[det.class_name] = len(class_ids)
        class_index.append(class_ids[det.class_name])
        confidences.append(det.confidence)
        labels.append(f"{det.class_name} {det.confidence:.2f}")

    detections_sv = sv.Detections(
        xyxy=boxes,
        class_id=np.array(class_index, dtype=np.int32),
        confidence=np.array(confidences, dtype=np.float32),
    )

    annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    annotated = annotator.annotate(scene=image.copy(), detections=detections_sv)
    annotated = label_annotator.annotate(scene=annotated, detections=detections_sv, labels=labels)

    preview_path = output_dir / "annotated.jpg"
    if preview_path.exists() and not force:
        raise FileExistsError(f"{preview_path} already exists. Use --force to overwrite.")

    cv2.imwrite(str(preview_path), annotated)
    return preview_path


class YOLOEPipelineApp:
    """Class wrapper to load YAML config and run the YOLOE pipeline."""

    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
        self.config: Optional[YOLOEConfig] = None

    def load(self) -> None:
        self.config = load_config(self.config_path)

    def run(self) -> None:
        if self.config is None:
            self.load()
        assert self.config is not None

        device = resolve_device(self.config.device)

        try:
            images = collect_images(self.config.input_path)
        except (FileNotFoundError, ValueError) as exc:
            raise RuntimeError(str(exc)) from exc

        model = load_model(self.config.model, device)
        prompts = prepare_classes(model, self.config.prompt)
        class_lookup = {idx: name for idx, name in enumerate(prompts)}

        output_root = Path(self.config.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        for image_path in images:
            print(f"[YOLOE] Processing {image_path} ...")
            detections = run_inference(
                model=model,
                image_path=image_path,
                image_size=self.config.image_size,
                confidence=self.config.confidence,
                iou=self.config.iou,
                class_lookup=class_lookup,
            )

            image_output_dir = output_root / image_path.stem
            result_path = image_output_dir / "yoloe_result.json"

            save_json(
                detections=detections,
                output_path=result_path,
                image_path=image_path,
                model_id=self.config.model,
                prompts=prompts,
                force=self.config.force,
            )
            print(f"  • Saved detections to {result_path}")

            if self.config.save_visuals:
                preview = annotate(
                    image_path=image_path,
                    detections=detections,
                    output_dir=image_output_dir,
                    force=self.config.force,
                )
                if preview:
                    print(f"  • Saved preview to {preview}")


def main() -> None:
    app = YOLOEPipelineApp()
    try:
        app.run()
    except KeyboardInterrupt:  # pragma: no cover - user abort
        print("\nInterrupted by user.")
        sys.exit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
