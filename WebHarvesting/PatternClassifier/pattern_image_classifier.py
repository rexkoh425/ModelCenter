"""
Classify new candlestick screenshots by comparing them to reference images via CLIP.

Expected directory layout:
    reference_catalog/
        Bullish_Engulfing/
            sample1.png
            sample2.png
        Shooting_Star/
            sample.png

The script encodes every reference image, then scores incoming query images and
returns the top-k matches plus (optional) behavioral metadata scraped earlier.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import yaml
from PIL import Image
from torch.nn.functional import normalize
from transformers import CLIPModel, CLIPProcessor


DEFAULT_CONFIG_FILENAME = "pattern_image_classifier_config.yaml"
DEFAULT_CONFIG_PATH = Path(__file__).with_name(DEFAULT_CONFIG_FILENAME)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(slots=True)
class ModelConfig:
    clip_model_id: str
    device: str = "auto"


@dataclass(slots=True)
class RunConfig:
    reference_dir: Path
    input_path: Path
    top_k: int = 3
    output_json: Path = Path("./WebHarvesting/TradingPatterns/data/pattern_image_matches.json")
    behavior_jsonl: Optional[Path] = None


def load_config(config_path: Path) -> tuple[ModelConfig, RunConfig]:
    expanded = Path(config_path).expanduser().resolve()
    if not expanded.is_file():
        raise FileNotFoundError(f"Config file not found: {expanded}")
    with expanded.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    model_section = data.get("model", {})
    run_section = data.get("run", {})

    model_cfg = ModelConfig(
        clip_model_id=str(model_section.get("clip_model_id", "openai/clip-vit-base-patch32")),
        device=str(model_section.get("device", "auto")),
    )
    run_cfg = RunConfig(
        reference_dir=Path(run_section.get("reference_dir", "./WebHarvesting/TradingPatterns/reference_catalog")).expanduser(),
        input_path=Path(run_section.get("input_path", "./WebHarvesting/TradingPatterns/new_samples")).expanduser(),
        top_k=int(run_section.get("top_k", 3)),
        output_json=Path(run_section.get("output_json", "./WebHarvesting/TradingPatterns/data/pattern_image_matches.json")).expanduser(),
        behavior_jsonl=Path(run_section["behavior_jsonl"]).expanduser() if run_section.get("behavior_jsonl") else None,
    )
    run_cfg.output_json.parent.mkdir(parents=True, exist_ok=True)
    return model_cfg, run_cfg


class PatternImageClassifier:
    def __init__(self, model_cfg: ModelConfig, run_cfg: RunConfig) -> None:
        self.model_cfg = model_cfg
        self.run_cfg = run_cfg

        self.device = self._resolve_device(model_cfg.device)
        self.processor = CLIPProcessor.from_pretrained(model_cfg.clip_model_id)
        self.model = CLIPModel.from_pretrained(model_cfg.clip_model_id).to(self.device)
        self.model.eval()

        self.behavior_map = self._load_behavior_map(run_cfg.behavior_jsonl)
        self.reference_embeddings = self._encode_reference_catalog(run_cfg.reference_dir)

    @staticmethod
    def _resolve_device(device_pref: str) -> torch.device:
        if device_pref == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_pref)

    @staticmethod
    def _load_behavior_map(path: Optional[Path]) -> Dict[str, Dict[str, str]]:
        if not path or not path.is_file():
            return {}
        mapping: Dict[str, Dict[str, str]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                name = str(record.get("pattern_name") or "").strip()
                if not name:
                    continue
                mapping[name.lower()] = {
                    "behavior": record.get("behavior"),
                    "expected_behavior": record.get("expected_behavior"),
                    "summary": record.get("summary"),
                    "source": record.get("source"),
                }
        return mapping

    def _encode_reference_catalog(self, root: Path) -> List[Tuple[str, Path, torch.Tensor]]:
        if not root.exists():
            raise FileNotFoundError(f"Reference directory not found: {root}")
        entries: List[Tuple[str, Path, torch.Tensor]] = []
        for subdir in sorted([p for p in root.iterdir() if p.is_dir()]):
            label = subdir.name.replace("_", " ").strip()
            for img_path in iter_images(subdir):
                embedding = self._encode_image(img_path)
                entries.append((label, img_path, embedding))
        if not entries:
            raise RuntimeError(f"No reference images found under {root}")
        return entries

    def _encode_image(self, image_path: Path) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embeddings = self.model.get_image_features(**inputs)
            embeddings = normalize(embeddings, p=2, dim=-1)
        return embeddings.squeeze(0).cpu()

    def classify_inputs(self, inputs: Sequence[Path]) -> List[dict]:
        ref_embeddings = torch.stack([emb for _, _, emb in self.reference_embeddings])
        results: List[dict] = []
        for img_path in inputs:
            embedding = self._encode_image(img_path)
            scores = (ref_embeddings @ embedding)
            topk = torch.topk(scores, k=min(self.run_cfg.top_k, scores.numel()))
            matches = []
            for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
                label, ref_path, _ = self.reference_embeddings[idx]
                meta = self.behavior_map.get(label.lower())
                matches.append(
                    {
                        "pattern_name": label,
                        "similarity": round(float(score), 4),
                        "reference_image": str(ref_path),
                        "behavior": meta.get("behavior") if meta else None,
                        "expected_behavior": meta.get("expected_behavior") if meta else None,
                        "source": meta.get("source") if meta else None,
                    }
                )
            results.append({"query_image": str(img_path), "matches": matches})
        return results


def iter_images(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for file in sorted(path.glob("**/*")):
        if file.suffix.lower() in IMAGE_EXTENSIONS and file.is_file():
            yield file


def gather_query_images(input_path: Path) -> List[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    if input_path.is_file():
        return [input_path]
    return list(iter_images(input_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify candlestick screenshots by similarity to known patterns.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to pattern_image_classifier_config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        model_cfg, run_cfg = load_config(args.config)
        classifier = PatternImageClassifier(model_cfg, run_cfg)
        query_images = gather_query_images(run_cfg.input_path)
        results = classifier.classify_inputs(query_images)
        with run_cfg.output_json.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
        print(f"Saved classification results for {len(results)} images to {run_cfg.output_json}")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
