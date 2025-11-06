#!/usr/bin/env python3
"""Create a random subset of a pooled COCO dataset.

The script reads a COCO annotation file, samples N images, copies their
corresponding files into `<output>/images`, and writes a trimmed JSON with
fresh image/annotation identifiers. Use it after pooling datasets when you
want a smaller experiment split.

Example:
    python sample_coco_subset.py ^
        --dataset-root C:/data/roadlane_pooled/train ^
        --annotations pooled_annotations.coco.json ^
        --output-dir C:/data/roadlane_subset/train ^
        --num-samples 500 --seed 21
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Randomly sample a COCO dataset.")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Folder that contains the COCO JSON and image files.",
    )
    parser.add_argument(
        "--annotations",
        default="pooled_annotations.coco.json",
        help="Annotation filename located inside --dataset-root.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination folder that will receive the sampled dataset.",
    )
    parser.add_argument(
        "--output-json",
        default="sampled_annotations.coco.json",
        help="Filename for the sampled annotation JSON (relative to --output-dir).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        required=True,
        help="How many images to keep in the subset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed so sampling can be reproduced.",
    )
    return parser.parse_args()


def unique_filename(base: str, used: Dict[str, int]) -> str:
    if base not in used:
        used[base] = 1
        return base

    stem = Path(base).stem
    suffix = Path(base).suffix
    idx = used[base]
    while True:
        candidate = f"{stem}_{idx}{suffix}"
        if candidate not in used:
            used[candidate] = 1
            used[base] = idx + 1
            return candidate
        idx += 1


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    annotations_path = dataset_root / args.annotations
    if not annotations_path.is_file():
        raise FileNotFoundError(f"Could not find {annotations_path}")

    with annotations_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    images: List[Dict] = data.get("images", [])
    if args.num_samples > len(images):
        raise ValueError(f"Requested {args.num_samples} samples but only {len(images)} images exist.")

    rng = random.Random(args.seed)
    selected = rng.sample(images, args.num_samples)

    output_dir = Path(args.output_dir).expanduser().resolve()
    used_names: Dict[str, int] = {}
    image_id_map: Dict[int, int] = {}
    new_images: List[Dict] = []
    next_img_id = 1

    for image in selected:
        rel_path = Path(image["file_name"])
        source_path = dataset_root / rel_path
        if not source_path.is_file():
            raise FileNotFoundError(f"Image file {source_path} referenced in JSON is missing.")

        target_filename = unique_filename(rel_path.name, used_names)
        target_rel_path = Path("images") / target_filename
        target_path = output_dir / target_rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)

        new_image = dict(image)
        new_image["id"] = next_img_id
        new_image["file_name"] = target_rel_path.as_posix()
        image_id_map[int(image["id"])] = next_img_id
        next_img_id += 1
        new_images.append(new_image)

    next_annotation_id = 1
    new_annotations: List[Dict] = []
    for annotation in data.get("annotations", []):
        src_image_id = int(annotation["image_id"])
        if src_image_id not in image_id_map:
            continue
        new_annotation = dict(annotation)
        new_annotation["id"] = next_annotation_id
        new_annotation["image_id"] = image_id_map[src_image_id]
        new_annotations.append(new_annotation)
        next_annotation_id += 1

    sampled = {
        "info": data.get("info") or {"description": "Sampled COCO subset"},
        "licenses": data.get("licenses", []),
        "images": new_images,
        "annotations": new_annotations,
        "categories": data.get("categories", []),
    }

    output_json = output_dir / args.output_json
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(sampled, handle, indent=2)

    print(f"[OK] Saved {args.num_samples} sampled images to {output_json}")


if __name__ == "__main__":
    main()
