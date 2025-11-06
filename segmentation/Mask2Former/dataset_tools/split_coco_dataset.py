#!/usr/bin/env python3
"""Split a single COCO dataset into multiple disjoint subsets.

Example:
    python split_coco_dataset.py ^
        --dataset-root C:/data/pooled_all ^
        --annotations pooled_annotations.coco.json ^
        --output-root C:/data/pooled_75205 ^
        --splits train=75 val=20 test=5 ^
        --seed 123
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a COCO dataset into multiple subsets.")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Directory that hosts the COCO JSON and image files.",
    )
    parser.add_argument(
        "--annotations",
        default="pooled_annotations.coco.json",
        help="Annotation filename inside --dataset-root.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Base directory that will contain <split>/images and JSON files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        required=True,
        help="Split definitions like train=75 val=20 test=5 (percentages or ratios).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible shuffling.",
    )
    return parser.parse_args()


def parse_split_spec(specs: Sequence[str]) -> List[Tuple[str, float]]:
    parsed: List[Tuple[str, float]] = []
    for item in specs:
        if "=" not in item:
            raise ValueError(f"Split '{item}' must look like name=value.")
        name, raw_value = item.split("=", 1)
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"Split ratio '{raw_value}' is not numeric.") from exc
        parsed.append((name.strip(), value))

    total = sum(value for _, value in parsed)
    if total <= 0:
        raise ValueError("Sum of split specifications must be > 0.")

    # Interpret as percentages if the sum is larger than 1.
    if total > 1.5:
        parsed = [(name, value / 100.0) for name, value in parsed]
        total = sum(value for _, value in parsed)

    if not abs(total - 1.0) <= 1e-6:
        # Normalize to sum exactly 1.0 in case of rounding.
        parsed = [(name, value / total) for name, value in parsed]
    return parsed


def allocate_counts(total_images: int, splits: List[Tuple[str, float]]) -> List[Tuple[str, int]]:
    exact_counts = []
    int_counts = []
    for name, ratio in splits:
        exact = total_images * ratio
        exact_counts.append((name, exact))
        int_counts.append((name, int(exact)))

    assigned = sum(count for _, count in int_counts)
    remainder = total_images - assigned

    if remainder > 0:
        # Distribute remaining images to splits with largest fractional parts.
        fractions = sorted(
            ((name, exact - int(exact)) for (name, exact), (_, _) in zip(exact_counts, int_counts)),
            key=lambda pair: pair[1],
            reverse=True,
        )
        idx = 0
        while remainder > 0 and idx < len(fractions):
            name, _ = fractions[idx]
            for i, (split_name, count) in enumerate(int_counts):
                if split_name == name:
                    int_counts[i] = (split_name, count + 1)
                    remainder -= 1
                    break
            idx += 1

    if remainder != 0:
        raise RuntimeError("Could not allocate split counts correctly.")
    return int_counts


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


def write_split(
    split_name: str,
    image_records: List[Dict],
    annotations_by_image: Dict[int, List[Dict]],
    dataset_root: Path,
    output_root: Path,
) -> None:
    used_names: Dict[str, int] = {}
    new_images: List[Dict] = []
    new_annotations: List[Dict] = []
    next_image_id = 1
    next_ann_id = 1

    split_dir = output_root / split_name
    images_dir = split_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for image in image_records:
        rel_path = Path(image["file_name"])
        src = dataset_root / rel_path
        if not src.is_file():
            raise FileNotFoundError(f"Missing image referenced in JSON: {src}")

        target_name = unique_filename(rel_path.name, used_names)
        target_rel_path = Path("images") / target_name
        dst = split_dir / target_rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

        remapped_image = dict(image)
        remapped_image["id"] = next_image_id
        remapped_image["file_name"] = target_rel_path.as_posix()
        new_images.append(remapped_image)

        for annotation in annotations_by_image.get(int(image["id"]), []):
            remapped_annotation = dict(annotation)
            remapped_annotation["id"] = next_ann_id
            remapped_annotation["image_id"] = next_image_id
            new_annotations.append(remapped_annotation)
            next_ann_id += 1

        next_image_id += 1

    subset = {
        "info": {},
        "licenses": [],
        "images": new_images,
        "annotations": new_annotations,
        "categories": [],
    }
    return subset


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    annotations_path = dataset_root / args.annotations
    if not annotations_path.is_file():
        raise FileNotFoundError(f"Could not find annotation file: {annotations_path}")

    with annotations_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    splits = parse_split_spec(args.splits)

    images = list(data.get("images", []))
    if not images:
        raise ValueError("Dataset contains no images.")

    rng = random.Random(args.seed)
    rng.shuffle(images)

    split_counts = allocate_counts(len(images), splits)

    annotations_by_image: Dict[int, List[Dict]] = {}
    for annotation in data.get("annotations", []):
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    cursor = 0
    for split_name, count in split_counts:
        subset_images = images[cursor : cursor + count]
        cursor += count
        subset = write_split(
            split_name,
            subset_images,
            annotations_by_image,
            dataset_root,
            output_root,
        )
        subset["info"] = data.get("info", {"description": "COCO split"})
        subset["licenses"] = data.get("licenses", [])
        subset["categories"] = data.get("categories", [])

        output_json = output_root / split_name / f"{split_name}_annotations.coco.json"
        with output_json.open("w", encoding="utf-8") as handle:
            json.dump(subset, handle, indent=2)
        print(f"[OK] Wrote {count} images to {output_json}")


if __name__ == "__main__":
    main()
