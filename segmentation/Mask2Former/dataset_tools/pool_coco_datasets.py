#!/usr/bin/env python3
"""Merge multiple Roboflow/COCO directories into a single pooled dataset.

Each input directory is expected to contain a COCO-style JSON file (Roboflow
ships `_annotations.coco.json` by default) plus the referenced images.
The script copies every image into `<output>/images`, rewrites the annotation
JSON so file paths stay valid, and remaps identifiers to keep them unique.

Example:
    python pool_coco_datasets.py ^
        --inputs C:/data/lane-set1/train C:/data/lane-set2/train ^
        --output-dir C:/data/roadlane_pooled/train
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pool multiple COCO datasets.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Folders that contain both the COCO JSON file and images.",
    )
    parser.add_argument(
        "--annotation-name",
        default="_annotations.coco.json",
        help="Annotation filename inside every input directory.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination directory that will contain `images/` and the pooled JSON.",
    )
    parser.add_argument(
        "--output-json",
        default="pooled_annotations.coco.json",
        help="Filename for the merged COCO JSON (relative to --output-dir).",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dedupe_filename(base_name: str, used: Dict[str, int]) -> str:
    """Return a unique filename by tracking how many times base has been used."""
    count = used.get(base_name, 0)
    if count == 0:
        used[base_name] = 1
        return base_name
    stem = Path(base_name).stem
    suffix = Path(base_name).suffix
    while True:
        candidate = f"{stem}_{count}{suffix}"
        if candidate not in used:
            used[candidate] = 1
            used[base_name] = count + 1
            return candidate
        count += 1


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"COCO file {path} does not contain a JSON object.")
    return data


def normalize_categories(
    categories: List[Dict],
    global_map: Dict[str, int],
    next_category_id: int,
) -> Tuple[List[Dict], Dict[int, int], int]:
    """Ensure consistent category IDs across datasets."""
    remap: Dict[int, int] = {}
    for category in categories:
        name = str(category["name"]).strip().lower()
        if name not in global_map:
            global_map[name] = next_category_id
            next_category_id += 1
        remap[int(category["id"])] = global_map[name]
    return categories, remap, next_category_id


def pool_datasets(
    inputs: List[Path],
    annotation_name: str,
    output_dir: Path,
    output_json: Path,
) -> None:
    ensure_dir(output_dir)
    images_out_dir = output_dir / "images"
    ensure_dir(images_out_dir)

    pooled = {
        "info": {"description": "Pooled COCO dataset", "version": "1.0"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }

    next_img_id = 1
    next_ann_id = 1
    next_category_id = 1
    next_license_id = 1
    category_lookup: Dict[str, int] = {}
    license_lookup: Dict[Tuple[str, str], int] = {}
    used_filenames: Dict[str, int] = {}

    for dataset_idx, dataset_path in enumerate(inputs, start=1):
        dataset_dir = dataset_path.expanduser().resolve()
        annotation_path = dataset_dir / annotation_name
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Missing {annotation_name} in {dataset_dir}")

        data = load_json(annotation_path)
        _, category_remap, next_category_id = normalize_categories(
            data.get("categories", []),
            category_lookup,
            next_category_id,
        )

        # Append new categories in global order once
        pooled["categories"] = [
            {"id": cid, "name": name}
            for name, cid in sorted(category_lookup.items(), key=lambda item: item[1])
        ]

        license_id_map: Dict[int, int] = {}
        for lic in data.get("licenses") or []:
            key = (str(lic.get("name")), str(lic.get("url")))
            if key not in license_lookup:
                license_lookup[key] = next_license_id
                lic_copy = dict(lic)
                lic_copy["id"] = next_license_id
                pooled["licenses"].append(lic_copy)
                next_license_id += 1
            license_id_map[int(lic["id"])] = license_lookup[key]

        prefix = dataset_dir.name.lower()
        image_id_map: Dict[int, int] = {}
        for image in data.get("images", []):
            original_name = Path(image["file_name"]).name
            new_name = dedupe_filename(f"{prefix}_{original_name}", used_filenames)
            rel_path = Path("images") / new_name

            src_candidates = [
                dataset_dir / image["file_name"],
                dataset_dir / original_name,
            ]
            src_image = next((candidate for candidate in src_candidates if candidate.is_file()), None)
            if src_image is None:
                raise FileNotFoundError(f"Could not locate {image['file_name']} in {dataset_dir}")

            dst_image = output_dir / rel_path
            dst_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_image, dst_image)

            new_id = next_img_id
            image_id_map[int(image["id"])] = new_id
            next_img_id += 1

            new_image = dict(image)
            new_image["id"] = new_id
            new_image["file_name"] = rel_path.as_posix()
            if "license" in new_image:
                original_license = new_image["license"]
                new_image["license"] = license_id_map.get(int(original_license), original_license)
            pooled["images"].append(new_image)

        for annotation in data.get("annotations", []):
            src_image_id = int(annotation["image_id"])
            if src_image_id not in image_id_map:
                continue
            new_annotation = dict(annotation)
            new_annotation["id"] = next_ann_id
            new_annotation["image_id"] = image_id_map[src_image_id]
            new_annotation["category_id"] = category_remap[int(annotation["category_id"])]
            pooled["annotations"].append(new_annotation)
            next_ann_id += 1

        print(f"[+] Added {len(image_id_map)} images from {dataset_dir}")

    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(pooled, handle, indent=2)
    print(f"[OK] Wrote pooled dataset to {output_json}")


def main() -> None:
    args = parse_args()
    inputs = [Path(item) for item in args.inputs]
    output_dir = Path(args.output_dir)
    output_json = output_dir / args.output_json
    pool_datasets(inputs, args.annotation_name, output_dir, output_json)


if __name__ == "__main__":
    main()
