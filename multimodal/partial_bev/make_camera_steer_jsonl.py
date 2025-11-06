"""
Build a JSONL index for camera-only steering data.

Expects a CSV with columns:
filename,steer

Example CSV row:
frame_000123.png,-0.12

Output JSONL rows:
{"camera": "D:/Datasets/CarlaDataSet/CameraFront_Steer/frame_000123.png", "steer": -0.12}
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images-root", required=True, help="Root folder containing camera frames.")
    p.add_argument("--steer-file", required=True, help="CSV (filename,steer) or plain text with one steer value per line.")
    p.add_argument("--out-jsonl", required=True, help="Where to write the JSONL index.")
    return p.parse_args()


def load_mapping(csv_path: Path) -> List[dict]:
    rows = []
    if csv_path.suffix.lower() == ".csv":
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "filename" not in reader.fieldnames or "steer" not in reader.fieldnames:
                raise ValueError("CSV must have columns: filename, steer")
            for r in reader:
                fn = r["filename"].strip()
                if not fn:
                    continue
                steer = float(r["steer"])
                rows.append({"filename": fn, "steer": steer})
    else:
        # Plain text: one steer value per line; filenames inferred by sorted images
        with csv_path.open("r", encoding="utf-8") as f:
            vals = [float(line.strip()) for line in f if line.strip()]
        rows = [{"filename": None, "steer": v} for v in vals]
    return rows


def main():
    args = parse_args()
    root = Path(args.images_root)
    steer_path = Path(args.steer_file)
    out_path = Path(args.out_jsonl)
    rows = load_mapping(steer_path)
    images = sorted(root.glob("*.png"))
    if rows and rows[0]["filename"] is None:
        if len(images) != len(rows):
            raise ValueError(f"Count mismatch: {len(images)} images vs {len(rows)} steer values")
        for i, r in enumerate(rows):
            r["filename"] = images[i].name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            cam_path = root / r["filename"]
            f.write(json.dumps({"camera": str(cam_path), "steer": r["steer"]}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} entries to {out_path}")


if __name__ == "__main__":
    main()
