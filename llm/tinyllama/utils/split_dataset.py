"""
Split a JSONL dataset into train/val/test with configurable ratios.
Default: 90% train, 5% val, 5% test.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSONL (e.g., train.jsonl)")
    ap.add_argument("--out-dir", default="/app/Output/splits", help="Base output directory (default: /app/Output/splits)")
    ap.add_argument("--out-train", required=False, help="Output train JSONL (overrides --out-dir)")
    ap.add_argument("--out-val", required=False, help="Output val JSONL (overrides --out-dir)")
    ap.add_argument("--out-test", required=False, help="Output test JSONL (overrides --out-dir)")
    ap.add_argument("--val-ratio", type=float, default=0.05, help="Fraction for val (default 0.05)")
    ap.add_argument("--test-ratio", type=float, default=0.05, help="Fraction for test (default 0.05)")
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    args = ap.parse_args()

    rows = load_jsonl(Path(args.input))
    random.Random(args.seed).shuffle(rows)

    n = len(rows)
    n_val = int(n * args.val_ratio)
    n_test = int(n * args.test_ratio)
    n_train = n - n_val - n_test

    train_rows = rows[:n_train]
    val_rows = rows[n_train:n_train + n_val]
    test_rows = rows[n_train + n_val:]

    out_dir = Path(args.out_dir)
    out_train = Path(args.out_train) if args.out_train else out_dir / "train.jsonl"
    out_val = Path(args.out_val) if args.out_val else out_dir / "val.jsonl"
    out_test = Path(args.out_test) if args.out_test else out_dir / "test.jsonl"

    write_jsonl(out_train, train_rows)
    write_jsonl(out_val, val_rows)
    write_jsonl(out_test, test_rows)

    print(f"Total: {n} -> train {len(train_rows)} ({out_train}), val {len(val_rows)} ({out_val}), test {len(test_rows)} ({out_test})")


if __name__ == "__main__":
    main()
