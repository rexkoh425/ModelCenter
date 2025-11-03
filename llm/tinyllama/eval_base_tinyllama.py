"""
Evaluate the base TinyLlama model on a JSONL test set (no LoRA adapter).

Usage:
python models/llm/tinyllama/eval_base_tinyllama.py \
  --test-jsonl models/llm/tinyllama/splits/test.jsonl \
  --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --max-length 256 \
  --batch-size 4
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.llm.tinyllama.finetune_tinyllama import build_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test-jsonl", required=True, help="Path to test JSONL.")
    p.add_argument("--model-name", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", torch_dtype="auto")

    _, test_ds = build_dataset(Path(args.test_jsonl), Path(args.test_jsonl), tok, args.max_length)

    args_eval = TrainingArguments(
        output_dir="/tmp/tiny_base_eval",
        per_device_eval_batch_size=args.batch_size,
        fp16=True,
        report_to="none",
    )
    trainer = Trainer(model=model, args=args_eval, eval_dataset=test_ds)
    metrics = trainer.evaluate()
    print(metrics)
    if "eval_loss" in metrics:
        print("Perplexity:", math.exp(metrics["eval_loss"]))


if __name__ == "__main__":
    main()
