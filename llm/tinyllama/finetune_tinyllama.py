"""
LoRA fine-tuning for TinyLlama-1.1B-Chat on JSONL command data.

Expected JSONL schema (train/val):
{"query": "...", "command": {...}}  # command will be serialized as pretty JSON for the label

Usage (example):
python finetune_tinyllama.py \
  --train-jsonl data/train.jsonl \
  --val-jsonl data/val.jsonl \
  --output-dir models/llm/tinyllama/finetuned \
  --config finetune_config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import yaml
from pathlib import Path
import importlib.util
from typing import Iterable, List, Tuple

import datasets
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def enable_perf_features():
    """Backend toggles that typically improve training throughput."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def format_example(example: dict) -> str:
    query = example.get("query") or ""
    command = example.get("command") or {}
    # Emit compact, single-line JSON to make the generation task simpler
    command_str = json.dumps(command, ensure_ascii=False, separators=(",", ":"))
    return f"User: {query}\nAssistant: {command_str}\n"


def build_dataset(train_path: Path, val_path: Path, tokenizer, max_length: int):
    """
    Wrap each row as a JSON string to avoid Arrow schema conflicts
    when fields vary in type (e.g., bool/int/str inside 'command').
    """
    train_rows = [json.dumps(r) for r in load_jsonl(train_path)]
    val_rows = [json.dumps(r) for r in load_jsonl(val_path)]

    def to_features(batch: dict) -> dict:
        objs = [json.loads(s) for s in batch["raw"]]
        texts = [format_example(ex) for ex in objs]
        toks = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        toks["labels"] = toks["input_ids"].copy()
        return toks

    train_ds = datasets.Dataset.from_dict({"raw": train_rows}).map(
        to_features, batched=True, remove_columns=["raw"]
    )
    val_ds = datasets.Dataset.from_dict({"raw": val_rows}).map(
        to_features, batched=True, remove_columns=["raw"]
    )
    return train_ds, val_ds


def parse_args():
    p = argparse.ArgumentParser()
    # Made optional so a config file can supply them; validated after config load.
    p.add_argument("--train-jsonl", required=False, help="Path to training JSONL.")
    p.add_argument("--val-jsonl", required=False, help="Path to validation JSONL.")
    p.add_argument("--output-dir", required=False, help="Where to save LoRA adapter.")
    p.add_argument("--config", default=None, help="Optional YAML/JSON config with overrides.")
    p.add_argument("--model-name", default=DEFAULT_MODEL)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--per-device-train-batch-size", type=int, default=6)
    p.add_argument("--per-device-eval-batch-size", type=int, default=6)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=24, help="DataLoader workers.")
    return p.parse_args()


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    enable_perf_features()
    args = parse_args()
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text())
        for k, v in cfg.items():
            if hasattr(args, k):
                cur = getattr(args, k)
                try:
                    # preserve the original argparse-inferred type when overriding
                    casted = type(cur)(v)
                except Exception:
                    casted = v
                setattr(args, k, casted)
    # Validate required paths after config overrides
    if not args.train_jsonl or not args.val_jsonl or not args.output_dir:
        raise SystemExit("Missing required paths: train-jsonl, val-jsonl, output-dir (pass as args or via config).")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": dtype,
    }
    flash_available = importlib.util.find_spec("flash_attn") is not None
    try:
        attn_kwargs = {"attn_implementation": "flash_attention_2"} if flash_available else {}
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            **attn_kwargs,
            **model_kwargs,
        )
    except (TypeError, ValueError, ImportError):
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            **model_kwargs,
        )
    model = prepare_model_for_kbit_training(model)
    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)

    train_ds, val_ds = build_dataset(Path(args.train_jsonl), Path(args.val_jsonl), tokenizer, args.max_length)

    optim_supported = "optim" in TrainingArguments.__init__.__code__.co_varnames
    optim_arg = {"optim": "adamw_torch"} if optim_supported else {}
    common_training_kwargs = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=20,
        save_steps=200,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=not use_bf16,
        seed=args.seed,
        report_to="none",
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        **optim_arg,
    )
    try:
        training_args = TrainingArguments(
            **common_training_kwargs,
            evaluation_strategy="no",  # disable mid-epoch eval to avoid worker/input edge cases
            save_strategy="steps",
        )
    except TypeError:
        # Fallback for older transformers versions without evaluation/save strategy args
        training_args = TrainingArguments(
            **common_training_kwargs,
            do_eval=False,
        )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=None,
        data_collator=collator,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved LoRA adapter + tokenizer to {output_dir}")


if __name__ == "__main__":
    main()
