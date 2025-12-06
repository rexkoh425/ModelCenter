#!/usr/bin/env python3
"""
TinyLlama LoRA SFT (query -> command) — stable FP16 AMP training on 16GB VRAM.

Fixes your crash:
  ValueError: Attempting to unscale FP16 gradients.

Root cause: some trainable params (LoRA adapters) ended up FP16, but GradScaler
expects trainable params to be FP32-mastered.
Solution: keep base model weights FP16 to save VRAM, BUT force ALL trainable
(LoRA) params to FP32 immediately after get_peft_model().

Features:
- Loss only on assistant tokens (prompt masked to -100)
- Uses chat template if available (tokenizer.apply_chat_template)
- Optional gradient checkpointing (faster OFF for 1.1B @ 512 with 16GB)
- Defaults tuned for 16GB VRAM: bs=8, ga=2, workers=4
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import datasets
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model


DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def read_text(path: str | None) -> str:
    if not path:
        return ""
    return Path(path).read_text(encoding="utf-8").strip()


def to_cmd_text(cmd, json_commands: bool) -> str:
    if json_commands:
        return json.dumps(cmd, ensure_ascii=False, separators=(",", ":"))
    if isinstance(cmd, str):
        return cmd
    return str(cmd)


def build_messages(query: str, cmd_text: str, system_prompt: str) -> tuple[list[dict], list[dict]]:
    prompt_msgs: list[dict] = []
    if system_prompt:
        prompt_msgs.append({"role": "system", "content": system_prompt})
    prompt_msgs.append({"role": "user", "content": query})
    full_msgs = prompt_msgs + [{"role": "assistant", "content": cmd_text}]
    return prompt_msgs, full_msgs


def tokenize_with_assistant_only_labels(
    tokenizer: AutoTokenizer,
    prompt_msgs: list[dict],
    full_msgs: list[dict],
    max_length: int,
) -> dict:
    """
    Creates input_ids/attention_mask/labels with labels=-100 for prompt tokens
    and labels=token_id for assistant tokens only.
    """
    eos_id = tokenizer.eos_token_id

    try:
        prompt_ids = tokenizer.apply_chat_template(
            prompt_msgs,
            tokenize=True,
            add_generation_prompt=True,   # includes assistant prefix; defines boundary
        )
        full_ids = tokenizer.apply_chat_template(
            full_msgs,
            tokenize=True,
            add_generation_prompt=False,
        )
    except Exception:
        # Fallback if no chat template exists
        sys = prompt_msgs[0]["content"] if (prompt_msgs and prompt_msgs[0]["role"] == "system") else ""
        user = prompt_msgs[-1]["content"]
        prompt_text = (f"{sys}\n" if sys else "") + f"User: {user}\nAssistant: "
        full_text = prompt_text + full_msgs[-1]["content"]
        prompt_ids = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=True)["input_ids"]

    # Ensure EOS at end
    if eos_id is not None and (len(full_ids) == 0 or full_ids[-1] != eos_id):
        full_ids = full_ids + [eos_id]

    prompt_len_orig = len(prompt_ids)
    full_len_orig = len(full_ids)

    # Left-truncate to preserve assistant tail
    drop = max(0, full_len_orig - max_length)
    if drop > 0:
        full_ids = full_ids[drop:]
        prompt_len = max(0, prompt_len_orig - drop)
    else:
        prompt_len = prompt_len_orig

    prompt_len = min(prompt_len, len(full_ids))

    labels = [-100] * prompt_len + full_ids[prompt_len:]
    attention_mask = [1] * len(full_ids)

    return {"input_ids": full_ids, "attention_mask": attention_mask, "labels": labels}


class DataCollatorForCausalLM:
    """Pad input_ids/attention_mask; pad labels with -100."""

    def __init__(self, tokenizer: AutoTokenizer, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict]) -> dict:
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]
        out_labels = torch.full((len(labels), max_len), -100, dtype=torch.long)
        for i, lab in enumerate(labels):
            out_labels[i, : len(lab)] = torch.tensor(lab, dtype=torch.long)

        batch["labels"] = out_labels
        return batch


def force_trainable_fp32(model) -> None:
    """
    CRITICAL FIX: make sure ALL trainable params are FP32 so GradScaler can unscale.
    """
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--val-jsonl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default=DEFAULT_MODEL)

    p.add_argument("--system-prompt", default=None)

    p.add_argument("--string-commands", action="store_true")
    p.add_argument("--json-commands", action="store_true")

    p.add_argument("--max-length", type=int, default=512)

    # Defaults tuned for 16GB VRAM @ 512 tokens
    p.add_argument("--per-device-train-batch-size", type=int, default=8)
    p.add_argument("--per-device-eval-batch-size", type=int, default=8)
    p.add_argument("--gradient-accumulation-steps", type=int, default=2)

    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-grad-norm", type=float, default=1.0)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)

    # LoRA
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    # Optional (usually OFF for 1.1B @ 512 on 16GB)
    p.add_argument("--gradient-checkpointing", action="store_true")

    # Logging/saving
    p.add_argument("--logging-steps", type=int, default=20)
    p.add_argument("--eval-strategy", default="epoch", choices=["no", "steps", "epoch"])
    p.add_argument("--save-strategy", default="epoch", choices=["no", "steps", "epoch"])
    p.add_argument("--save-steps", type=int, default=1000)
    p.add_argument("--save-total-limit", type=int, default=2)

    p.add_argument("--resume-from-checkpoint", default=None)

    return p.parse_args()


def make_training_args(args: argparse.Namespace) -> TrainingArguments:
    """
    Transformers renamed evaluation_strategy -> eval_strategy in newer versions.
    Use a try/except so it works across your installed version.
    """
    common = dict(
        output_dir=str(args.output_dir),

        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",

        max_grad_norm=args.max_grad_norm,

        # FP16 mixed precision (AMP)
        bf16=False,
        fp16=True,

        logging_steps=args.logging_steps,

        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        load_best_model_at_end=(args.eval_strategy != "no" and args.save_strategy != "no"),
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,

        report_to="none",
        seed=args.seed,

        remove_unused_columns=False,
        optim="adamw_torch",
    )

    try:
        return TrainingArguments(**common, eval_strategy=args.eval_strategy)
    except TypeError:
        return TrainingArguments(**common, evaluation_strategy=args.eval_strategy)


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.backends.cuda.matmul.allow_tf32 = True

    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    json_commands = bool(args.json_commands) and not bool(args.string_commands)
    system_prompt = read_text(args.system_prompt)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    raw = datasets.load_dataset(
        "json",
        data_files={"train": args.train_jsonl, "validation": args.val_jsonl},
    )

    def preprocess(example: dict) -> dict:
        query = example.get("query") or ""
        cmd = example.get("command")
        cmd_text = to_cmd_text(cmd, json_commands=json_commands)
        prompt_msgs, full_msgs = build_messages(query, cmd_text, system_prompt)
        return tokenize_with_assistant_only_labels(
            tokenizer=tokenizer,
            prompt_msgs=prompt_msgs,
            full_msgs=full_msgs,
            max_length=args.max_length,
        )

    train_ds = raw["train"].map(
        preprocess,
        remove_columns=raw["train"].column_names,
        num_proc=max(1, args.num_workers),
        desc="Tokenizing train",
    )
    val_ds = raw["validation"].map(
        preprocess,
        remove_columns=raw["validation"].column_names,
        num_proc=max(1, args.num_workers),
        desc="Tokenizing val",
    )

    # Base weights in FP16 to save VRAM (they are frozen; only LoRA trains)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    # Required for checkpointing stability
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, peft_cfg)

    # CRITICAL FIX for your GradScaler crash
    force_trainable_fp32(model)

    # Needed for PEFT + some checkpointing flows
    model.enable_input_require_grads()

    # Sanity: should show {torch.float32}
    print("Trainable dtypes:", {p.dtype for p in model.parameters() if p.requires_grad})
    model.print_trainable_parameters()

    training_args = make_training_args(args)
    collator = DataCollatorForCausalLM(tokenizer=tokenizer, pad_to_multiple_of=8)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds if args.eval_strategy != "no" else None,
        data_collator=collator,
    )

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if args.eval_strategy != "no":
        metrics = trainer.evaluate()
        if "eval_loss" in metrics:
            metrics["eval_perplexity"] = math.exp(metrics["eval_loss"]) if metrics["eval_loss"] < 50 else float("inf")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\nSaved LoRA adapter + tokenizer to: {args.output_dir}\n")


if __name__ == "__main__":
    main()
