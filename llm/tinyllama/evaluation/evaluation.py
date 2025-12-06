#!/usr/bin/env python3
"""
Evaluation script (generation-based) for TinyLlama command agent.

Input JSONL schema:
{"query": "...", "command": "..."}

Outputs:
- Writes JSONL with cleaned predictions + metrics per example
- Prints aggregate metrics:
  - exact_match
  - exact_match_normalized
  - avg_char_similarity (SequenceMatcher ratio)
  - avg_edit_similarity (1 - normalized Levenshtein)
  - avg_token_f1 (bag-of-words token F1)
  - avg_jaccard (token set overlap)

Run examples:
python /Storage/LlamaDataset/run_1/evaluation.py \
  --test-jsonl /Storage/LlamaDataset/run_1/test.jsonl \
  --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter-path /app/Output/run_1_lora_best_fp16_bs8 \
  --system-prompt /Storage/LlamaDataset/system_prompt.txt \
  --batch-size 8 \
  --max-length 512 \
  --max-new-tokens 128 \
  --fp16 \
  --output-preds /app/Output/run_1_lora_best_fp16_bs8/preds_test.scored.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import time  # ADDED
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


ASSIST_TOK = "<|assistant|>"
USER_TOK = "<|user|>"
SYS_TOK = "<|system|>"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test-jsonl", required=True)
    p.add_argument("--model-name", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--adapter-path", default=None)
    p.add_argument("--system-prompt", default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--output-preds", required=True)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--keep-first-line",
        action="store_true",
        help="If commands should be single-line, keep only first line.",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Log progress every N examples.",
    )  # ADDED
    return p.parse_args()


def read_text(path: str | None) -> str:
    if not path:
        return ""
    return Path(path).read_text(encoding="utf-8").strip()


def load_jsonl(path: Path, limit: int = 0) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def extract_assistant_only(pred: str) -> str:
    """
    If pred contains chat-template text (system/user/assistant tokens), keep only assistant content.
    """
    s = pred

    if ASSIST_TOK in s:
        s = s.split(ASSIST_TOK, 1)[1]
    elif "Assistant:" in s:
        s = s.split("Assistant:", 1)[1]

    # Cut off any accidental role tokens after assistant
    for cut_tok in (USER_TOK, SYS_TOK, ASSIST_TOK):
        if cut_tok in s:
            s = s.split(cut_tok, 1)[0]

    s = s.replace("<|eot_id|>", "").replace("</s>", "").replace("<|endoftext|>", "")
    return s.strip()


def normalize_cmd(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(";")
    return s


def tokenize(s: str) -> List[str]:
    s = s.strip()
    return s.split() if s else []


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def edit_similarity(a: str, b: str) -> float:
    a = normalize_cmd(a)
    b = normalize_cmd(b)
    d = levenshtein(a, b)
    return 1.0 - (d / max(1, max(len(a), len(b))))


def token_f1(pred_toks: List[str], gold_toks: List[str]) -> Tuple[float, float, float]:
    pc = Counter(pred_toks)
    gc = Counter(gold_toks)
    inter = sum((pc & gc).values())
    p = inter / max(1, len(pred_toks))
    r = inter / max(1, len(gold_toks))
    f1 = 0.0 if (p + r) == 0 else (2 * p * r / (p + r))
    return p, r, f1


def jaccard(pred_toks: List[str], gold_toks: List[str]) -> float:
    sp = set(pred_toks)
    sg = set(gold_toks)
    if not sp and not sg:
        return 1.0
    return len(sp & sg) / max(1, len(sp | sg))


def build_prompt(tokenizer, query: str, system_prompt: str) -> str:
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": query})
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        prefix = f"{system_prompt}\n" if system_prompt else ""
        return f"{prefix}User: {query}\nAssistant:"


@torch.inference_mode()
def generate_batch(model, tokenizer, prompts: List[str], max_new_tokens: int, max_length: int) -> List[str]:
    """
    Correct extraction:
    - tokenize prompts with padding
    - generate
    - for each row, slice out tokens after its actual prompt length (attention_mask sum)
    """
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(model.device)

    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    prompt_lens = enc["attention_mask"].sum(dim=1).tolist()

    preds = []
    for i in range(out.size(0)):
        gen_ids = out[i, prompt_lens[i] :]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        preds.append(text)
    return preds


def main():
    args = parse_args()
    start_time = time.time()  # ADDED

    system_prompt = read_text(args.system_prompt)

    print("Loading tokenizer and model...", flush=True)  # ADDED
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    dtype = torch.float16 if args.fp16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=dtype,
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    print("Model loaded.", flush=True)  # ADDED

    print(f"Loading data from {args.test_jsonl} (limit={args.limit})...", flush=True)  # ADDED
    rows = load_jsonl(Path(args.test_jsonl), limit=args.limit)
    total = len(rows)
    print(f"Loaded {total} examples.", flush=True)  # ADDED

    print("Building prompts...", flush=True)  # ADDED
    queries = [r.get("query", "") for r in rows]
    gold = [str(r.get("command", "")).strip() for r in rows]
    prompts = [build_prompt(tok, q, system_prompt) for q in queries]
    print("Prompts built.", flush=True)  # ADDED

    out_path = Path(args.output_preds)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    exact = 0
    exact_norm = 0
    sum_char_sim = 0.0
    sum_edit_sim = 0.0
    sum_token_f1 = 0.0
    sum_jaccard = 0.0

    processed = 0  # ADDED

    print(
        f"Starting evaluation over {total} examples "
        f"(batch_size={args.batch_size}, log_every={args.log_every})...",
        flush=True,
    )  # ADDED

    with out_path.open("w", encoding="utf-8") as w:
        for i in range(0, total, args.batch_size):
            p_batch = prompts[i : i + args.batch_size]
            q_batch = queries[i : i + args.batch_size]
            g_batch = gold[i : i + args.batch_size]

            preds_raw = generate_batch(model, tok, p_batch, args.max_new_tokens, args.max_length)

            for q, gt, pred in zip(q_batch, g_batch, preds_raw):
                # pred from token slicing should already be assistant-only,
                # but we keep this as a safety net.
                pred_clean = extract_assistant_only(pred)
                if args.keep_first_line:
                    pred_clean = pred_clean.splitlines()[0].strip() if pred_clean else pred_clean

                gt_n = normalize_cmd(gt)
                pr_n = normalize_cmd(pred_clean)

                em = (pred_clean == gt)
                emn = (pr_n == gt_n)

                char_sim = SequenceMatcher(None, pr_n, gt_n).ratio()
                ed_sim = edit_similarity(pred_clean, gt)

                pt = tokenize(pr_n)
                gtoks = tokenize(gt_n)
                _, _, tf1 = token_f1(pt, gtoks)
                jac = jaccard(pt, gtoks)

                exact += int(em)
                exact_norm += int(emn)
                sum_char_sim += char_sim
                sum_edit_sim += ed_sim
                sum_token_f1 += tf1
                sum_jaccard += jac

                w.write(
                    json.dumps(
                        {
                            "query": q,
                            "gold": gt,
                            "pred": pred_clean,
                            "pred_norm": pr_n,
                            "exact": em,
                            "exact_norm": emn,
                            "char_similarity": round(char_sim, 6),
                            "edit_similarity": round(ed_sim, 6),
                            "token_f1": round(tf1, 6),
                            "token_jaccard": round(jac, 6),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                processed += 1  # ADDED

                if processed % args.log_every == 0 or processed == total:  # ADDED
                    elapsed = time.time() - start_time
                    pct = processed / max(1, total) * 100.0
                    print(
                        f"[progress] {processed}/{total} "
                        f"({pct:.1f}%) elapsed={elapsed:.1f}s",
                        flush=True,
                    )

    print(f"examples: {total}")
    print(f"exact_match: {exact / max(1, total):.4f}")
    print(f"exact_match_normalized: {exact_norm / max(1, total):.4f}")
    print(f"avg_char_similarity: {sum_char_sim / max(1, total):.4f}")
    print(f"avg_edit_similarity: {sum_edit_sim / max(1, total):.4f}")
    print(f"avg_token_f1: {sum_token_f1 / max(1, total):.4f}")
    print(f"avg_token_jaccard: {sum_jaccard / max(1, total):.4f}")
    print(f"saved_predictions: {out_path}")
    total_elapsed = time.time() - start_time  # ADDED
    print(f"total_elapsed: {total_elapsed:.1f}s")  # ADDED


if __name__ == "__main__":
    main()
