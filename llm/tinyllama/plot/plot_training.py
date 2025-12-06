#!/usr/bin/env python3
"""
Plot TinyLlama fine-tuning curves from Hugging Face Trainer's trainer_state.json.

Usage:
  python plot_trainer_state.py --input /path/to/run_or_trainer_state.json --out-dir plots --show

Outputs:
  - loss_curves.png (train loss + eval loss)
  - loss_curves_ppl.png (optional, if --plot-ppl)
  - lr_curve.png (if learning_rate exists)
  - grad_norm.png (if grad_norm exists)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def moving_average(xs: List[float], window: int) -> List[float]:
    if window <= 1 or not xs:
        return xs[:]
    out: List[float] = []
    q: List[float] = []
    s = 0.0
    for v in xs:
        q.append(v)
        s += v
        if len(q) > window:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def safe_exp(x: float) -> float:
    x = max(min(x, 100.0), -100.0)
    return float(math.exp(x))


def find_trainer_state(inp: Path) -> Path:
    if inp.is_file():
        if inp.name == "trainer_state.json":
            return inp
        raise FileNotFoundError(f"Expected trainer_state.json, got file: {inp}")
    cand = inp / "trainer_state.json"
    if cand.exists():
        return cand
    # fallback: search nested
    for p in inp.rglob("trainer_state.json"):
        return p
    raise FileNotFoundError(f"Could not find trainer_state.json under: {inp}")


def load_log_history(trainer_state_path: Path) -> List[Dict[str, Any]]:
    data = read_json(trainer_state_path)
    if not isinstance(data, dict) or "log_history" not in data or not isinstance(data["log_history"], list):
        raise ValueError(f"Invalid HF trainer_state.json (missing log_history): {trainer_state_path}")
    return [r for r in data["log_history"] if isinstance(r, dict)]


def ensure_step(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    HF Trainer logs usually have step or global_step. Ensure every record has int 'step'.
    """
    for r in records:
        if "step" in r and isinstance(r["step"], float):
            r["step"] = int(r["step"])
        if "step" not in r:
            gs = r.get("global_step")
            if isinstance(gs, (int, float)):
                r["step"] = int(gs)

    # If still missing, enumerate (rare)
    if not any(isinstance(r.get("step"), (int, float)) for r in records):
        for i, r in enumerate(records):
            r["step"] = i

    return records


def sort_by_step(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rs = [r for r in records if isinstance(r.get("step"), (int, float))]
    rs.sort(key=lambda r: float(r["step"]))
    for r in rs:
        if isinstance(r["step"], float):
            r["step"] = int(r["step"])
    return rs


def extract_series(records: List[Dict[str, Any]], key: str) -> Tuple[List[int], List[float]]:
    xs: List[int] = []
    ys: List[float] = []
    for r in records:
        if key not in r:
            continue
        step = r.get("step")
        v = r.get(key)
        if not isinstance(step, (int, float)):
            continue
        if not isinstance(v, (int, float)):
            continue
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            continue
        xs.append(int(step))
        ys.append(float(v))
    return xs, ys


def savefig(path: Path, dpi: int = 160) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def summarize(records: List[Dict[str, Any]]) -> str:
    records = sort_by_step(ensure_step(records))
    x_tr, y_tr = extract_series(records, "loss")
    x_ev, y_ev = extract_series(records, "eval_loss")

    parts = []
    if x_tr:
        parts.append(f"train last: step={x_tr[-1]}, loss={y_tr[-1]:.4f}")
    if x_ev:
        best = min(y_ev)
        best_step = x_ev[y_ev.index(best)]
        parts.append(f"eval last: step={x_ev[-1]}, loss={y_ev[-1]:.4f} | best: {best:.4f} @ step={best_step}")
    return " | ".join(parts) if parts else "No loss/eval_loss found in log_history."


def plot_all(
    records: List[Dict[str, Any]],
    out_dir: Path,
    smooth: int,
    plot_ppl: bool,
    title: str,
) -> None:
    records = sort_by_step(ensure_step(records))

    # Loss curves
    x_tr, y_tr = extract_series(records, "loss")
    x_ev, y_ev = extract_series(records, "eval_loss")

    y_tr_s = moving_average(y_tr, smooth)
    y_ev_s = moving_average(y_ev, max(1, min(smooth, 5)))

    plt.figure()
    if x_tr:
        plt.plot(x_tr, y_tr_s, label=f"train loss (ma{smooth})" if smooth > 1 else "train loss")
    if x_ev:
        plt.plot(x_ev, y_ev_s, label="eval loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    savefig(out_dir / "loss_curves.png")

    # Optional perplexity
    if plot_ppl and (x_tr or x_ev):
        plt.figure()
        if x_tr:
            ppl_tr = [safe_exp(v) for v in y_tr_s]
            plt.plot(x_tr, ppl_tr, label="train ppl (exp(loss))")
        if x_ev:
            ppl_ev = [safe_exp(v) for v in y_ev_s]
            plt.plot(x_ev, ppl_ev, label="eval ppl (exp(loss))")
        plt.xlabel("step")
        plt.ylabel("perplexity")
        plt.yscale("log")
        plt.title(title + " (perplexity, log scale)")
        plt.grid(True, alpha=0.25, which="both")
        plt.legend()
        savefig(out_dir / "loss_curves_ppl.png")

    # Learning rate
    x_lr, y_lr = extract_series(records, "learning_rate")
    if x_lr:
        plt.figure()
        plt.plot(x_lr, y_lr, label="learning_rate")
        plt.xlabel("step")
        plt.ylabel("lr")
        plt.title(title)
        plt.grid(True, alpha=0.25)
        plt.legend()
        savefig(out_dir / "lr_curve.png")

    # Grad norm
    x_gn, y_gn = extract_series(records, "grad_norm")
    if x_gn:
        y_gn_s = moving_average(y_gn, max(1, min(smooth, 20)))
        plt.figure()
        plt.plot(x_gn, y_gn_s, label="grad_norm" + (f" (ma{min(smooth,20)})" if smooth > 1 else ""))
        plt.xlabel("step")
        plt.ylabel("grad_norm")
        plt.title(title)
        plt.grid(True, alpha=0.25)
        plt.legend()
        savefig(out_dir / "grad_norm.png")


def show_interactive(records: List[Dict[str, Any]], smooth: int, plot_ppl: bool, title: str) -> None:
    records = sort_by_step(ensure_step(records))

    x_tr, y_tr = extract_series(records, "loss")
    x_ev, y_ev = extract_series(records, "eval_loss")
    y_tr_s = moving_average(y_tr, smooth)
    y_ev_s = moving_average(y_ev, max(1, min(smooth, 5)))

    plt.figure()
    if x_tr:
        plt.plot(x_tr, y_tr_s, label=f"train loss (ma{smooth})" if smooth > 1 else "train loss")
    if x_ev:
        plt.plot(x_ev, y_ev_s, label="eval loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()

    x_lr, y_lr = extract_series(records, "learning_rate")
    if x_lr:
        plt.figure()
        plt.plot(x_lr, y_lr, label="learning_rate")
        plt.xlabel("step")
        plt.ylabel("lr")
        plt.title(title)
        plt.grid(True, alpha=0.25)
        plt.legend()

    x_gn, y_gn = extract_series(records, "grad_norm")
    if x_gn:
        y_gn_s = moving_average(y_gn, max(1, min(smooth, 20)))
        plt.figure()
        plt.plot(x_gn, y_gn_s, label="grad_norm")
        plt.xlabel("step")
        plt.ylabel("grad_norm")
        plt.title(title)
        plt.grid(True, alpha=0.25)
        plt.legend()

    if plot_ppl and (x_tr or x_ev):
        plt.figure()
        if x_tr:
            plt.plot(x_tr, [safe_exp(v) for v in y_tr_s], label="train ppl")
        if x_ev:
            plt.plot(x_ev, [safe_exp(v) for v in y_ev_s], label="eval ppl")
        plt.yscale("log")
        plt.xlabel("step")
        plt.ylabel("perplexity")
        plt.title(title + " (perplexity)")
        plt.grid(True, alpha=0.25, which="both")
        plt.legend()

    plt.show()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Run dir containing trainer_state.json OR path to trainer_state.json")
    ap.add_argument("--out-dir", default="plots", help="Where to save PNG plots")
    ap.add_argument("--smooth", type=int, default=25, help="Moving average window for train loss smoothing")
    ap.add_argument("--plot-ppl", action="store_true", help="Also plot perplexity=exp(loss) (log scale)")
    ap.add_argument("--show", action="store_true", help="Show interactive plots too")
    args = ap.parse_args()

    inp = Path(args.input).expanduser()
    out_dir = Path(args.out_dir).expanduser()

    ts_path = find_trainer_state(inp)
    records = load_log_history(ts_path)

    title = f"HF Trainer curves\n{ts_path}"
    print(f"[loaded] {ts_path}")
    print(f"[summary] {summarize(records)}")

    plot_all(records, out_dir, smooth=max(1, args.smooth), plot_ppl=bool(args.plot_ppl), title=title)
    print(f"[saved] {out_dir}")

    if args.show:
        show_interactive(records, smooth=max(1, args.smooth), plot_ppl=bool(args.plot_ppl), title=title)


if __name__ == "__main__":
    main()
