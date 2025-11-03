"""
Utility script to download and run deepseek-ai/deepseek-llm-7b-chat.

Usage:
    # (Optional) edit DeepSeek/deepseek_runner_config.yaml
    python -m DeepSeek.deepseek_runner
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml


DEFAULT_REPO_ID = "deepseek-ai/deepseek-llm-7b-chat"
DEFAULT_CONFIG_FILENAME = "deepseek_runner_config.yaml"
DEFAULT_CONFIG_PATH = Path(__file__).with_name(DEFAULT_CONFIG_FILENAME)
DEFAULT_LOCAL_MODEL_DIR = Path(__file__).resolve().parent / "models" / "deepseek-llm-7b-chat"


@dataclass(slots=True)
class DeepSeekModelConfig:
    repo_id: str = DEFAULT_REPO_ID
    local_dir: Path = DEFAULT_LOCAL_MODEL_DIR
    revision: Optional[str] = None
    device: str = "auto"  # "auto", "cuda", "cuda:0", "cpu"
    dtype: str = "auto"  # "auto", "float16", "bfloat16", "float32"
    download_only: bool = False
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass(slots=True)
class DeepSeekRunConfig:
    prompt: str = "Tell me a short fun fact about space."
    system_prompt: Optional[str] = "You are a concise, helpful assistant."
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None


def _coerce_dtype(name: str) -> Optional[torch.dtype]:
    if not name or str(name).lower() == "auto":
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(str(name).lower())


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> tuple[DeepSeekModelConfig, DeepSeekRunConfig]:
    cfg_path = Path(path).expanduser()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    data = yaml.safe_load(cfg_path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")
    model_section = data.get("model", {}) or {}
    run_section = data.get("run", {}) or {}
    model_cfg = DeepSeekModelConfig(
        repo_id=model_section.get("repo_id", DEFAULT_REPO_ID),
        local_dir=Path(model_section.get("local_dir", DEFAULT_LOCAL_MODEL_DIR)).expanduser(),
        revision=model_section.get("revision") or None,
        device=str(model_section.get("device", "auto")),
        dtype=str(model_section.get("dtype", "auto")),
        download_only=bool(model_section.get("download_only", False)),
        max_new_tokens=int(model_section.get("max_new_tokens", 512)),
        temperature=float(model_section.get("temperature", 0.7)),
        top_p=float(model_section.get("top_p", 0.9)),
    )
    run_cfg = DeepSeekRunConfig(
        prompt=str(run_section.get("prompt", DeepSeekRunConfig.prompt)),
        system_prompt=run_section.get("system_prompt") or None,
        max_new_tokens=run_section.get("max_new_tokens"),
        temperature=run_section.get("temperature"),
        top_p=run_section.get("top_p"),
    )
    return model_cfg, run_cfg


def ensure_model_present(cfg: DeepSeekModelConfig) -> Path:
    cfg.local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=cfg.repo_id,
        local_dir=cfg.local_dir,
        local_dir_use_symlinks=False,
        revision=cfg.revision,
        allow_patterns=["*.bin", "*.pt", "*.json", "*.model", "*.txt", "*.py", "*.md"],
    )
    return cfg.local_dir


def _select_device(device: str) -> str:
    device = str(device or "auto").lower()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def build_prompt_messages(prompt: str, system_prompt: Optional[str]) -> list[dict]:
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def generate_chat(
    model,
    tokenizer,
    messages: Sequence[dict],
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs.shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    model_cfg, run_cfg = load_config(DEFAULT_CONFIG_PATH)
    local_dir = ensure_model_present(model_cfg)
    if model_cfg.download_only:
        print(f"Model downloaded to: {local_dir}")
        return

    device_name = _select_device(model_cfg.device)
    dtype = _coerce_dtype(model_cfg.dtype)
    device = torch.device(device_name)

    tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    if device.type != "cuda":
        model.to(device)
    model.eval()

    messages = build_prompt_messages(run_cfg.prompt, run_cfg.system_prompt)
    max_new_tokens = run_cfg.max_new_tokens or model_cfg.max_new_tokens
    temperature = run_cfg.temperature if run_cfg.temperature is not None else model_cfg.temperature
    top_p = run_cfg.top_p if run_cfg.top_p is not None else model_cfg.top_p

    response = generate_chat(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    print("\n--- Model response ---")
    print(response)


if __name__ == "__main__":
    main()
