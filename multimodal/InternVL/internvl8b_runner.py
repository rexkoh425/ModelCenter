"""
Utility script to download and run InternVL3.5-8B in 8-bit precision.

Usage:
    python -m InternVL.internvl8b_runner --config InternVL/internvl8b_runner_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import yaml
from transformers import AutoModel, AutoTokenizer


DEFAULT_REPO_ID = "OpenGVLab/InternVL3_5-8B"
DEFAULT_CONFIG_FILENAME = "internvl8b_runner_config.yaml"
DEFAULT_CONFIG_PATH = Path(__file__).with_name(DEFAULT_CONFIG_FILENAME)
DEFAULT_LOCAL_MODEL_DIR = Path(__file__).resolve().parent / "models" / "InternVL3_5-8B"


@dataclass(slots=True)
class InternVLModelConfig:
    repo_id: str = DEFAULT_REPO_ID
    local_dir: Path = DEFAULT_LOCAL_MODEL_DIR
    revision: Optional[str] = None
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    load_in_8bit: bool = True
    low_cpu_mem_usage: bool = True
    use_flash_attn: bool = True


@dataclass(slots=True)
class InternVLRunSettings:
    prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    max_new_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 0.9
    do_sample: bool = True
    num_beams: Optional[int] = None


def load_internvl_config(config_path: Path) -> Tuple[InternVLModelConfig, InternVLRunSettings]:
    expanded = Path(config_path).expanduser().resolve()
    if not expanded.is_file():
        raise FileNotFoundError(f"Config file not found: {expanded}")

    with expanded.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")

    model_section = data.get("model", {})
    if not isinstance(model_section, dict):
        raise ValueError("Config section 'model' must be a mapping.")

    run_section = data.get("run", {})
    if not isinstance(run_section, dict):
        raise ValueError("Config section 'run' must be a mapping.")

    local_dir_value = model_section.get("local_dir")
    local_dir = (
        Path(str(local_dir_value)).expanduser().resolve()
        if local_dir_value
        else DEFAULT_LOCAL_MODEL_DIR
    )
    local_dir.mkdir(parents=True, exist_ok=True)

    model_config = InternVLModelConfig(
        repo_id=str(model_section.get("repo_id", DEFAULT_REPO_ID)),
        local_dir=local_dir,
        revision=str(model_section["revision"]).strip() if model_section.get("revision") else None,
        device_map=str(model_section.get("device_map", "auto")),
        torch_dtype=str(model_section.get("torch_dtype", "bfloat16")),
        load_in_8bit=bool(model_section.get("load_in_8bit", True)),
        low_cpu_mem_usage=bool(model_section.get("low_cpu_mem_usage", True)),
        use_flash_attn=bool(model_section.get("use_flash_attn", True)),
    )

    run_config = InternVLRunSettings(
        prompt=run_section.get("prompt"),
        system_prompt=run_section.get("system_prompt"),
        max_new_tokens=int(run_section.get("max_new_tokens", 512)),
        temperature=float(run_section.get("temperature", 0.6)),
        top_p=float(run_section.get("top_p", 0.9)),
        do_sample=bool(run_section.get("do_sample", True)),
        num_beams=int(run_section["num_beams"]) if run_section.get("num_beams") else None,
    )

    return model_config, run_config


class InternVL8BRunner:
    """High-level helper to run InternVL3.5-8B in 8-bit precision."""

    def __init__(
        self,
        config: InternVLModelConfig,
        generation_defaults: Optional[InternVLRunSettings] = None,
    ) -> None:
        self.config = config
        self.defaults = generation_defaults or InternVLRunSettings()
        self.cache_dir = self.config.local_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        dtype = self._resolve_dtype(self.config.torch_dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.repo_id,
            cache_dir=self.cache_dir,
            revision=self.config.revision,
            trust_remote_code=True,
            use_fast=False,
        )
        self.model = AutoModel.from_pretrained(
            self.config.repo_id,
            cache_dir=self.cache_dir,
            revision=self.config.revision,
            torch_dtype=dtype,
            load_in_8bit=self.config.load_in_8bit,
            low_cpu_mem_usage=self.config.low_cpu_mem_usage,
            use_flash_attn=self.config.use_flash_attn,
            device_map=self.config.device_map,
            trust_remote_code=True,
        ).eval()

    @staticmethod
    def _resolve_dtype(name: str) -> torch.dtype:
        lookup = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        key = (name or "bfloat16").lower()
        if key not in lookup:
            raise ValueError(f"Unsupported dtype: {name}")
        return lookup[key]

    def chat(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
        num_beams: Optional[int] = None,
        history=None,
    ) -> tuple[str, list]:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt is empty.")

        merged_prompt = prompt.strip()
        if system_prompt:
            merged_prompt = f"{system_prompt.strip()}\n{merged_prompt}"

        generation_config = {
            "max_new_tokens": max_new_tokens or self.defaults.max_new_tokens,
            "temperature": temperature if temperature is not None else self.defaults.temperature,
            "top_p": top_p if top_p is not None else self.defaults.top_p,
            "do_sample": do_sample if do_sample is not None else self.defaults.do_sample,
        }
        beams = num_beams or self.defaults.num_beams
        if beams:
            generation_config["num_beams"] = beams

        response, updated_history = self.model.chat(
            tokenizer=self.tokenizer,
            pixel_values=None,
            question=merged_prompt,
            generation_config=generation_config,
            history=history,
            return_history=True,
        )
        return response, updated_history


class InternVLRunnerApp:
    """CLI wrapper that loads YAML config and executes a chat turn."""

    def __init__(self, config_path: Path | None = None) -> None:
        resolved = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
        self.config_path = resolved
        self.model_config, self.run_settings = load_internvl_config(self.config_path)
        self.runner = InternVL8BRunner(self.model_config, self.run_settings)

    def run(self, prompt_override: Optional[str] = None) -> None:
        prompt = (prompt_override or self.run_settings.prompt or "").strip()
        if not prompt:
            prompt = input("Enter a prompt for InternVL3.5-8B: ").strip()
        if not prompt:
            raise ValueError("A prompt is required.")

        response, _ = self.runner.chat(
            prompt=prompt,
            system_prompt=self.run_settings.system_prompt,
            max_new_tokens=self.run_settings.max_new_tokens,
            temperature=self.run_settings.temperature,
            top_p=self.run_settings.top_p,
            do_sample=self.run_settings.do_sample,
            num_beams=self.run_settings.num_beams,
        )
        print(response)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run InternVL3.5-8B in 8-bit mode.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to internvl8b_runner_config.yaml",
    )
    parser.add_argument("--prompt", type=str, help="Override run.prompt from the config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        app = InternVLRunnerApp(args.config)
        app.run(prompt_override=args.prompt)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
    except Exception as exc:  # pragma: no cover - runtime errors
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
