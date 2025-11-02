"""
Utility class to download and run the Qwen3-8B model locally.

Usage:
    # Edit Qwen3/qwen3_runner_config.yaml, then run:
    python -m Qwen3.qwen3_runner
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml


DEFAULT_REPO_ID = "Qwen/Qwen3-8B-Instruct"
DEFAULT_CONFIG_FILENAME = "qwen3_runner_config.yaml"
DEFAULT_CONFIG_PATH = Path(__file__).with_name(DEFAULT_CONFIG_FILENAME)
DEFAULT_LOCAL_MODEL_DIR = Path(__file__).resolve().parent / "models" / "Qwen3-8B-Instruct"


@dataclass(slots=True)
class Qwen3Config:
    repo_id: str = DEFAULT_REPO_ID
    local_dir: Path = DEFAULT_LOCAL_MODEL_DIR
    revision: Optional[str] = None
    device: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    download_only: bool = False


@dataclass(slots=True)
class Qwen3PromptSettings:
    prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None


def load_app_config(config_path: Path) -> tuple[Qwen3Config, Qwen3PromptSettings]:
    expanded = Path(config_path).expanduser()
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

    def coerce_bool(value, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y", "on"}:
                return True
            if lowered in {"false", "0", "no", "n", "off"}:
                return False
        return bool(value)

    def coerce_int(value, default: Optional[int] = None) -> int:
        if value is None:
            if default is None:
                raise ValueError("Missing required integer value.")
            return default
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid integer value: {value}") from exc

    def coerce_optional_int(value) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid integer value: {value}") from exc

    def coerce_float(value, default: Optional[float] = None) -> float:
        if value is None:
            if default is None:
                raise ValueError("Missing required float value.")
            return default
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid float value: {value}") from exc

    def coerce_optional_float(value) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid float value: {value}") from exc

    local_dir_value = model_section.get("local_dir")
    local_dir = (
        Path(str(local_dir_value)).expanduser().resolve()
        if local_dir_value
        else DEFAULT_LOCAL_MODEL_DIR
    )

    qwen_config = Qwen3Config(
        repo_id=str(model_section.get("repo_id", DEFAULT_REPO_ID)),
        local_dir=local_dir,
        revision=str(model_section["revision"]).strip() if model_section.get("revision") else None,
        device=str(model_section.get("device", "auto")),
        dtype=str(model_section.get("dtype", "auto")),
        max_new_tokens=coerce_int(model_section.get("max_new_tokens"), 512),
        temperature=coerce_float(model_section.get("temperature"), 0.7),
        top_p=coerce_float(model_section.get("top_p"), 0.9),
        download_only=coerce_bool(model_section.get("download_only"), False),
    )

    prompt_settings = Qwen3PromptSettings(
        prompt=str(run_section["prompt"]).strip() if run_section.get("prompt") else None,
        system_prompt=str(run_section["system_prompt"]).strip()
        if run_section.get("system_prompt")
        else None,
        max_new_tokens=coerce_optional_int(run_section.get("max_new_tokens")),
        temperature=coerce_optional_float(run_section.get("temperature")),
        top_p=coerce_optional_float(run_section.get("top_p")),
    )

    return qwen_config, prompt_settings


class Qwen3Runner:
    """Downloads Qwen3 weights (if needed) and provides a text generation helper."""

    def __init__(self, config: Optional[Qwen3Config] = None) -> None:
        self.config = config or Qwen3Config()
        self.local_dir = Path(self.config.local_dir).expanduser()
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self._ensure_local_weights()
        self.device = self._resolve_device(self.config.device)
        self.torch_dtype = self._resolve_dtype(self.config.dtype)

        self.tokenizer = None
        self.model = None
        if not self.config.download_only:
            self._load_pipeline()

    @staticmethod
    def _resolve_device(device_pref: str) -> torch.device:
        if device_pref == "auto":
            return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        return torch.device(device_pref)

    def _resolve_dtype(self, dtype_pref: str) -> torch.dtype:
        if dtype_pref == "auto":
            if self.device.type == "cuda":
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                return torch.float16
            return torch.float32
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        lowered = dtype_pref.lower()
        if lowered not in mapping:
            raise ValueError(f"Unsupported dtype: {dtype_pref}")
        return mapping[lowered]

    def _ensure_local_weights(self) -> Path:
        """Download the model snapshot if it is not already materialised."""
        markers = ["config.json", "model.safetensors", "model.bin", "pytorch_model.bin"]
        if any((self.local_dir / marker).exists() for marker in markers):
            return self.local_dir

        snapshot_download(
            repo_id=self.config.repo_id,
            local_dir=self.local_dir,
            local_dir_use_symlinks=False,
            revision=self.config.revision,
        )
        return self.local_dir

    def _load_pipeline(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            device_map="auto" if self.device.type == "cuda" else None,
        )
        if self.device.type == "cpu":
            self.model.to(self.device)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        if self.config.download_only:
            raise RuntimeError("Qwen3Runner was initialised in download-only mode.")
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded; set download_only=False or call _load_pipeline().")

        max_tokens = max_new_tokens or self.config.max_new_tokens
        sampling_temp = temperature or self.config.temperature
        sampling_top_p = top_p or self.config.top_p

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)

        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        output = self.model.generate(
            inputs,
            max_new_tokens=max_tokens,
            temperature=sampling_temp,
            top_p=sampling_top_p,
            do_sample=True,
            pad_token_id=pad_token_id,
        )
        prompt_length = inputs.shape[-1]
        generated_tokens = output[0, prompt_length:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


class Qwen3RunnerApp:
    """High-level wrapper that loads YAML config and executes prompts."""

    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
        self.model_config: Optional[Qwen3Config] = None
        self.prompt_settings: Optional[Qwen3PromptSettings] = None

    def load(self) -> None:
        self.model_config, self.prompt_settings = load_app_config(self.config_path)

    def run(self) -> None:
        if self.model_config is None or self.prompt_settings is None:
            self.load()
        assert self.model_config is not None
        assert self.prompt_settings is not None

        runner = Qwen3Runner(self.model_config)
        if self.model_config.download_only:
            print(f"Model files available at: {runner.model_path}")
            return

        if not self.prompt_settings.prompt:
            raise ValueError("Config field 'run.prompt' is required when download_only is false.")

        response = runner.generate(
            prompt=self.prompt_settings.prompt,
            system_prompt=self.prompt_settings.system_prompt,
            max_new_tokens=self.prompt_settings.max_new_tokens,
            temperature=self.prompt_settings.temperature,
            top_p=self.prompt_settings.top_p,
        )
        print(response)


def main() -> None:
    app = Qwen3RunnerApp()
    try:
        app.run()
    except KeyboardInterrupt:  # pragma: no cover - user interrupt
        print("\nInterrupted by user.")
        sys.exit(130)
    except Exception as exc:  # pragma: no cover - runtime errors
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
