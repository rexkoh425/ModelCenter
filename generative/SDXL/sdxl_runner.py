"""
Utility to load Stable Diffusion XL Base 1.0 and render prompts on a 16 GB GPU.

Example:
    python -m SDXL.sdxl_runner --config SDXL/sdxl_runner_config.yaml \\
        --prompt "A futuristic concept car cockpit, hyperrealistic, golden hour."
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import yaml
from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline


DEFAULT_CONFIG_FILENAME = "sdxl_runner_config.yaml"
DEFAULT_CONFIG_PATH = Path(__file__).with_name(DEFAULT_CONFIG_FILENAME)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


@dataclass(slots=True)
class SDXLModelConfig:
    base_model_id: str
    refiner_model_id: Optional[str]
    enable_refiner: bool
    torch_dtype: str
    use_xformers: bool
    guidance_scale: float
    num_inference_steps: int
    refiner_guidance_scale: float
    refiner_steps: int
    variant: Optional[str]


@dataclass(slots=True)
class SDXLRunConfig:
    prompt: Optional[str]
    negative_prompt: Optional[str]
    width: int
    height: int
    num_images: int
    seed: Optional[int]
    output_dir: Path
    prefix: str
    auto_filename: bool


def load_config(config_path: Path) -> tuple[SDXLModelConfig, SDXLRunConfig]:
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

    model_config = SDXLModelConfig(
        base_model_id=str(model_section.get("base_model_id", "stabilityai/stable-diffusion-xl-base-1.0")),
        refiner_model_id=model_section.get("refiner_model_id"),
        enable_refiner=bool(model_section.get("enable_refiner", False)),
        torch_dtype=str(model_section.get("torch_dtype", "bfloat16")),
        use_xformers=bool(model_section.get("use_xformers", True)),
        guidance_scale=float(model_section.get("guidance_scale", 7.5)),
        num_inference_steps=int(model_section.get("num_inference_steps", 30)),
        refiner_guidance_scale=float(model_section.get("refiner_guidance_scale", 5.0)),
        refiner_steps=int(model_section.get("refiner_steps", 10)),
        variant=model_section.get("variant"),
    )

    output_dir_value = run_section.get("output_dir")
    output_dir = (
        Path(output_dir_value).expanduser().resolve()
        if output_dir_value
        else DEFAULT_OUTPUT_DIR
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = SDXLRunConfig(
        prompt=run_section.get("prompt"),
        negative_prompt=run_section.get("negative_prompt"),
        width=int(run_section.get("width", 1024)),
        height=int(run_section.get("height", 1024)),
        num_images=int(run_section.get("num_images", 1)),
        seed=int(run_section["seed"]) if run_section.get("seed") is not None else None,
        output_dir=output_dir,
        prefix=str(run_section.get("prefix", "sdxl")),
        auto_filename=bool(run_section.get("auto_filename", True)),
    )
    return model_config, run_config


class SDXLRunner:
    def __init__(self, model_config: SDXLModelConfig):
        self.model_config = model_config
        dtype = self._resolve_dtype(model_config.torch_dtype)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.base_pipe = StableDiffusionXLPipeline.from_pretrained(
            model_config.base_model_id,
            torch_dtype=dtype,
            variant=model_config.variant,
        )
        if model_config.use_xformers and hasattr(self.base_pipe, "enable_xformers_memory_efficient_attention"):
            self.base_pipe.enable_xformers_memory_efficient_attention()
        self.base_pipe.to(self.device)

        self.refiner_pipe: Optional[StableDiffusionXLImg2ImgPipeline] = None
        if model_config.enable_refiner and model_config.refiner_model_id:
            self.refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                model_config.refiner_model_id,
                text_encoder_2=self.base_pipe.text_encoder_2,
                vae=self.base_pipe.vae,
                torch_dtype=dtype,
                variant=model_config.variant,
            )
            if model_config.use_xformers and hasattr(self.refiner_pipe, "enable_xformers_memory_efficient_attention"):
                self.refiner_pipe.enable_xformers_memory_efficient_attention()
            self.refiner_pipe.to(self.device)

    @staticmethod
    def _resolve_dtype(value: str) -> torch.dtype:
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        lowered = (value or "bfloat16").lower()
        if lowered not in mapping:
            raise ValueError(f"Unsupported torch_dtype: {value}")
        return mapping[lowered]

    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: Optional[str],
        width: int,
        height: int,
        num_images: int,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        refiner_guidance_scale: Optional[float] = None,
        refiner_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        if width % 8 != 0 or height % 8 != 0:
            raise ValueError("Width and height must be multiples of 8 for SDXL.")
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        gs = guidance_scale or self.model_config.guidance_scale
        steps = num_inference_steps or self.model_config.num_inference_steps

        output_type = "latent" if self.refiner_pipe else "pil"
        base_result = self.base_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=gs,
            num_images_per_prompt=num_images,
            generator=generator,
            output_type=output_type,
        )

        if not self.refiner_pipe:
            return base_result.images

        latents = base_result.images
        refiner_gs = refiner_guidance_scale or self.model_config.refiner_guidance_scale
        refiner_steps = refiner_steps or self.model_config.refiner_steps
        refined = self.refiner_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=latents,
            guidance_scale=refiner_gs,
            num_inference_steps=refiner_steps,
            generator=generator,
        )
        return refined.images


class SDXLApp:
    def __init__(self, config_path: Path | None = None) -> None:
        resolved = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
        self.model_config, self.run_config = load_config(resolved)
        self.runner = SDXLRunner(self.model_config)

    def run(self, prompt_override: Optional[str] = None) -> None:
        prompt = (prompt_override or self.run_config.prompt or "").strip()
        if not prompt:
            prompt = input("Enter a prompt for SDXL: ").strip()
        if not prompt:
            raise ValueError("A prompt is required.")

        images = self.runner.generate(
            prompt=prompt,
            negative_prompt=self.run_config.negative_prompt,
            width=self.run_config.width,
            height=self.run_config.height,
            num_images=self.run_config.num_images,
            seed=self.run_config.seed,
        )
        self._save_images(images, prompt)

    def _save_images(self, images, prompt: str) -> None:
        for idx, image in enumerate(images):
            filename = self._build_filename(prompt, idx)
            path = self.run_config.output_dir / filename
            image.save(path)
            print(f"Saved {path}")

    def _build_filename(self, prompt: str, index: int) -> str:
        safe_prompt = "".join(ch for ch in prompt.lower().strip().replace(" ", "_") if ch.isalnum() or ch in {"_", "-"})
        safe_prompt = safe_prompt[:60] or "prompt"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = f"{self.run_config.prefix}_{timestamp}_{safe_prompt}_{index}".rstrip("_")
        if not self.run_config.auto_filename:
            base = f"{self.run_config.prefix}_{index}"
        return f"{base}.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stable Diffusion XL Base 1.0")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to sdxl_runner_config.yaml",
    )
    parser.add_argument("--prompt", type=str, help="Override run.prompt from the config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        app = SDXLApp(args.config)
        app.run(prompt_override=args.prompt)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

