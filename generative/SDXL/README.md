# SDXL Runner (16 GB Friendly)

This module mirrors the other project folders and wraps `stabilityai/stable-diffusion-xl-base-1.0` so you can prompt SDXL on a single 16 GB GPU. It exposes a YAML-driven CLI runner that:

1. Loads the base SDXL pipeline (bf16/fp16) with optional xFormers memory-efficient attention.
2. Optionally chains the official SDXL refiner for a second pass.
3. Saves generated PNGs into an `outputs/` directory with prompt-based filenames.

## Requirements

- Python 3.10+
- `torch>=2.2` w/ CUDA 12.x (bf16 or fp16 on 16 GB GPUs like RTX 4080/4090/6000 Ada)
- `diffusers>=0.29`, `transformers>=4.42`, `accelerate`, `safetensors`, `xformers` (recommended)
- Plenty of disk space (~13 GB for base weights + 13 GB if you enable refiner)

Install deps (example):

```bash
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers[torch] transformers accelerate safetensors xformers
```

## Configuration

Edit `sdxl_runner_config.yaml`:

- `model.base_model_id` – change to a community mix if needed.
- `model.enable_refiner` – flip to `true` once you’ve downloaded the refiner weights (optional on 16 GB; keep steps small).
- `run.prompt` / `negative_prompt` / `width` / `height` – defaults are 1024² (fits in ~13 GB VRAM with xFormers).
- `run.seed` – set to lock outputs across runs.
- `run.output_dir` – images land here; directories are created automatically.

## Usage

```bash
python -m SDXL.sdxl_runner --config SDXL/sdxl_runner_config.yaml \
    --prompt "Futuristic EV interior, driver POV, neon reflections, ultra detailed"
```

- Omit `--prompt` to use the config’s prompt (or type interactively).
- Add `enable_refiner: true` once you’re comfortable with the VRAM overhead (still within 16 GB if you lower steps or run refiner sequentially).

## Tips for 16 GB Cards

- Keep resolution ≤1024 on single pass; push higher via tiled/hires fixes after saving.
- Leave `torch_dtype: bfloat16` (or `float16`) and ensure `use_xformers: true` to stay under the 16 GB ceiling.
- For batch sizes >1 or ControlNet, consider `torch_compile` or `--medvram` settings in your UI if integrating elsewhere.

This folder stays self-contained, so you can treat it like the other project modules (BLIP, Qwen3, etc.)—configure YAML, run the module, and capture artifacts under `SDXL/outputs`. Feel free to extend the runner with ControlNet hooks, prompt schedulers, or LoRA merging for more advanced workflows.
