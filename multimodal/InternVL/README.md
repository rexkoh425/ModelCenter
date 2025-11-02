# InternVL3.5-8B Runner (8-bit)

This module mirrors the Qwen runner layout but targets `OpenGVLab/InternVL3_5-8B` with 8-bit weights so it fits on 16 GB GPUs. It wraps Hugging Face Transformers’ custom InternVL codepath and exposes a simple CLI for text-only chatting (images can be added later by wiring in pixel values before calling `model.chat`).

## Files

| File | Purpose |
| --- | --- |
| `internvl8b_runner.py` | Loads tokenizer/model with `load_in_8bit=True`, builds generation config, and issues a chat turn. |
| `internvl8b_runner_config.yaml` | Controls repo id, cache dir, router settings, default prompt/system prompt, etc. |
| `models/` | Created on first run to cache the InternVL weights locally. |

## Requirements

- `transformers>=4.52.1`
- `torch>=2.3` with CUDA 12.x
- `bitsandbytes` (for `load_in_8bit=True`)
- FlashAttention (optional but recommended; disable via `use_flash_attn: false` if kernels are unavailable)

## Run It

```bash
python -m InternVL.internvl8b_runner --config InternVL/internvl8b_runner_config.yaml \
    --prompt "Write a SQL query counting completed orders per week."
```

The script:
1. Downloads / caches the checkpoint under `InternVL/models/InternVL3_5-8B`.
2. Loads the model in 8-bit with `device_map="auto"` so it can span multiple GPUs if available.
3. Calls `model.chat(...)` for a text-only prompt and prints the response.

To add image support later, feed a tensor into `pixel_values` before calling `runner.chat`. The README from the InternVL repo (mirrored in `internvl_readme.md`) contains helper functions such as `dynamic_preprocess` and `load_image` for building those tensors.

