# DeepSeek LLM (7B Chat)

Minimal runner to download and query `deepseek-ai/deepseek-llm-7b-chat` locally.

## Files
- `deepseek_runner.py` — downloads the model (if missing) and runs a single chat prompt from config.
- `deepseek_runner_config.yaml` — edit repo, local cache path, device/dtype, and prompts.

## Quick start
```bash
# From repo root
python -m DeepSeek.deepseek_runner
```
This will cache the model under `DeepSeek/models/deepseek-llm-7b-chat/`, then print the response.

## Configuration
`deepseek_runner_config.yaml` fields:
- `model.repo_id`: HF repo (default `deepseek-ai/deepseek-llm-7b-chat`).
- `model.local_dir`: where to cache weights.
- `model.device`: `auto` (prefers CUDA), `cuda`, `cuda:0`, or `cpu`.
- `model.dtype`: `auto`/`float16`/`bfloat16`/`float32`.
- `model.max_new_tokens`, `temperature`, `top_p`: generation defaults.
- `model.download_only`: if true, only downloads then exits.
- `run.prompt` / `run.system_prompt`: chat content.
- `run.max_new_tokens`, `run.temperature`, `run.top_p`: optional per-run overrides.

## Notes
- Requires `transformers` and `huggingface_hub` (already used elsewhere in this repo).
- For GPU, ensure the environment has CUDA and enough VRAM; otherwise set `device: cpu`.
