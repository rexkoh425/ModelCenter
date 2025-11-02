# Qwen3-8B Runner

This folder keeps everything needed to materialise and run the `Qwen/Qwen3-8B-Instruct` model.
Weights land in `Qwen3/models/Qwen3-8B-Instruct` so they stay isolated from the BLIP workflows.

> **Note:** Qwen models are gated on Hugging Face. Make sure you have accepted the license and that
> your `HF_TOKEN` environment variable (or `huggingface-cli login`) is configured before downloading.

The runner reads `qwen3_runner_config.yaml`. It has two sections:

- `model`: Hugging Face identifiers, device, dtype, download-only toggle, and default decoding params.
- `run`: The actual prompt/system prompt plus optional overrides for `max_new_tokens`, `temperature`, and `top_p`.

## Configure & Download

1. Edit `Qwen3/qwen3_runner_config.yaml`.
2. In the `model` section, set `download_only: true` if you just want the weights. Adjust `local_dir` to change where
   the snapshot is stored.
3. Run:

   ```bash
   conda activate general-labeller
   python -m Qwen3.qwen3_runner
   ```

4. Once the files exist locally, flip `download_only` back to `false` before generating text.

## Run a Prompt

Set `run.prompt` (and optionally `run.system_prompt`) in the YAML, leave `download_only: false`, then execute:

```bash
python -m Qwen3.qwen3_runner
```

Any overrides placed under `run` win over the defaults defined in the `model` section.

## Use the Class Inside Python

```python
from Qwen3 import Qwen3Config, Qwen3Runner

config = Qwen3Config(device="cuda", max_new_tokens=256)
runner = Qwen3Runner(config)
response = runner.generate(
    prompt="Write a short poem about Singapore's skyline at dusk."
)
print(response)
```

You can override the Hugging Face repo, local directory, dtype, or device through `Qwen3Config`.
Set `download_only=True` if you just need to fetch the files without loading them into memory.
