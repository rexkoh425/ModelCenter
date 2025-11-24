# SQL MoE Orchestrator (Qwen3-8B)

This package wraps the existing `Qwen3` runner with a mixture-of-experts (MoE) controller tailored for SQL generation. The orchestrator coordinates three lightweight agents—`planner`, `sql_writer`, and `validator`—that each call Qwen3-8B with specialist prompts. The validator produces the final SQL plus a short explanation.

## Layout

| File | Purpose |
| --- | --- |
| `sql_moe_orchestrator.py` | Entry point and orchestration logic. |
| `sql_moe_orchestrator_config.yaml` | Configures the Qwen runner, router, schema, and default run payload. |
| `schemas/ecommerce_sample_schema.sql` | Sample schema that can be replaced with your own database definition. |
| `time_series.py` | Standalone sequential MoE for transformer-based time-series fine-tuning. |

## Quickstart

1. Ensure the base Qwen3 runner is configured (edit `Qwen3/qwen3_runner_config.yaml` if needed).
2. Update `MoE/sql_moe_orchestrator_config.yaml` with your schema (either inline or via `schema_file`).
3. Run the orchestrator:
   ```bash
   python -m MoE.sql_moe_orchestrator --config MoE/sql_moe_orchestrator_config.yaml \
       --query "List monthly revenue and refund totals for 2024."
   ```
4. Use `--quiet` to suppress expert dumps or `--extra-context` to pass additional constraints at runtime.

The script prints the router’s plan, each expert’s JSON (unless `--quiet`), the final SQL, and the validator’s explanation. Replace the sample schema with your warehouse objects to adapt it to your environment.

## Time-series MoE (sequential experts)

`time_series.py` provides a GPU-friendly transformer MoE stack where each expert specializes in a signal component (trend/seasonality/anomaly) and the router calls the experts sequentially. This keeps memory requirements low (fits in ~16 GB with the default configs) and produces a handy "debate" style aggregation log.

### Quickstart

1. Install PyTorch (CUDA build recommended) and launch the demo:
   ```bash
   python -m MoE.time_series
   ```
   The script fabricates synthetic data, builds three experts, and fine-tunes them for a few epochs.
2. Replace `_generate_synthetic_series` / `_build_metadata` with your loaders, or wrap your tensors using `TimeSeriesWindowDataset`.
3. To customize experts, pass your own `ExpertConfig` list into `TimeSeriesMoE`. You can safely increase/decrease model width/layers and add/removal experts as long as they share the same input dimensionality.

### Fine-tuning hooks

- `fine_tune` accepts any `DataLoader` that returns `(window, target, metadata)` triplets. Windows are shaped `(batch, seq_len, feature_dim)`; targets are `(batch, horizon)`; metadata is optional `(batch, metadata_dim)`.
- The router uses half precision (`fp16`) by default. Switch to `"fp32"` in `RouterConfig` if you prefer full precision or are training on CPU.
- Gradients are clipped and scaled, so you can safely push learning rates up when running with small batches.

### QFormer-style experts

- Each expert now includes a lightweight Q-former that owns eight learned question tokens by default. The tokens attend over the encoded window so every expert can “ask” about specific temporal evidence before debating with peers.
- Override `num_questions` or provide `question_templates` in `ExpertConfig` to describe what each expert should focus on (e.g., `"seasonal_expert: does the 24h cycle strengthen?"`). The router log now records both the question embeddings and their human-readable prompts for inspection.

### Stock consensus workflow

- The new `StockPredictor` provides the main price proposal on GPU, while `StockConsensusSystem` wraps it with the sequential experts so they can challenge the call using indicator-specific evidence (SMA, momentum, volatility, etc.).
- `run_debate` keeps looping until the experts agree that the base prediction sits within a configurable tolerance; if prolonged disagreement persists, their proposals are averaged in a final “forced consensus” round.
- Each expert raises doubts aggressively (low thresholds and positive `doubt_bias`), and they continue to hold their stance for multiple rounds whenever their indicator disagrees with the main model.
- Run `python -m MoE.time_series` to see the full pipeline: the script fabricates stock-like data, fine-tunes the MoE on GPU (falling back to CPU if CUDA is unavailable), trains the main predictor, and prints the debate transcript plus price/explanation outputs.
