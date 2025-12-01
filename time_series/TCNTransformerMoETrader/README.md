# TCN + Transformer + MoE Hourly Trader

This folder packages a full research-grade prototype for hourly swing trading:

- `hourly_swing_model.py` – contains dataset helpers, sinusoidal encodings, TCN feature extractors, Transformer encoder layers (standard and sparse Mixture-of-Experts), and a configurable prediction head that supports regression and classification objectives. It also includes training utilities, schedulers, and example usage.
- `data_scraper/` – scripts for downloading hourly bars from FinancialModelingPrep, persisting them to CSV/Parquet, and preparing tensors for the model (`data_pipeline.py`).
- `.env` – expected to hold API keys or run-time configuration for the scraper.

## Typical workflow

1. Populate environment variables (e.g., FMP API key) and run `python TCNTransformerMoETrader/data_scraper/data_pipeline.py --symbols AAPL,MSFT`.
2. Load the resulting tensors (see `HourlySwingDataset` usage at the bottom of `hourly_swing_model.py`) and split into train/validation sets.
3. Instantiate `HourlySwingModel` with desired hyperparameters:
   ```python
   model = HourlySwingModel(seq_len=128, feature_dim=32, task_type="classification")
   ```
4. Train using the provided training loop + `TrainingConfig`, or integrate the model into your custom trainer.

Feel free to extend this folder with experiment notebooks, additional scrapers, or checkpoint management scripts. Keep sensitive credentials in `.env` and exclude them from version control.*** End Patch
