# WebHarvesting Suite

This workspace bundles lightweight scrapers and classifiers for candlestick-pattern research:

- **CandleScraper/** – pulls pattern descriptions from curated websites, tags each entry with heuristic metadata, and writes JSONL files for downstream consumption (`candlestick_scraper.py`, configurable via `candlestick_scraper_config.yaml`).
- **PatternClassifier/** – uses CLIP embeddings to match new candlestick screenshots against a reference catalog, optionally enriching matches with metadata exported by CandleScraper.

## Quick start

1. Install dependencies shared across both tools:
   ```bash
   pip install requests beautifulsoup4 lxml pyyaml pillow torch torchvision transformers
   ```
2. Run the scrapers/classifiers:
   - `python -m WebHarvesting.CandleScraper.candlestick_scraper --config WebHarvesting/CandleScraper/candlestick_scraper_config.yaml`
   - `python -m WebHarvesting.PatternClassifier.pattern_image_classifier --config WebHarvesting/PatternClassifier/pattern_image_classifier_config.yaml`

The output artifacts land inside the respective `data/` directories (JSONL metadata, pattern match reports, etc.). Extend this folder with new providers or downstream consumers as your trading-data workflows grow.*** End Patch
