# PatternClassifier

This module compares fresh candlestick screenshots against a labeled reference catalog using CLIP embeddings. It is designed to complement the CandleScraper output by reusing the same pattern names and behavioral metadata.

## Reference Layout

```
WebHarvesting/PatternClassifier/reference_catalog/
  Bullish_Engulfing/img1.png
  Bullish_Engulfing/img2.png
  Shooting_Star/example.png
```

Each subdirectory name becomes the label for the contained images. Add as many curated examples as you like; the richer the catalog, the better the cosine-similarity matches.

## Configuration

- `model.clip_model_id` – any CLIP checkpoint from Hugging Face (default: `openai/clip-vit-base-patch32`).
- `run.reference_dir` – root folder for the labeled catalog.
- `run.input_path` – file or folder of query images to classify.
- `run.top_k` – number of matches per query.
- `run.behavior_jsonl` – optional link to `WebHarvesting/CandleScraper/data/candlestick_patterns.jsonl` so each match includes scraped descriptions.

Edit `pattern_image_classifier_config.yaml` to point at your folders. Paths are resolved relative to the repo unless you provide absolute ones.

## Usage

```bash
python -m WebHarvesting.PatternClassifier.pattern_image_classifier \
    --config WebHarvesting/PatternClassifier/pattern_image_classifier_config.yaml
```

The script saves `pattern_image_matches.json`, containing each query image plus its top-k matches, similarity scores, and any attached behavior summaries. Use it to triage new screenshots, auto-label historical candles, or validate that scraped definitions line up with visual data.
