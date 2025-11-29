# CandleScraper

This module scrapes descriptive text about candlestick patterns and heuristically tags each one with its likely market behavior (bullish, bearish, or neutral) and expected outcome (reversal, continuation, breakout, etc.). The goal is to bootstrap a structured knowledge base you can feed into trading bots, labeling pipelines, or teaching material.

## How It Works

1. Load `candlestick_scraper_config.yaml`, which lists any number of data sources (URL + CSS selectors).
2. Fetch each page with `requests` using a custom user agent.
3. Parse headings and the paragraphs that follow using BeautifulSoup.
4. Generate simple behavioral tags based on keyword heuristics.
5. Append JSONL records (`pattern_name`, `behavior`, `expected_behavior`, `summary`, `source`, etc.) to `CandleScraper/data/candlestick_patterns.jsonl`.

## Requirements

Environment dependencies are already tracked in `environment.yml`, but in case you need a pip install directly:

```bash
pip install requests beautifulsoup4 lxml pyyaml
```

## Configuration

- `sources` – array of websites to harvest. Each entry needs a human-readable name, URL, CSS selector for headings, optional `stop_tags`, `max_patterns`, and a `min_words` threshold.
- `output` – JSONL destination path and whether to append or overwrite.
- `user_agent` / `timeout_seconds` – polite scraping knobs.

You can add more providers by copying an existing entry and adjusting the selectors. The scraper stops collecting text when it reaches another heading listed in `stop_tags`, which keeps sections clean even on long-form articles.

## Usage

```bash
python -m WebHarvesting.CandleScraper.candlestick_scraper \
    --config WebHarvesting/CandleScraper/candlestick_scraper_config.yaml
```

After running, inspect `WebHarvesting/CandleScraper/data/candlestick_patterns.jsonl` to see structured output like:

```json
{"pattern_name": "Bullish Engulfing", "behavior": "bullish", "expected_behavior": "Bullish: signals potential reversal; indicates likely trend continuation", ...}
```

These records can seed your own knowledge graphs, power educational chatbots, or act as metadata for labeling candle images. Extend the heuristics in `candlestick_scraper.py` if you need finer-grained behaviors (e.g., probability scores, market context, timeframe hints). Use `append: false` in the config when you want a clean rebuild.
