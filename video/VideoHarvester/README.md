# FPV Video Scraper

This utility searches for Creative Commons, first-person driving videos (driver POV) and optionally downloads them. It relies on the official YouTube Data API for discovery and `yt-dlp` for retrieval.

## Features

- Query multiple search phrases targeting POV/driver-seat footage.
- Filter by duration, required keywords, publish window, and Creative Commons license.
- Persist metadata to `metadata.jsonl` for later curation or model training.
- Optional downloads with resumable `yt-dlp` runs and skip-existing safeguards.
- Dry-run mode for quick previews when testing new queries.

## Requirements

1. Python 3.10+ with `requests`, `PyYAML`, and `yt-dlp` installed (`pip install requests pyyaml yt-dlp`).  
2. Google/YouTube Data API key stored in an environment variable (defaults to `YOUTUBE_API_KEY`).  
3. `transformers`/`torch` are **not** required—this is a standalone scraper.

## Configuration

Edit `fpv_video_scraper_config.yaml`:

- `youtube` – API settings (region, license, duration bucket, published window, per-query limit).  
- `search` – List of search phrases plus optional duration bounds and keywords that must appear in the title/description.  
- `download` – Output paths, max total videos, whether to download assets, and dry-run toggle.

All paths are resolved relative to the repository root, and the script will create directories on first run.

## Usage

```bash
export YOUTUBE_API_KEY="YOUR_KEY"
python -m VideoHarvester.fpv_video_scraper \
    --config VideoHarvester/fpv_video_scraper_config.yaml
```

To only collect metadata, pass `--dry-run` (or set `download.download_assets: false`). The metadata JSONL contains title, description, duration, stats, and the source search query for easy filtering downstream. You can plug those URLs into your own review UI or extend the script to capture thumbnails / transcripts.

## Extending

- Adjust `search.require_terms` for language-specific cues (e.g., `"dashcam"`, `"autoroute"`).  
- Add more providers by creating additional client classes (e.g., Pexels, Vimeo) and instantiating them inside `FPVVideoScraper`.  
- Add vision-language validation by hooking a detector that inspects downloaded frames to ensure they really are first-person driving footage before labeling.
