"""
Pipeline for sourcing Creative Commons first-person driving videos.

The scraper queries YouTube Data API v3 for Creative Commons licensed POV driving clips,
stores structured metadata, and optionally downloads the videos via yt-dlp.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests
import yaml


ISO_8601_DURATION_PREFIX = "PT"

DEFAULT_CONFIG_FILENAME = "fpv_video_scraper_config.yaml"
DEFAULT_CONFIG_PATH = Path(__file__).with_name(DEFAULT_CONFIG_FILENAME)


@dataclass(slots=True)
class YouTubeConfig:
    api_key_env: str = "YOUTUBE_API_KEY"
    region_code: str = "US"
    video_license: str = "creativeCommon"
    video_duration: str = "medium"
    published_after: Optional[str] = None  # ISO timestamp
    published_before: Optional[str] = None
    max_results_per_term: int = 25


@dataclass(slots=True)
class DownloadConfig:
    output_dir: Path
    metadata_path: Path
    max_videos: int = 30
    skip_existing: bool = True
    download_assets: bool = True
    prefer_audio: bool = False
    dry_run: bool = False


@dataclass(slots=True)
class SearchConfig:
    queries: List[str]
    min_duration_seconds: Optional[int] = 30
    max_duration_seconds: Optional[int] = 600
    require_terms: List[str] = None


@dataclass(slots=True)
class ScraperConfig:
    youtube: YouTubeConfig
    download: DownloadConfig
    search: SearchConfig


def load_config(config_path: Path) -> ScraperConfig:
    expanded = Path(config_path).expanduser().resolve()
    if not expanded.is_file():
        raise FileNotFoundError(f"Config file not found: {expanded}")

    with expanded.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")

    yt_section = data.get("youtube", {})
    dl_section = data.get("download", {})
    search_section = data.get("search", {})

    youtube_cfg = YouTubeConfig(
        api_key_env=str(yt_section.get("api_key_env", "YOUTUBE_API_KEY")),
        region_code=str(yt_section.get("region_code", "US")),
        video_license=str(yt_section.get("video_license", "creativeCommon")),
        video_duration=str(yt_section.get("video_duration", "medium")),
        published_after=yt_section.get("published_after"),
        published_before=yt_section.get("published_before"),
        max_results_per_term=int(yt_section.get("max_results_per_term", 25)),
    )

    output_dir = Path(dl_section.get("output_dir", "./data/fpv_videos")).expanduser()
    metadata_path = Path(dl_section.get("metadata_path", output_dir / "metadata.jsonl")).expanduser()
    download_cfg = DownloadConfig(
        output_dir=output_dir,
        metadata_path=metadata_path,
        max_videos=int(dl_section.get("max_videos", 30)),
        skip_existing=bool(dl_section.get("skip_existing", True)),
        download_assets=bool(dl_section.get("download_assets", True)),
        prefer_audio=bool(dl_section.get("prefer_audio", False)),
        dry_run=bool(dl_section.get("dry_run", False)),
    )

    queries = search_section.get("queries")
    if not queries:
        raise ValueError("search.queries is required and must be a list of search strings.")
    if not isinstance(queries, list):
        raise ValueError("search.queries must be a list of strings.")

    require_terms = search_section.get("require_terms") or []
    if not isinstance(require_terms, list):
        raise ValueError("search.require_terms must be a list when provided.")

    search_cfg = SearchConfig(
        queries=[str(q) for q in queries],
        min_duration_seconds=_optional_int(search_section.get("min_duration_seconds")),
        max_duration_seconds=_optional_int(search_section.get("max_duration_seconds")),
        require_terms=[str(term).lower() for term in require_terms],
    )

    return ScraperConfig(youtube=youtube_cfg, download=download_cfg, search=search_cfg)


def _optional_int(value) -> Optional[int]:
    if value is None:
        return None
    return int(value)


class YouTubeCreativeCommonsClient:
    """Thin wrapper around YouTube Data API v3 calls."""

    BASE_URL = "https://www.googleapis.com/youtube/v3"

    def __init__(self, api_key: str, config: YouTubeConfig) -> None:
        if not api_key:
            raise ValueError("YouTube API key is required. Set it in the environment per config youtube.api_key_env.")
        self.api_key = api_key
        self.config = config
        self.session = requests.Session()

    def search_videos(self, query: str) -> List[Dict]:
        params = {
            "part": "snippet",
            "type": "video",
            "videoLicense": self.config.video_license,
            "videoEmbeddable": "true",
            "videoSyndicated": "true",
            "videoDuration": self.config.video_duration,
            "regionCode": self.config.region_code,
            "maxResults": min(self.config.max_results_per_term, 50),
            "q": query,
            "key": self.api_key,
            "order": "relevance",
        }
        if self.config.published_after:
            params["publishedAfter"] = self.config.published_after
        if self.config.published_before:
            params["publishedBefore"] = self.config.published_before

        response = self.session.get(f"{self.BASE_URL}/search", params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        items = payload.get("items", [])
        video_ids = [item["id"]["videoId"] for item in items if item.get("id", {}).get("videoId")]
        if not video_ids:
            return []
        return self.fetch_video_metadata(video_ids)

    def fetch_video_metadata(self, video_ids: Iterable[str]) -> List[Dict]:
        chunks = _chunked(list(video_ids), 50)
        videos: List[Dict] = []
        for chunk in chunks:
            params = {
                "part": "snippet,contentDetails,statistics",
                "id": ",".join(chunk),
                "key": self.api_key,
            }
            response = self.session.get(f"{self.BASE_URL}/videos", params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            videos.extend(payload.get("items", []))
        return videos


def _chunked(seq: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def iso_duration_to_seconds(duration: str) -> int:
    # Simple ISO8601 duration parsing for formats like PT5M10S
    if not duration.startswith(ISO_8601_DURATION_PREFIX):
        raise ValueError(f"Unexpected duration format: {duration}")
    duration = duration[len(ISO_8601_DURATION_PREFIX) :]
    hours = minutes = seconds = 0
    num = ""
    for char in duration:
        if char.isdigit():
            num += char
            continue
        if char == "H":
            hours = int(num or 0)
        elif char == "M":
            minutes = int(num or 0)
        elif char == "S":
            seconds = int(num or 0)
        num = ""
    return hours * 3600 + minutes * 60 + seconds


class FPVVideoScraper:
    def __init__(self, config: ScraperConfig) -> None:
        self.config = config
        api_key = _get_env(config.youtube.api_key_env)
        self.youtube = YouTubeCreativeCommonsClient(api_key, config.youtube)
        self.download_dir = config.download.output_dir
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = config.download.metadata_path
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

    def run(self) -> List[Dict]:
        collected: List[Dict] = []
        for query in self.config.search.queries:
            videos = self.youtube.search_videos(query)
            filtered = self._filter_videos(videos)
            for video in filtered:
                if len(collected) >= self.config.download.max_videos:
                    break
                collected.append(self._format_metadata(video, source_query=query))
            if len(collected) >= self.config.download.max_videos:
                break

        if not collected:
            print("No videos matched the filters.")
            return []

        self._append_metadata(collected)

        if self.config.download.download_assets and not self.config.download.dry_run:
            self._download_videos(collected)
        else:
            print("Download skipped (download_assets=false or dry_run=true).")

        return collected

    def _filter_videos(self, videos: Iterable[Dict]) -> List[Dict]:
        output = []
        for video in videos:
            duration = video.get("contentDetails", {}).get("duration")
            if not duration:
                continue
            try:
                seconds = iso_duration_to_seconds(duration)
            except ValueError:
                continue
            min_d = self.config.search.min_duration_seconds
            max_d = self.config.search.max_duration_seconds
            if min_d and seconds < min_d:
                continue
            if max_d and seconds > max_d:
                continue
            title = video.get("snippet", {}).get("title", "")
            description = video.get("snippet", {}).get("description", "")
            haystack = f"{title}\n{description}".lower()
            if any(term not in haystack for term in self.config.search.require_terms):
                continue
            output.append(video)
        return output

    def _format_metadata(self, video: Dict, source_query: str) -> Dict:
        snippet = video.get("snippet", {})
        stats = video.get("statistics", {})
        content = video.get("contentDetails", {})
        video_id = video.get("id")
        duration_seconds = iso_duration_to_seconds(content.get("duration", "PT0S"))
        published_at = snippet.get("publishedAt")
        try:
            published_unix = int(datetime.fromisoformat(published_at.replace("Z", "+00:00")).timestamp()) if published_at else None
        except Exception:
            published_unix = None

        return {
            "video_id": video_id,
            "title": snippet.get("title"),
            "description": snippet.get("description"),
            "channel_title": snippet.get("channelTitle"),
            "thumbnail_url": snippet.get("thumbnails", {}).get("high", {}).get("url"),
            "duration_seconds": duration_seconds,
            "published_at": published_at,
            "published_unix": published_unix,
            "license": self.config.youtube.video_license,
            "view_count": int(stats.get("viewCount", 0)) if stats.get("viewCount") else None,
            "like_count": int(stats.get("likeCount", 0)) if stats.get("likeCount") else None,
            "query": source_query,
            "video_url": f"https://www.youtube.com/watch?v={video_id}",
            "download_path": str(self.download_dir / f"{video_id}.%(ext)s"),
        }

    def _append_metadata(self, records: List[Dict]) -> None:
        with self.metadata_path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Appended {len(records)} records to {self.metadata_path}")

    def _download_videos(self, records: Iterable[Dict]) -> None:
        for record in records:
            video_url = record["video_url"]
            output_template = str(self.download_dir / f"{record['video_id']}.%(ext)s")
            if self.config.download.skip_existing:
                expected_mp4 = output_template.replace("%(ext)s", "mp4")
                if Path(expected_mp4).exists():
                    print(f"Skipping existing file {expected_mp4}")
                    continue

            format_selector = "bestaudio/best" if self.config.download.prefer_audio else "bestvideo[height<=1080]+bestaudio/best"
            cmd = [
                "yt-dlp",
                "--quiet",
                "--no-warnings",
                "-f",
                format_selector,
                "-o",
                output_template,
                video_url,
            ]
            print(f"Downloading {video_url}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as exc:
                print(f"Failed to download {video_url}: {exc}")


def _get_env(name: str) -> str:
    import os

    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Environment variable '{name}' is not set but required for YouTube API.")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape first-person driving videos.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to fpv_video_scraper_config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Fetch metadata but skip downloads.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        config = load_config(args.config)
        if args.dry_run:
            config.download.dry_run = True
            config.download.download_assets = False
        scraper = FPVVideoScraper(config)
        scraper.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

