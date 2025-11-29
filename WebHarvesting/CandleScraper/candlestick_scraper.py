"""
Scrape candlestick pattern descriptions from configurable web sources.

Each source defines a CSS selector for headings (pattern names). For every heading we
collect text from subsequent siblings until the next heading of the same or higher level,
then derive simple heuristics about bullish/bearish intent and expected behavior.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import requests
import yaml
from bs4 import BeautifulSoup, Tag


DEFAULT_CONFIG_FILENAME = "candlestick_scraper_config.yaml"
DEFAULT_CONFIG_PATH = Path(__file__).with_name(DEFAULT_CONFIG_FILENAME)


@dataclass(slots=True)
class SourceConfig:
    name: str
    url: str
    heading_selector: str
    stop_tags: List[str]
    max_patterns: Optional[int] = None
    min_words: int = 15


@dataclass(slots=True)
class OutputConfig:
    metadata_path: Path
    append: bool = True


@dataclass(slots=True)
class ScraperConfig:
    sources: List[SourceConfig]
    output: OutputConfig
    user_agent: str = "CandlestickScraper/1.0 (+https://example.com)"
    timeout_seconds: int = 20


def load_config(config_path: Path) -> ScraperConfig:
    expanded = Path(config_path).expanduser().resolve()
    if not expanded.is_file():
        raise FileNotFoundError(f"Config file not found: {expanded}")
    with expanded.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    sources_data = data.get("sources")
    if not sources_data or not isinstance(sources_data, list):
        raise ValueError("Config must define a non-empty 'sources' list.")

    sources: List[SourceConfig] = []
    for entry in sources_data:
        if not isinstance(entry, dict):
            raise ValueError("Each source config must be a mapping.")
        stop_tags = [tag.lower() for tag in entry.get("stop_tags", [])]
        if not stop_tags:
            # Default to same heading tag(s) as selector (e.g., h2, h3)
            inferred = re.findall(r"(h[1-6])", entry.get("heading_selector", ""))
            stop_tags = [tag.lower() for tag in inferred] or ["h1", "h2", "h3"]
        sources.append(
            SourceConfig(
                name=str(entry.get("name")),
                url=str(entry.get("url")),
                heading_selector=str(entry.get("heading_selector")),
                stop_tags=stop_tags,
                max_patterns=entry.get("max_patterns"),
                min_words=int(entry.get("min_words", 15)),
            )
        )

    output_section = data.get("output", {})
    output_path = Path(output_section.get("metadata_path", "./WebHarvesting/CandleScraper/data/candlestick_patterns.jsonl")).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = OutputConfig(metadata_path=output_path, append=bool(output_section.get("append", True)))

    user_agent = str(data.get("user_agent") or "CandlestickScraper/1.0 (+https://example.com)")
    timeout_seconds = int(data.get("timeout_seconds", 20))

    return ScraperConfig(sources=sources, output=output, user_agent=user_agent, timeout_seconds=timeout_seconds)


class CandlestickScraper:
    def __init__(self, config: ScraperConfig) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.headers["User-Agent"] = config.user_agent

    def run(self) -> List[dict]:
        all_records: List[dict] = []
        for source in self.config.sources:
            try:
                html = self._fetch(source.url)
            except Exception as exc:
                print(f"[WARN] Failed to fetch {source.url}: {exc}")
                continue
            soup = BeautifulSoup(html, "lxml")
            headings = soup.select(source.heading_selector)
            if not headings:
                print(f"[WARN] No headings found for selector '{source.heading_selector}' on {source.url}")
                continue
            count = 0
            for heading in headings:
                if not isinstance(heading, Tag):
                    continue
                pattern_name = heading.get_text(" ", strip=True)
                description = extract_section_text(heading, stop_tags=source.stop_tags)
                if not description or len(description.split()) < source.min_words:
                    continue
                behavior = classify_behavior(description)
                expectation = infer_expectation(description, behavior)
                record = {
                    "pattern_name": pattern_name,
                    "behavior": behavior,
                    "expected_behavior": expectation,
                    "source": source.name,
                    "source_url": source.url,
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                    "summary": description,
                }
                all_records.append(record)
                count += 1
                if source.max_patterns and count >= source.max_patterns:
                    break
        if not all_records:
            print("No candlestick patterns were harvested.")
            return []
        self._write_records(all_records)
        print(f"Wrote {len(all_records)} pattern records to {self.config.output.metadata_path}")
        return all_records

    def _fetch(self, url: str) -> str:
        resp = self.session.get(url, timeout=self.config.timeout_seconds)
        resp.raise_for_status()
        return resp.text

    def _write_records(self, records: Iterable[dict]) -> None:
        mode = "a" if self.config.output.append else "w"
        with self.config.output.metadata_path.open(mode, encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_section_text(heading: Tag, stop_tags: Iterable[str]) -> str:
    stop_set = {tag.lower() for tag in stop_tags}
    texts: List[str] = []
    for sibling in heading.next_siblings:
        if isinstance(sibling, str):
            continue
        if not isinstance(sibling, Tag):
            continue
        name = (sibling.name or "").lower()
        if name in stop_set:
            break
        if name in {"p", "li"}:
            texts.append(sibling.get_text(" ", strip=True))
        elif name in {"ul", "ol"}:
            texts.append(" ".join(item.get_text(" ", strip=True) for item in sibling.find_all("li")))
    return " ".join(texts).strip()


def classify_behavior(text: str) -> str:
    lowered = text.lower()
    bullish_keywords = {"bullish", "rally", "uptrend", "buyers", "demand", "accumulation"}
    bearish_keywords = {"bearish", "selloff", "downtrend", "sellers", "supply", "distribution"}
    if any(word in lowered for word in bullish_keywords) and not any(word in lowered for word in bearish_keywords):
        return "bullish"
    if any(word in lowered for word in bearish_keywords) and not any(word in lowered for word in bullish_keywords):
        return "bearish"
    if "reversal" in lowered and "uptrend" in lowered:
        return "bullish"
    if "reversal" in lowered and "downtrend" in lowered:
        return "bearish"
    return "neutral"


def infer_expectation(text: str, behavior: str) -> str:
    lowered = text.lower()
    expectation_parts: List[str] = []

    if "reversal" in lowered:
        expectation_parts.append("signals potential reversal")
    if "continuation" in lowered:
        expectation_parts.append("indicates likely trend continuation")
    if "breakout" in lowered:
        expectation_parts.append("anticipates breakout move")
    if not expectation_parts:
        expectation_parts.append("describes general market sentiment shift")

    bias = behavior if behavior != "neutral" else "context-dependent"
    return f"{bias.capitalize()}: " + "; ".join(expectation_parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape candlestick patterns and expected behaviors.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to candlestick_scraper_config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        config = load_config(args.config)
        scraper = CandlestickScraper(config)
        scraper.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
