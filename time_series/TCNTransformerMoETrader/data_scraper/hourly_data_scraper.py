"""
Utility script for collecting hourly market data from Yahoo Finance's public chart API.

The script fetches OHLCV bars for each requested ticker symbol between start and
end dates, then saves the result to CSV for downstream training.

Usage:
    python hourly_data_scraper.py --tickers BTC-USD ETH-USD AAPL MSFT \
        --start 2023-01-01 --end 2023-03-01 \
        --out data/hourly_quotes.csv
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests

try:
    import yfinance  # type: ignore
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    yfinance = None  # type: ignore
    pd = None  # type: ignore


YAHOO_CHART_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    "?period1={period1}&period2={period2}&interval=1h&events=div,splits"
    "&includeAdjustedClose=true"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; HourlySwingScraper/1.0; +https://example.com)",
    "Accept": "application/json,text/javascript,*/*;q=0.01",
}


def parse_date(date_str: str) -> int:
    """Convert YYYY-MM-DD string into Unix timestamp (seconds)."""
    try:
        date = dt.datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Date '{date_str}' must match YYYY-MM-DD"
        ) from exc
    return int(date.replace(tzinfo=dt.timezone.utc).timestamp())


def fetch_symbol_data_yahoo(
    symbol: str,
    start_ts: int,
    end_ts: int,
    session: Optional[requests.Session] = None,
    max_retries: int = 5,
    backoff_seconds: float = 2.0,
) -> List[Dict[str, float]]:
    """Fetch hourly OHLCV bars for a ticker between timestamps with retries."""
    url = YAHOO_CHART_URL.format(symbol=symbol, period1=start_ts, period2=end_ts)
    response: Optional[requests.Response] = None
    session = session or requests.Session()
    for attempt in range(max_retries):
        response = session.get(url, headers=HEADERS, timeout=30)
        if response.status_code == 200:
            break
        if response.status_code in {422, 429, 500, 502, 503, 504} and attempt < max_retries - 1:
            wait = backoff_seconds * (2 ** attempt)
            print(f"  Received {response.status_code} for {symbol}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
            continue
        raise RuntimeError(f"Failed to fetch {symbol} ({response.status_code})")

    if response is None:
        raise RuntimeError(f"Unable to fetch {symbol}; no response received.")

    payload = response.json()
    result = payload.get("chart", {}).get("result")
    if not result:
        raise RuntimeError(f"No chart data returned for {symbol}: {json.dumps(payload)}")
    data = result[0]
    timestamps = data.get("timestamp", [])
    indicators = data.get("indicators", {}).get("quote", [{}])[0]

    bars: List[Dict[str, float]] = []
    for idx, ts in enumerate(timestamps):
        bar = {
            "symbol": symbol,
            "timestamp": dt.datetime.utcfromtimestamp(ts).isoformat(),
            "open": indicators.get("open", [None])[idx],
            "high": indicators.get("high", [None])[idx],
            "low": indicators.get("low", [None])[idx],
            "close": indicators.get("close", [None])[idx],
            "volume": indicators.get("volume", [None])[idx],
        }
        if any(value is None for key, value in bar.items() if key not in {"symbol", "timestamp"}):
            continue
        bars.append(bar)
    return bars


def save_to_csv(rows: Iterable[Dict[str, float]], output_path: Path) -> None:
    """Write rows to CSV, creating parent directories if needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    with output_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fetch_symbol_data_yfinance(symbol: str, start: str, end: str) -> List[Dict[str, float]]:
    """Fetch hourly OHLCV bars using yfinance (handles cookies, throttling)."""
    if yfinance is None or pd is None:
        raise RuntimeError("Install yfinance and pandas to use the 'yfinance' provider: pip install yfinance pandas")
    data = yfinance.download(
        tickers=symbol,
        interval="60m",
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        actions=False,
    )
    if data.empty:
        print(f"  No data returned for {symbol} via yfinance.")
        return []
    data = data.rename(
        columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    )
    data = data.reset_index()
    bars: List[Dict[str, float]] = []
    for _, row in data.iterrows():
        timestamp = row["Datetime"] if "Datetime" in row else row["index"]
        bars.append(
            {
                "symbol": symbol,
                "timestamp": pd.to_datetime(timestamp, utc=True).isoformat(),  # type: ignore[arg-type]
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
        )
    return bars


def scrape_dataset(
    tickers: List[str],
    start: str,
    end: str,
    provider: str = "yfinance",
    chunk_days: int = 20,
    pause_seconds: float = 0.5,
) -> List[Dict[str, float]]:
    """Scrape hourly data for multiple tickers."""
    provider = provider.lower()
    start_ts = parse_date(start)
    end_ts = parse_date(end)
    if end_ts <= start_ts:
        raise ValueError("End date must be after start date.")

    dataset: List[Dict[str, float]] = []
    session = requests.Session() if provider == "yahoo" else None
    for ticker in tickers:
        print(f"Fetching {ticker}...")
        if provider == "yfinance":
            bars = fetch_symbol_data_yfinance(ticker, start, end)
            dataset.extend(bars)
            print(f"  Retrieved {len(bars)} records via yfinance.")
            time.sleep(pause_seconds)
        elif provider == "yahoo":
            window = chunk_days * 24 * 3600
            cursor = start_ts
            ticker_rows: List[Dict[str, float]] = []
            while cursor < end_ts:
                chunk_end = min(end_ts, cursor + window)
                chunk = fetch_symbol_data_yahoo(
                    ticker,
                    cursor,
                    chunk_end,
                    session=session,
                )
                ticker_rows.extend(chunk)
                cursor = chunk_end - 3600  # overlap 1 hour to avoid gaps
                time.sleep(pause_seconds)
            dataset.extend(ticker_rows)
            print(f"  Retrieved {len(ticker_rows)} records via raw Yahoo.")
        else:
            raise ValueError("provider must be 'yfinance' or 'yahoo'")
    print(f"Total records collected: {len(dataset)}")
    return dataset


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scrape hourly OHLCV data from Yahoo Finance.")
    parser.add_argument("--tickers", nargs="+", required=True, help="List of ticker symbols.")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD).")
    parser.add_argument("--out", default="data/hourly_quotes.csv", help="Output CSV path.")
    parser.add_argument(
        "--provider",
        choices=["yfinance", "yahoo"],
        default="yfinance",
        help="Backend to use for scraping (yfinance recommended).",
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=20,
        help="For provider=yahoo: maximum number of days per request window.",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.5,
        help="Pause duration between API calls (seconds).",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    dataset = scrape_dataset(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        provider=args.provider,
        chunk_days=args.chunk_days,
        pause_seconds=args.pause,
    )
    output_path = Path(args.out)
    save_to_csv(dataset, output_path)
    print(f"Saved dataset to {output_path.resolve()}")


if __name__ == "__main__":
    main()
