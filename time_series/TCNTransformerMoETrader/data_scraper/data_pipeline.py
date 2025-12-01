"""
Multi-provider hourly data creation pipeline for swing-trading datasets.

Supports Financial Modeling Prep (FMP), Alpha Vantage, and Twelve Data.
Each provider fetches hourly OHLCV bars for requested tickers, combines
the results, and stores them locally for downstream ML pipelines.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol

import numpy as np
import pandas as pd
import requests


# -----------------------------------------------------------------------------
# Provider interface definitions
# -----------------------------------------------------------------------------
class Provider(Protocol):
    """Interface for hourly data providers."""

    name: str

    def fetch_hourly(self, symbol: str, start: dt.datetime, end: dt.datetime) -> List[Dict[str, float]]:
        ...


def _parse_date(value: str) -> dt.datetime:
    return dt.datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)


def _ensure_api_key(env_var: str, provided: Optional[str], provider_name: str) -> str:
    key = provided or os.getenv(env_var)
    if not key:
        raise RuntimeError(
            f"{provider_name} API key missing. Provide via --api-key or set {env_var}."
        )
    return key


class FMPProvider:
    """Financial Modeling Prep hourly data provider."""

    name = "financial_modeling_prep"
    BASE_URL = "https://financialmodelingprep.com/api/v3/historical-chart/1hour/{symbol}"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = _ensure_api_key("FMP_API_KEY", api_key, "FMP")

    def fetch_hourly(self, symbol: str, start: dt.datetime, end: dt.datetime) -> List[Dict[str, float]]:
        url = self.BASE_URL.format(symbol=symbol)
        params = {"from": start.date().isoformat(), "to": end.date().isoformat(), "apikey": self.api_key}
        response = requests.get(url, params=params, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f"FMP request failed for {symbol}: {response.status_code} {response.text[:120]}")
        data = response.json()
        records = []
        for row in data:
            timestamp = dt.datetime.fromisoformat(row["date"]).replace(tzinfo=dt.timezone.utc)
            if timestamp < start or timestamp > end:
                continue
            records.append(
                {
                    "symbol": symbol,
                    "timestamp": timestamp.isoformat(),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                    "provider": self.name,
                }
            )
        return records


class AlphaVantageProvider:
    """Alpha Vantage hourly data provider (TIME_SERIES_INTRADAY)."""

    name = "alpha_vantage"
    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = _ensure_api_key("ALPHAVANTAGE_API_KEY", api_key, "Alpha Vantage")

    def fetch_hourly(self, symbol: str, start: dt.datetime, end: dt.datetime) -> List[Dict[str, float]]:
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": "60min",
            "outputsize": "full",
            "apikey": self.api_key,
        }
        response = requests.get(self.BASE_URL, params=params, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f"Alpha Vantage request failed for {symbol}: {response.status_code} {response.text[:120]}")
        payload = response.json()
        key = "Time Series (60min)"
        series = payload.get(key, {})
        if not series:
            raise RuntimeError(f"Alpha Vantage response missing '{key}' for {symbol}: {payload}")

        records: List[Dict[str, float]] = []
        for ts, values in series.items():
            timestamp = dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
            if not (start <= timestamp <= end):
                continue
            records.append(
                {
                    "symbol": symbol,
                    "timestamp": timestamp.isoformat(),
                    "open": float(values["1. open"]),
                    "high": float(values["2. high"]),
                    "low": float(values["3. low"]),
                    "close": float(values["4. close"]),
                    "volume": float(values["5. volume"]),
                    "provider": self.name,
                }
            )
        records.sort(key=lambda r: r["timestamp"])
        return records


class TwelveDataProvider:
    """Twelve Data hourly provider for wide global coverage."""

    name = "twelve_data"
    BASE_URL = "https://api.twelvedata.com/time_series"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = _ensure_api_key("TWELVEDATA_API_KEY", api_key, "Twelve Data")

    def fetch_hourly(self, symbol: str, start: dt.datetime, end: dt.datetime) -> List[Dict[str, float]]:
        params = {
            "symbol": symbol,
            "interval": "1h",
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
            "timezone": "UTC",
            "format": "JSON",
            "apikey": self.api_key,
        }
        response = requests.get(self.BASE_URL, params=params, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f"Twelve Data request failed for {symbol}: {response.status_code} {response.text[:120]}")
        payload = response.json()
        values = payload.get("values", [])
        records: List[Dict[str, float]] = []
        for row in values:
            timestamp = dt.datetime.strptime(row["datetime"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
            if timestamp < start or timestamp > end:
                continue
            records.append(
                {
                    "symbol": symbol,
                    "timestamp": timestamp.isoformat(),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                    "provider": self.name,
                }
            )
        records.sort(key=lambda r: r["timestamp"])
        return records


# -----------------------------------------------------------------------------
# Pipeline components
# -----------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    tickers: List[str]
    start: str
    end: str
    provider: str = "fmp"
    api_key: Optional[str] = None
    output_csv: str = "TCNTransformerMoETrader/data/hourly_dataset.csv"
    output_parquet: Optional[str] = "TCNTransformerMoETrader/data/hourly_dataset.parquet"
    rate_limit_sec: float = 1.0
    deduplicate: bool = True
    compute_log_returns: bool = True

    def provider_instance(self) -> Provider:
        provider = self.provider.lower()
        if provider == "fmp":
            return FMPProvider(api_key=self.api_key)
        if provider == "alphavantage":
            return AlphaVantageProvider(api_key=self.api_key)
        if provider == "twelvedata":
            return TwelveDataProvider(api_key=self.api_key)
        raise ValueError(f"Unknown provider '{self.provider}'. Choose from fmp, alphavantage, twelvedata.")


class DataPipeline:
    """End-to-end pipeline orchestrating fetch, clean, and storage."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.provider = config.provider_instance()
        self.start_dt = _parse_date(config.start)
        self.end_dt = _parse_date(config.end)

    def _to_dataframe(self, rows: Iterable[Dict[str, float]]) -> pd.DataFrame:
        df = pd.DataFrame(list(rows))
        if df.empty:
            raise RuntimeError("No data collected; check tickers, dates, and provider quota.")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.sort_values(["symbol", "timestamp"], inplace=True)
        if self.config.deduplicate:
            df.drop_duplicates(subset=["symbol", "timestamp"], keep="last", inplace=True)
        if self.config.compute_log_returns:
            log_ret = np.log(df["close"] / df.groupby("symbol")["close"].shift(1))
            log_ret = log_ret.replace([np.inf, -np.inf], 0.0).fillna(0.0)
            df["log_return"] = log_ret
        return df

    def run(self) -> pd.DataFrame:
        all_rows: List[Dict[str, float]] = []
        for symbol in self.config.tickers:
            print(f"[{self.provider.name}] Fetching {symbol} {self.start_dt.date()} -> {self.end_dt.date()}")
            rows = self.provider.fetch_hourly(symbol, self.start_dt, self.end_dt)
            print(f"  Retrieved {len(rows)} bars.")
            all_rows.extend(rows)
            time.sleep(self.config.rate_limit_sec)
        df = self._to_dataframe(all_rows)
        self._persist(df)
        return df

    def _persist(self, df: pd.DataFrame) -> None:
        csv_path = Path(self.config.output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV to {csv_path.resolve()}")
        if self.config.output_parquet:
            parquet_path = Path(self.config.output_parquet)
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(parquet_path, index=False)
            print(f"Saved Parquet to {parquet_path.resolve()}")


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-provider hourly data pipeline.")
    parser.add_argument("--tickers", nargs="+", required=True, help="List of ticker symbols.")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD).")
    parser.add_argument(
        "--provider",
        choices=["fmp", "alphavantage", "twelvedata"],
        default="fmp",
        help="Underlying data provider.",
    )
    parser.add_argument("--api-key", help="API key for the selected provider (or set env var).")
    parser.add_argument("--out-csv", default="TCNTransformerMoETrader/data/hourly_dataset.csv", help="Output CSV path.")
    parser.add_argument("--out-parquet", default="TCNTransformerMoETrader/data/hourly_dataset.parquet", help="Optional Parquet path.")
    parser.add_argument("--rate-limit", type=float, default=1.0, help="Seconds to wait between provider calls.")
    parser.add_argument("--no-log-return", action="store_true", help="Disable log return computation.")
    parser.add_argument("--no-dedupe", action="store_true", help="Keep duplicate bars if returned.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        provider=args.provider,
        api_key=args.api_key,
        output_csv=args.out_csv,
        output_parquet=args.out_parquet,
        rate_limit_sec=args.rate_limit,
        deduplicate=not args.no_dedupe,
        compute_log_returns=not args.no_log_return,
    )
    pipeline = DataPipeline(config)
    df = pipeline.run()
    print(f"Final dataset shape: {df.shape}")


if __name__ == "__main__":
    main()
