"""
data/pull_alpaca_daily.py — Pull S&P 500 daily bars + macro indicators from Alpaca.

Output
------
  data/raw/daily/<YYYYMMDD>/<TICKER>.parquet   per-ticker daily features
  data/raw/daily_<YYYYMMDD>.parquet            merged all tickers
  data/raw/market_daily_<YYYYMMDD>.parquet     SPY + macro market features

Usage
-----
  uv run python data/pull_alpaca_daily.py
  uv run python data/pull_alpaca_daily.py --start 2020-01-01 --tickers AAPL MSFT NVDA
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.macro import MACRO_COLS, get_all_macro  # noqa: E402

_env_path = ROOT / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ[_k.strip()] = _v.strip().strip('"').strip("'")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
log = logging.getLogger("pull_alpaca_daily")
console = Console()

RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK_YEARS = 5
VOL_WINDOW = 20
TRADING_DAYS_YEAR = 252
CHUNK_DAYS = 365
MAX_RETRIES = 5
RETRY_SLEEP = 10

SP500_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "JPM", "V", "UNH",
    "XOM", "JNJ", "WMT", "MA", "PG", "HD", "CVX", "MRK", "ABBV", "KO",
    "PEP", "COST", "AVGO", "TMO", "MCD", "CSCO", "ACN", "ABT", "DHR", "NEE",
    "LIN", "WFC", "PM", "TXN", "BMY", "UNP", "RTX", "HON", "QCOM", "LOW",
    "AMGN", "INTU", "SPGI", "BA", "SBUX", "ELV", "GS", "BLK", "AMD", "CAT",
    "ADP", "GILD", "MDT", "ISRG", "TJX", "VRTX", "SYK", "PFE", "DE", "LMT",
    "CB", "MMC", "REGN", "ADI", "CI", "PLD", "SO", "ZTS", "MO", "DUK",
    "NKE", "CME", "BDX", "CL", "EOG", "SLB", "USB", "PNC", "ITW", "EQIX",
    "SPY",
]


def get_creds() -> tuple[str, str]:
    key = os.environ.get("ALPACA_API_KEY", "").strip()
    secret = os.environ.get("ALPACA_SECRET_KEY", "").strip()
    if not key or not secret:
        console.print(
            "[red bold]Missing credentials.[/red bold]\n"
            "Fill in [cyan].env[/cyan]:\n"
            "  ALPACA_API_KEY=your_key\n"
            "  ALPACA_SECRET_KEY=your_secret"
        )
        sys.exit(1)
    return key, secret


def validate_keys(key: str, secret: str) -> None:
    r = requests.get(
        "https://paper-api.alpaca.markets/v2/account",
        headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
        timeout=10,
    )
    if r.status_code == 401:
        console.print("[red bold]401 Unauthorized.[/red bold] Update [cyan].env[/cyan]")
        sys.exit(1)
    log.info("API keys validated ✓")


def fetch_chunk(
    client: StockHistoricalDataClient,
    symbols: list[str],
    start: datetime,
    end: datetime,
    feed: str,
) -> pd.DataFrame:
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed=feed,
        adjustment="all",
    )
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.get_stock_bars(req).df
        except Exception as exc:
            msg = str(exc)
            if "401" in msg:
                log.error("401 Unauthorized — update .env and retry")
                sys.exit(1)
            if "invalid symbol" in msg.lower():
                log.warning(f"Invalid symbol skipped: {msg}")
                return pd.DataFrame()
            if attempt == MAX_RETRIES:
                log.error(f"Chunk {start.date()}→{end.date()} failed: {exc}")
                return pd.DataFrame()
            time.sleep(RETRY_SLEEP * attempt)
    return pd.DataFrame()


def fetch_bars(
    client: StockHistoricalDataClient,
    symbols: list[str],
    start: datetime,
    end: datetime,
    feed: str,
) -> pd.DataFrame:
    chunks, cursor = [], start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=CHUNK_DAYS), end)
        df = fetch_chunk(client, symbols, cursor, chunk_end, feed)
        if not df.empty:
            chunks.append(df)
        cursor = chunk_end + timedelta(seconds=1)
    return pd.concat(chunks).sort_index() if chunks else pd.DataFrame()


def find_actual_start(
    client: StockHistoricalDataClient,
    ticker: str,
    desired_start: datetime,
    end: datetime,
    feed: str,
) -> datetime:
    cursor = desired_start
    while cursor < end:
        probe_end = min(cursor + timedelta(days=365), end)
        req = StockBarsRequest(
            symbol_or_symbols=[ticker],
            timeframe=TimeFrame.Day,
            start=cursor,
            end=probe_end,
            feed=feed,
            adjustment="all",
            limit=1,
        )
        try:
            df = client.get_stock_bars(req).df
            if not df.empty:
                first_bar = df.index.get_level_values("timestamp")[0].to_pydatetime()
                actual = first_bar.replace(hour=0, minute=0, second=0, microsecond=0)
                return max(desired_start, actual)
        except Exception:
            pass
        cursor = probe_end + timedelta(seconds=1)
    return desired_start


def _align_daily_causal(series: pd.Series, bar_index: pd.DatetimeIndex) -> np.ndarray:
    s = series.dropna()
    if s.empty:
        return np.full(len(bar_index), np.nan)
    s = s.copy()
    s.index = s.index.tz_localize(None).normalize().as_unit("ns")
    s = s[~s.index.duplicated(keep="last")].sort_index().reset_index()
    s.columns = ["date", "v"]
    left = pd.DataFrame({"date": bar_index.tz_localize(None).normalize().as_unit("ns")})
    return pd.merge_asof(left, s, on="date", direction="backward", allow_exact_matches=False)["v"].values


def compute_daily_features(
    df: pd.DataFrame,
    spy_ret: pd.Series,
    spy_rvol: pd.Series,
    macro_df: pd.DataFrame,
) -> pd.DataFrame:
    out = df[["open", "high", "low", "close", "volume"]].copy()
    out["ret"] = np.log(out["close"] / out["close"].shift(1))
    out["log_hl_range"] = np.log(out["high"] / out["low"].replace(0, np.nan)).fillna(0.0)
    out["log_oc_gap"] = np.log(out["close"] / out["open"].replace(0, np.nan)).fillna(0.0)
    aligned_spy = spy_ret.reindex(out.index)
    out["ret_vs_spy"] = out["ret"] - aligned_spy
    vol_mean = out["volume"].rolling(VOL_WINDOW, min_periods=1).mean()
    out["rel_volume"] = out["volume"] / vol_mean.replace(0, np.nan)
    ann = math.sqrt(TRADING_DAYS_YEAR)
    out["realized_vol"] = out["ret"].rolling(VOL_WINDOW, min_periods=2).std() * ann
    out["spy_ret"] = aligned_spy
    out["spy_realized_vol"] = spy_rvol.reindex(out.index)

    for col in MACRO_COLS:
        if col in macro_df.columns:
            out[col] = _align_daily_causal(macro_df[col], out.index)
        else:
            out[col] = np.nan

    return out


def save_ticker(ticker: str, df: pd.DataFrame, run_date: str) -> Path:
    ticker_dir = RAW_DIR / "daily" / run_date
    ticker_dir.mkdir(parents=True, exist_ok=True)
    path = ticker_dir / f"{ticker}.parquet"
    df.reset_index().to_parquet(path, index=False)
    return path


def merge_tickers(run_date: str) -> Path:
    files = sorted((RAW_DIR / "daily" / run_date).glob("*.parquet"))
    if not files:
        return Path()
    combined = pd.concat(
        [pd.read_parquet(f).assign(ticker=f.stem) for f in files],
        ignore_index=True,
    )
    out = RAW_DIR / f"daily_{run_date}.parquet"
    combined.to_parquet(out, index=False)
    log.info(f"Merged {len(files)} tickers → {out}  ({len(combined):,} rows)")
    return out


def run(
    start_str: Optional[str] = None,
    end_str: Optional[str] = None,
    feed: str = "iex",
    tickers_override: Optional[list[str]] = None,
) -> None:
    run_date = datetime.now(timezone.utc).strftime("%Y%m%d")
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=LOOKBACK_YEARS * 365)
    if start_str:
        start_dt = datetime.fromisoformat(start_str).replace(tzinfo=timezone.utc)
    if end_str:
        end_dt = datetime.fromisoformat(end_str).replace(tzinfo=timezone.utc)

    console.rule("[bold blue]Stock2Vec — Daily Alpaca Pull[/bold blue]")
    log.info(f"Date range : {start_dt.date()} → {end_dt.date()}")
    log.info(f"Feed       : {feed.upper()}")

    key, secret = get_creds()
    validate_keys(key, secret)
    client = StockHistoricalDataClient(api_key=key, secret_key=secret)

    universe = list(dict.fromkeys(tickers_override or SP500_TICKERS))
    if "SPY" not in universe:
        universe.append("SPY")
    log.info(f"Universe   : {len(universe)} symbols")

    _s, _e = start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")

    console.print("\n[dim]Fetching SPY …[/dim]")
    spy_raw = fetch_bars(client, ["SPY"], start_dt, end_dt, feed)
    if spy_raw.empty:
        log.error("Could not fetch SPY — aborting")
        sys.exit(1)

    spy_df = spy_raw.xs("SPY", level="symbol") if isinstance(spy_raw.index, pd.MultiIndex) else spy_raw
    spy_ret = np.log(spy_df["close"] / spy_df["close"].shift(1)).rename("spy_ret")
    spy_rvol = (spy_ret.rolling(VOL_WINDOW, min_periods=2).std() * math.sqrt(TRADING_DAYS_YEAR)).rename("spy_realized_vol")

    console.print("[dim]Fetching macro indicators …[/dim]")
    macro_df = get_all_macro(_s, _e)

    market_path = RAW_DIR / f"market_daily_{run_date}.parquet"
    mkt = pd.DataFrame({"spy_ret": spy_ret, "spy_realized_vol": spy_rvol})
    for col in MACRO_COLS:
        if col in macro_df.columns:
            mkt[col] = _align_daily_causal(macro_df[col], spy_ret.index)
    mkt.reset_index().to_parquet(market_path, index=False)
    log.info(f"Market features → {market_path}")

    ticker_dir = RAW_DIR / "daily" / run_date
    done = {f.stem for f in ticker_dir.glob("*.parquet")} if ticker_dir.exists() else set()
    remaining = [t for t in universe if t not in done]
    log.info(f"Tickers    : {len(remaining)} to fetch  ({len(done)} cached)")

    batch_size = 20
    batches = [remaining[i : i + batch_size] for i in range(0, len(remaining), batch_size)]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching daily bars …", total=len(remaining))
        for batch in batches:
            starts = {t: find_actual_start(client, t, start_dt, end_dt, feed) for t in batch}
            groups: dict[str, list[str]] = {}
            for t, s in starts.items():
                groups.setdefault(s.isoformat(), []).append(t)

            for iso_start, group in groups.items():
                g_start = datetime.fromisoformat(iso_start)
                batch_df = fetch_bars(client, group, g_start, end_dt, feed)
                if batch_df.empty:
                    progress.advance(task, len(group))
                    continue

                symbols = (
                    batch_df.index.get_level_values("symbol").unique()
                    if isinstance(batch_df.index, pd.MultiIndex)
                    else group[:1]
                )
                for ticker in symbols:
                    try:
                        tkr_df = (
                            batch_df.xs(ticker, level="symbol").copy()
                            if isinstance(batch_df.index, pd.MultiIndex)
                            else batch_df.copy()
                        )
                        if not tkr_df.empty:
                            feat = compute_daily_features(tkr_df, spy_ret, spy_rvol, macro_df)
                            save_ticker(ticker, feat, run_date)
                    except Exception as exc:
                        log.error(f"{ticker}: {exc}")
                    finally:
                        progress.advance(task, 1)

    console.print("\n[dim]Merging …[/dim]")
    merged = merge_tickers(run_date)

    console.rule("[bold green]Done[/bold green]")
    console.print(f"  Daily bars : [cyan]{merged}[/cyan]")
    console.print(f"  Market     : [cyan]{market_path}[/cyan]")
    console.print(f"  Per-ticker : [cyan]{ticker_dir}[/cyan]")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pull S&P 500 daily bars from Alpaca")
    p.add_argument("--start", default=None, help="YYYY-MM-DD")
    p.add_argument("--end", default=None, help="YYYY-MM-DD")
    p.add_argument("--feed", default=os.environ.get("ALPACA_FEED", "iex"), choices=["iex", "sip"])
    p.add_argument("--tickers", nargs="+", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(start_str=args.start, end_str=args.end, feed=args.feed, tickers_override=args.tickers)
