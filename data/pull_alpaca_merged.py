"""
Standalone Alpaca puller for Stock2Vec.

This version keeps the hardcoded S&P universe, preserves the downstream file
layout expected by prep_features.py, and adds the richer causal feature set plus
macro indicators from the divergent branch.
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
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
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
log = logging.getLogger("pull_alpaca_merged")
console = Console()

RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK_YEARS = 5
VOL_WINDOW = 20
TRADING_MINS_YEAR = 252 * 390
CHUNK_DAYS = 30
MAX_RETRIES = 5
RETRY_SLEEP = 10

SP500_TICKERS = [
    'MMM', 'AOS', 'ABT', 'ABBV', 'ACN', 'ADBE', 'AMD', 'AES', 'AFL', 'A',
    'ABNB', 'AKAM', 'ALB', 'ARE', 'ALGN', 'ALLE', 'LNT', 'ALL', 'GOOGL', 'GOOG',
    'MO', 'AMZN', 'AMCR', 'AEE', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP',
    'AME', 'AMGN', 'APH', 'ADI', 'AON', 'APA', 'APO', 'AAPL', 'AMAT', 'APP',
    'APTV', 'ACGL', 'ADM', 'ARES', 'ANET', 'AJG', 'AIZ', 'T', 'ATO', 'ADSK',
    'ADP', 'AZO', 'AVB', 'AVY', 'AXON', 'BKR', 'BALL', 'BAC', 'BAX', 'BDX',
    'BBY', 'TECH', 'BIIB', 'BLK', 'BX', 'XYZ', 'BNY', 'BA', 'BKNG', 'BSX',
    'BMY', 'AVGO', 'BR', 'BRO', 'BF.B', 'BLDR', 'BG', 'BXP', 'CHRW', 'CDNS',
    'CPT', 'CPB', 'COF', 'CAH', 'CCL', 'CARR', 'CVNA', 'CASY', 'CAT', 'CBOE',
    'CBRE', 'CDW', 'COR', 'CNC', 'CNP', 'CF', 'CRL', 'SCHW', 'CHTR', 'CVX',
    'CMG', 'CB', 'CHD', 'CIEN', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG',
    'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'COHR', 'COIN', 'CL', 'CMCSA', 'FIX',
    'CAG', 'COP', 'ED', 'STZ', 'CEG', 'COO', 'CPRT', 'CPAY', 'CTVA', 'CSGP',
    'COST', 'CRH', 'CRWD', 'CCI', 'CSX', 'CMI', 'CVS', 'DHR', 'DRI', 'DDOG',
    'DVA', 'DECK', 'DE', 'DELL', 'DAL', 'DVN', 'DXCM', 'FANG', 'DLR', 'DG',
    'DLTR', 'D', 'DPZ', 'DASH', 'DOV', 'DOW', 'DHI', 'DTE', 'DUK', 'DD',
    'ETN', 'EBAY', 'SATS', 'ECL', 'EIX', 'EW', 'EA', 'ELV', 'EME', 'EMR', 'ETR',
    'EOG', 'EQT', 'EFX', 'EQIX', 'EQR', 'ERIE', 'ESS', 'EL', 'EG', 'EVRG', 'ES',
    'EXC', 'EXE', 'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 'FDS', 'FICO', 'FAST',
    'FRT', 'FDX', 'FDXF', 'FIS', 'FITB', 'FSLR', 'FE', 'FISV', 'F', 'FTNT',
    'FTV', 'FOXA', 'FOX', 'BEN', 'FCX', 'GRMN', 'IT', 'GE', 'GEHC', 'GEV',
    'GEN', 'GNRC', 'GD', 'GIS', 'GM', 'GPC', 'GILD', 'GPN', 'GL', 'GDDY', 'GS',
    'HAL', 'HIG', 'HAS', 'HCA', 'DOC', 'HSIC', 'HSY', 'HPE', 'HLT', 'HD', 'HON',
    'HRL', 'HST', 'HWM', 'HPQ', 'HUBB', 'HUM', 'HBAN', 'HII', 'IBM', 'IEX',
    'IDXX', 'ITW', 'INCY', 'IR', 'PODD', 'INTC', 'IBKR', 'ICE', 'IFF', 'IP',
    'INTU', 'ISRG', 'IVZ', 'INVH', 'IQV', 'IRM', 'JBHT', 'JBL', 'JKHY', 'J',
    'JNJ', 'JCI', 'JPM', 'KVUE', 'KDP', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI',
    'KKR', 'KLAC', 'KHC', 'KR', 'LHX', 'LH', 'LRCX', 'LVS', 'LDOS', 'LEN',
    'LII', 'LLY', 'LIN', 'LYV', 'LMT', 'L', 'LOW', 'LULU', 'LITE', 'LYB',
    'MTB', 'MPC', 'MAR', 'MRSH', 'MLM', 'MAS', 'MA', 'MKC', 'MCD', 'MCK', 'MDT',
    'MRK', 'META', 'MET', 'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MAA', 'MRNA',
    'TAP', 'MDLZ', 'MPWR', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MSCI', 'NDAQ',
    'NTAP', 'NFLX', 'NEM', 'NWSA', 'NWS', 'NEE', 'NKE', 'NI', 'NDSN', 'NSC',
    'NTRS', 'NOC', 'NCLH', 'NRG', 'NUE', 'NVDA', 'NVR', 'NXPI', 'ORLY', 'OXY',
    'ODFL', 'OMC', 'ON', 'OKE', 'ORCL', 'OTIS', 'PCAR', 'PKG', 'PLTR', 'PANW',
    'PSKY', 'PH', 'PAYX', 'PYPL', 'PNR', 'PEP', 'PFE', 'PCG', 'PM', 'PSX',
    'PNW', 'PNC', 'POOL', 'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG',
    'PTC', 'PSA', 'PHM', 'PWR', 'QCOM', 'DGX', 'Q', 'RL', 'RJF', 'RTX', 'O',
    'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RVTY', 'HOOD', 'ROK', 'ROL', 'ROP',
    'ROST', 'RCL', 'SPGI', 'CRM', 'SNDK', 'SBAC', 'SLB', 'STX', 'SRE', 'NOW',
    'SHW', 'SPG', 'SWKS', 'SJM', 'SW', 'SNA', 'SOLV', 'SO', 'LUV', 'SWK',
    'SBUX', 'STT', 'STLD', 'STE', 'SYK', 'SMCI', 'SYF', 'SNPS', 'SYY', 'TMUS',
    'TROW', 'TTWO', 'TPR', 'TRGP', 'TGT', 'TEL', 'TDY', 'TER', 'TSLA', 'TXN',
    'TPL', 'TXT', 'TMO', 'TJX', 'TKO', 'TTD', 'TSCO', 'TT', 'TDG', 'TRV',
    'TRMB', 'TFC', 'TYL', 'TSN', 'USB', 'UBER', 'UDR', 'ULTA', 'UNP', 'UAL',
    'UPS', 'URI', 'UNH', 'UHS', 'VLO', 'VEEV', 'VTR', 'VLTO', 'VRSN', 'VRSK',
    'VZ', 'VRTX', 'VRT', 'VTRS', 'VICI', 'V', 'VST', 'VMC', 'WRB', 'GWW',
    'WAB', 'WMT', 'DIS', 'WBD', 'WM', 'WAT', 'WEC', 'WFC', 'WELL', 'WST',
    'WDC', 'WY', 'WSM', 'WMB', 'WTW', 'WDAY', 'WYNN', 'XEL', 'XYL', 'YUM',
    'ZBRA', 'ZBH', 'ZTS', 'SPY',
]

BAR_FEATURE_COLUMNS = [
    "ret",
    "log_hl_range",
    "log_oc_gap",
    "log_vwap_dev",
    "ret_vs_spy",
    "rel_volume",
    "realized_vol",
    "spy_ret",
    "spy_realized_vol",
    "vix_close",
    "yield_spread",
    "credit_spread",
    "dxy",
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
        console.print(
            "[red bold]401 Unauthorized.[/red bold]\n"
            "Keys are invalid or revoked.\n"
            "Regenerate at [cyan]https://app.alpaca.markets[/cyan] and update [cyan].env[/cyan]"
        )
        sys.exit(1)
    log.info("API keys validated ✓")


def get_sp500_tickers() -> list[str]:
    return list(SP500_TICKERS)


def get_vix(start: str, end: str) -> pd.Series:
    """
    Daily VIX close as a UTC-indexed Series.
    Loads from data/raw/vix.csv if present, otherwise fetches from Yahoo Finance.
    """
    csv_path = RAW_DIR / "vix.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
        df.index = (
            pd.to_datetime(df.index).tz_localize("UTC")
            if df.index.tz is None
            else pd.to_datetime(df.index).tz_convert("UTC")
        )
        log.info(f"VIX: loaded {len(df)} rows from {csv_path}")
        return df["vix_close"]

    try:
        s1 = int(pd.Timestamp(start).timestamp())
        s2 = int(pd.Timestamp(end).timestamp())
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX?period1={s1}&period2={s2}&interval=1d"
        data = requests.get(url, timeout=30).json()
        chart = data["chart"]["result"][0]
        s = pd.Series(
            chart["indicators"]["quote"][0]["close"],
            index=pd.to_datetime(chart["timestamp"], unit="s", utc=True),
            name="vix_close",
        )
        s.reset_index().rename(columns={"index": "date"}).to_csv(csv_path, index=False)
        log.info(f"VIX: fetched {len(s)} rows, cached to {csv_path}")
        return s
    except Exception as exc:
        log.warning(f"VIX fetch failed ({exc}) — vix_close will be NaN")
        return pd.Series(dtype=float, name="vix_close")


def fetch_chunk(
    client: StockHistoricalDataClient,
    symbols: list[str],
    start: datetime,
    end: datetime,
    feed: str,
) -> pd.DataFrame:
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame(30, TimeFrameUnit.Minute),
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
        probe_end = min(cursor + timedelta(days=90), end)
        req = StockBarsRequest(
            symbol_or_symbols=[ticker],
            timeframe=TimeFrame(30, TimeFrameUnit.Minute),
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
                if actual > desired_start:
                    log.info(f"  {ticker}: data starts {actual.date()} (pulling from there)")
                return max(desired_start, actual)
        except Exception as exc:
            if "invalid symbol" in str(exc).lower():
                return desired_start
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


def compute_features(
    df: pd.DataFrame,
    spy_ret: pd.Series,
    spy_rvol: pd.Series,
    macro_df: pd.DataFrame,
) -> pd.DataFrame:
    out = df[["open", "high", "low", "close", "volume"]].copy()
    out["ret"] = np.log(out["close"] / out["close"].shift(1))
    out["log_hl_range"] = np.log(out["high"] / out["low"].replace(0, np.nan)).fillna(0.0)
    out["log_oc_gap"] = np.log(out["close"] / out["open"].replace(0, np.nan)).fillna(0.0)
    if "vwap" in df.columns:
        out["log_vwap_dev"] = np.log(out["close"] / df["vwap"].replace(0, np.nan)).fillna(0.0)
    else:
        out["log_vwap_dev"] = 0.0
    aligned_spy_ret = spy_ret.reindex(out.index)
    out["ret_vs_spy"] = out["ret"] - aligned_spy_ret
    rolling_vol_mean = out["volume"].rolling(VOL_WINDOW, min_periods=1).mean()
    out["rel_volume"] = out["volume"] / rolling_vol_mean.replace(0, np.nan)
    ann_factor = math.sqrt(TRADING_MINS_YEAR / 30)
    out["realized_vol"] = out["ret"].rolling(VOL_WINDOW, min_periods=2).std() * ann_factor
    out["spy_ret"] = aligned_spy_ret
    out["spy_realized_vol"] = spy_rvol.reindex(out.index)
    for col in MACRO_COLS:
        if col in macro_df.columns:
            out[col] = _align_daily_causal(macro_df[col], out.index)
        else:
            out[col] = np.nan
    return out


def save_ticker(ticker: str, df: pd.DataFrame, run_date: str) -> Path:
    ticker_dir = RAW_DIR / "bars" / run_date
    ticker_dir.mkdir(parents=True, exist_ok=True)
    path = ticker_dir / f"{ticker}.parquet"
    df.reset_index().to_parquet(path, index=False)
    return path


def save_market(df: pd.DataFrame, run_date: str) -> Path:
    path = RAW_DIR / f"market_{run_date}.parquet"
    df.reset_index().to_parquet(path, index=False)
    return path


def merge_tickers(run_date: str) -> Path:
    files = sorted((RAW_DIR / "bars" / run_date).glob("*.parquet"))
    if not files:
        log.warning("No ticker files to merge")
        return Path()
    combined = pd.concat([pd.read_parquet(f).assign(ticker=f.stem) for f in files], ignore_index=True)
    out = RAW_DIR / f"bars_{run_date}.parquet"
    combined.to_parquet(out, index=False)
    log.info(f"Merged {len(files)} tickers → {out}  ({len(combined):,} rows)")
    return out


def run(
    start_str: Optional[str] = None,
    end_str: Optional[str] = None,
    chunk_days: int = CHUNK_DAYS,
    feed: str = "iex",
    tickers_override: Optional[list[str]] = None,
) -> None:
    global CHUNK_DAYS
    CHUNK_DAYS = chunk_days

    run_date = datetime.now(timezone.utc).strftime("%Y%m%d")
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=LOOKBACK_YEARS * 365)
    if start_str:
        start_dt = datetime.fromisoformat(start_str).replace(tzinfo=timezone.utc)
    if end_str:
        end_dt = datetime.fromisoformat(end_str).replace(tzinfo=timezone.utc)

    console.rule("[bold blue]Stock2Vec — Alpaca Data Pull (merged)[/bold blue]")
    log.info(f"Date range : {start_dt.date()} → {end_dt.date()}")
    log.info(f"Feed       : {feed.upper()}")
    log.info(f"Chunk size : {chunk_days} days")

    key, secret = get_creds()
    validate_keys(key, secret)
    client = StockHistoricalDataClient(api_key=key, secret_key=secret)

    universe = list(dict.fromkeys(tickers_override or get_sp500_tickers()))
    if "SPY" not in universe:
        universe.append("SPY")
    log.info(f"Universe   : {len(universe)} symbols")

    console.print("\n[dim]Fetching SPY …[/dim]")
    spy_raw = fetch_bars(client=client, symbols=["SPY"], start=start_dt, end=end_dt, feed=feed)
    if spy_raw.empty:
        log.error("Could not fetch SPY — aborting")
        sys.exit(1)

    spy_df = spy_raw.xs("SPY", level="symbol") if isinstance(spy_raw.index, pd.MultiIndex) else spy_raw
    spy_ret = np.log(spy_df["close"] / spy_df["close"].shift(1)).rename("spy_ret")
    ann = math.sqrt(TRADING_MINS_YEAR / 30)
    spy_rvol = (spy_ret.rolling(VOL_WINDOW, min_periods=2).std() * ann).rename("spy_realized_vol")

    console.print("[dim]Fetching macro indicators …[/dim]")
    _s, _e = start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")
    macro_df = get_all_macro(_s, _e)

    market_df = pd.DataFrame({
        "spy_ret": spy_ret,
        "spy_realized_vol": spy_rvol,
    })
    for col in MACRO_COLS:
        if col in macro_df.columns:
            market_df[col] = _align_daily_causal(macro_df[col], spy_ret.index)
    mkt_path = save_market(market_df, run_date)
    log.info(f"Market features → {mkt_path}  ({len(market_df):,} bars)")

    ticker_dir = RAW_DIR / "bars" / run_date
    done = {f.stem for f in ticker_dir.glob("*.parquet")} if ticker_dir.exists() else set()
    remaining = [t for t in universe if t not in done]
    log.info(f"Tickers    : {len(remaining)} to fetch  ({len(done)} already done)")

    batch_size = 10
    batches = [remaining[i : i + batch_size] for i in range(0, len(remaining), batch_size)]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching …", total=len(remaining))
        for batch in batches:
            starts: dict[str, datetime] = {}
            for ticker in batch:
                starts[ticker] = find_actual_start(client, ticker, start_dt, end_dt, feed)

            groups: dict[str, list[str]] = {}
            for ticker, start in starts.items():
                groups.setdefault(start.isoformat(), []).append(ticker)

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
                        tkr_df = batch_df.xs(ticker, level="symbol").copy() if isinstance(batch_df.index, pd.MultiIndex) else batch_df.copy()
                        if tkr_df.empty:
                            progress.advance(task, 1)
                            continue
                        feat_df = compute_features(
                            tkr_df,
                            spy_ret,
                            spy_rvol,
                            macro_df,
                        )
                        save_ticker(ticker, feat_df, run_date)
                    except Exception as exc:
                        log.error(f"  {ticker}: feature computation failed — {exc}")
                    finally:
                        progress.advance(task, 1)

    console.print("\n[dim]Merging per-ticker files …[/dim]")
    merged_path = merge_tickers(run_date)

    console.rule("[bold green]Done[/bold green]")
    console.print(f"  Bars file   : [cyan]{merged_path}[/cyan]")
    console.print(f"  Market file : [cyan]{mkt_path}[/cyan]")
    console.print(f"  Per-ticker  : [cyan]{ticker_dir}[/cyan]")
    console.print(
        "\n[dim]Feature columns in bars file:[/dim]\n"
        f"  {', '.join(BAR_FEATURE_COLUMNS)}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pull S&P 500 30-min bars + richer features from Alpaca",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--start", default=None, help="Start date YYYY-MM-DD (default: 5 years ago)")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    p.add_argument("--chunk-days", type=int, default=CHUNK_DAYS, help="Days per API request chunk")
    p.add_argument("--feed", default=os.environ.get("ALPACA_FEED", "iex"), choices=["iex", "sip"], help="Data feed (iex=free, sip=paid)")
    p.add_argument("--tickers", nargs="+", default=None, help="Override universe with specific tickers")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        start_str=args.start,
        end_str=args.end,
        chunk_days=args.chunk_days,
        feed=args.feed,
        tickers_override=args.tickers,
    )
