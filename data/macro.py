"""
Shared macro factor loaders for Stock2Vec.

Daily macro series are cached under data/raw/macro/ and reused across pulls.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
MACRO_DIR = ROOT / "data" / "raw" / "macro"
MACRO_DIR.mkdir(parents=True, exist_ok=True)

# Canonical macro column order — keep in sync with prep_features_daily.MACRO_COLS
MACRO_COLS = [
    "vix_close",
    "yield_spread",
    "credit_spread",
    "dxy",
    "fed_funds",
    "unemployment",
    "oil_wti",
    "gold",
]


def _load_macro_csv(filename: str, col_name: str) -> pd.Series:
    path = MACRO_DIR / filename
    if not path.exists():
        return pd.Series(dtype=float, name=col_name)
    df = pd.read_csv(path)
    date_col = "date" if "date" in df.columns else df.columns[0]
    val_col = "value" if "value" in df.columns else df.columns[-1]
    s = pd.Series(
        pd.to_numeric(df[val_col].replace(".", np.nan), errors="coerce").values,
        index=pd.to_datetime(df[date_col]),
        name=col_name,
    ).sort_index()
    s.index = s.index.tz_localize("UTC") if s.index.tz is None else s.index.tz_convert("UTC")
    return s.ffill()


def _save_macro_csv(s: pd.Series, filename: str) -> None:
    if s.empty:
        return
    out = s.rename("value").rename_axis("date").reset_index()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(MACRO_DIR / filename, index=False)


def _fetch_yahoo_daily(symbol: str, col_name: str, start: str, end: str) -> pd.Series:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        p1 = int(pd.to_datetime(start).timestamp())
        p2 = int(pd.to_datetime(end).timestamp())
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            f"?period1={p1}&period2={p2}&interval=1d"
        )
        chart = requests.get(url, headers=headers, timeout=15).json()["chart"]["result"][0]
        s = pd.Series(
            chart["indicators"]["quote"][0]["close"],
            index=pd.to_datetime(chart["timestamp"], unit="s"),
            name=col_name,
        )
        s.index = s.index.tz_localize("UTC")
        return s.ffill()
    except Exception as exc:
        log.warning(f"Yahoo {symbol} fetch failed: {exc}")
        return pd.Series(dtype=float, name=col_name)


def _fetch_fred_series(series_id: str, col_name: str, start: str, end: str) -> pd.Series:
    key = os.environ.get("FRED_API_KEY")
    if not key:
        log.warning(f"{col_name}: FRED_API_KEY not set — skipping FRED fetch")
        return pd.Series(dtype=float, name=col_name)
    try:
        url = (
            "https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}&api_key={key}&file_type=json"
            f"&observation_start={start}&observation_end={end}"
        )
        obs = requests.get(url, timeout=20).json()["observations"]
        s = pd.Series(
            pd.to_numeric([o["value"] for o in obs], errors="coerce"),
            index=pd.to_datetime([o["date"] for o in obs]),
            name=col_name,
        ).sort_index()
        s.index = s.index.tz_localize("UTC")
        return s.ffill()
    except Exception as exc:
        log.warning(f"FRED {series_id} fetch failed: {exc}")
        return pd.Series(dtype=float, name=col_name)


def _fetch_treasury_yield_spread(start: str, end: str) -> pd.Series:
    import xml.etree.ElementTree as ET

    ns = {
        "m": "http://schemas.microsoft.com/ado/2007/08/dataservices/metadata",
        "d": "http://schemas.microsoft.com/ado/2007/08/dataservices",
    }
    rows = {}
    for yr in range(pd.to_datetime(start).year, pd.to_datetime(end).year + 1):
        log.info(f"  Treasury: fetching {yr} yield curve …")
        url = (
            "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
            "pages/xml?data=daily_treasury_yield_curve&field_tdr_date_value=" + str(yr)
        )
        try:
            root = ET.fromstring(requests.get(url, timeout=20).text)
        except Exception as exc:
            log.warning(f"Treasury {yr} fetch failed: {exc}")
            continue
        for entry in root.findall(".//m:properties", ns):
            d = entry.find("d:NEW_DATE", ns)
            y2 = entry.find("d:BC_2YEAR", ns)
            y10 = entry.find("d:BC_10YEAR", ns)
            if d is None or y2 is None or y10 is None or not y2.text or not y10.text:
                continue
            rows[pd.to_datetime(d.text)] = float(y10.text) - float(y2.text)
    if not rows:
        return pd.Series(dtype=float, name="yield_spread")
    s = pd.Series(rows, name="yield_spread").sort_index()
    s.index = s.index.tz_localize("UTC")
    return s.ffill()


def get_vix(start: str, end: str) -> pd.Series:
    """Daily VIX close — loads cache or fetches from Yahoo."""
    local = _load_macro_csv("vix.csv", "vix_close")
    if not local.empty:
        return local
    s = _fetch_yahoo_daily("%5EVIX", "vix_close", start, end)
    _save_macro_csv(s, "vix.csv")
    return s


def get_yield_spread(start: str, end: str) -> pd.Series:
    local = _load_macro_csv("yield_spread.csv", "yield_spread")
    if not local.empty:
        return local
    s = _fetch_treasury_yield_spread(start, end)
    _save_macro_csv(s, "yield_spread.csv")
    return s


def get_credit_spread(start: str, end: str) -> pd.Series:
    local = _load_macro_csv("credit_spread.csv", "credit_spread")
    if not local.empty:
        return local
    s = _fetch_fred_series("BAA10Y", "credit_spread", start, end)
    _save_macro_csv(s, "credit_spread.csv")
    return s


def get_dxy(start: str, end: str) -> pd.Series:
    local = _load_macro_csv("dxy.csv", "dxy")
    if not local.empty:
        return local
    s = _fetch_yahoo_daily("DX-Y.NYB", "dxy", start, end)
    _save_macro_csv(s, "dxy.csv")
    return s


def get_fed_funds(start: str, end: str) -> pd.Series:
    local = _load_macro_csv("fed_funds.csv", "fed_funds")
    if not local.empty:
        return local
    s = _fetch_fred_series("FEDFUNDS", "fed_funds", start, end)
    _save_macro_csv(s, "fed_funds.csv")
    return s


def get_unemployment(start: str, end: str) -> pd.Series:
    local = _load_macro_csv("unemployment.csv", "unemployment")
    if not local.empty:
        return local
    s = _fetch_fred_series("UNRATE", "unemployment", start, end)
    _save_macro_csv(s, "unemployment.csv")
    return s


def get_oil_wti(start: str, end: str) -> pd.Series:
    local = _load_macro_csv("oil_wti.csv", "oil_wti")
    if not local.empty:
        return local
    s = _fetch_fred_series("DCOILWTICO", "oil_wti", start, end)
    if s.empty:
        s = _fetch_yahoo_daily("CL=F", "oil_wti", start, end)
    _save_macro_csv(s, "oil_wti.csv")
    return s


def get_gold(start: str, end: str) -> pd.Series:
    local = _load_macro_csv("gold.csv", "gold")
    if not local.empty:
        return local
    s = _fetch_yahoo_daily("GC=F", "gold", start, end)
    _save_macro_csv(s, "gold.csv")
    return s


def get_all_macro(start: str, end: str) -> pd.DataFrame:
    """
    Load or fetch all macro indicators and return a date-indexed DataFrame
    with columns matching MACRO_COLS.
    """
    series = {
        "vix_close":      get_vix(start, end),
        "yield_spread":   get_yield_spread(start, end),
        "credit_spread":  get_credit_spread(start, end),
        "dxy":            get_dxy(start, end),
        "fed_funds":      get_fed_funds(start, end),
        "unemployment":   get_unemployment(start, end),
        "oil_wti":        get_oil_wti(start, end),
        "gold":           get_gold(start, end),
    }
    df = pd.DataFrame(series).sort_index()
    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    return df.ffill().bfill()
