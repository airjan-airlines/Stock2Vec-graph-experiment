"""
create_dummy_db.py

Generates a synthetic LanceDB database that simulates what the real pipeline
will eventually consume: TCN-produced temporal state embeddings for a set of
tickers across a historical date range.

Data model
----------
- Price data is simulated at 30-minute bar resolution (13 bars/day for a
  standard 9:30–16:00 US equity session).
- One embedding is produced per ticker per trading day, anchored at the
  daily close bar (16:00). This mirrors the real workflow where the TCN
  runs once at end-of-day on the latest intraday history.
- The embedding encodes the last EMBEDDING_WINDOW_BARS 30-min bars
  (default 390 = 30 trading days of intraday data).
- Forward returns are computed at three horizons and stored as labels:
    fwd_ret_30min  — return of the very next 30-min bar (open of next session)
    fwd_ret_1d     — next trading day's close-to-close return
    fwd_ret_5d     — next 5 trading days' close-to-close return

Schema per record
-----------------
  vector        : float32[64]   — embedding (simulates TCN output)
  ticker        : str           — e.g. "AAPL"
  timestamp     : str           — ISO-8601 of the close bar that ends the window
                                  e.g. "2023-06-15T16:00:00"
  price         : float         — close price at timestamp
  fwd_ret_30min : float         — return of next 30-min bar
  fwd_ret_1d    : float         — 1-day forward return (next close / this close - 1)
  fwd_ret_5d    : float         — 5-day forward return
  regime        : str           — synthetic regime tag ("bull", "bear", "sideways")

Run with:
    uv run python data/create_dummy_db.py
"""

from __future__ import annotations

import sys
from datetime import datetime, time, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from vector_db.store import VectorStore, db_path_for

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH = db_path_for(daily=False)
TABLE_NAME = "embeddings"

EMBEDDING_DIM = 64           # matches trained encoder / real pipeline output
EMBEDDING_WINDOW_BARS = 390  # lookback window fed to TCN: 390 bars = 30 trading days

TICKERS = ["AAPL", "NVDA", "MSFT", "AMZN", "TSLA", "META", "GOOGL", "JPM", "SPY", "QQQ"]
START_DATE = datetime(2020, 1, 2)
END_DATE   = datetime(2024, 12, 31)

# US equity session: 9:30 – 16:00 → 13 bars of 30 min each
SESSION_OPEN  = time(9, 30)
SESSION_CLOSE = time(16, 0)
BARS_PER_DAY  = 13   # 9:30 10:00 10:30 … 15:30 16:00

rng = np.random.default_rng(42)


# ── Date / bar helpers ────────────────────────────────────────────────────────

def _trading_days(start: datetime, end: datetime) -> list[datetime]:
    """Weekday dates between start and end (Mon–Fri proxy for trading days)."""
    days, cur = [], start
    while cur <= end:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    return days


def _intraday_bars(day: datetime) -> list[datetime]:
    """Return the 13 bar-close timestamps for a single trading day."""
    bars = []
    t = datetime(day.year, day.month, day.day, 9, 30)
    for _ in range(BARS_PER_DAY):
        t += timedelta(minutes=30)
        bars.append(t)
    return bars  # last element is always 16:00


# ── Price simulation ──────────────────────────────────────────────────────────

def _simulate_30min_series(n_bars: int, start_price: float) -> np.ndarray:
    """
    Geometric Brownian Motion on 30-min bars.
    Annualised vol ~20 % → per-bar vol ≈ 0.20 / sqrt(252 * 13) ≈ 0.0035.
    """
    per_bar_vol  = 0.20 / np.sqrt(252 * BARS_PER_DAY)
    per_bar_drift = 0.08 / (252 * BARS_PER_DAY)   # ~8 % annual drift
    log_returns = rng.normal(per_bar_drift, per_bar_vol, size=n_bars)
    prices = start_price * np.exp(np.cumsum(log_returns))
    return prices.astype(np.float32)


# ── Regime detection ──────────────────────────────────────────────────────────

def _regime(prices: np.ndarray, bar_idx: int, window_bars: int = BARS_PER_DAY * 20) -> str:
    """
    Assign a regime based on return over the last `window_bars` bars.
    Thresholds are calibrated to 30-min bar returns.
    """
    start = max(0, bar_idx - window_bars)
    if bar_idx <= start:
        return "sideways"
    ret = (prices[bar_idx] - prices[start]) / prices[start]
    if ret > 0.05:
        return "bull"
    if ret < -0.05:
        return "bear"
    return "sideways"


# ── Embedding simulation ──────────────────────────────────────────────────────

def _make_embedding(prices: np.ndarray, bar_idx: int) -> np.ndarray:
    """
    Simulate a TCN embedding from the last EMBEDDING_WINDOW_BARS 30-min bars.

    In production this is the actual model output. Here we encode:
      - Normalised price shape of the window (captures pattern)
      - Per-bar log returns statistics (captures momentum / vol)
      - Controlled noise so vectors aren't identical across tickers

    The result is L2-normalised to the unit sphere, which makes cosine
    distance equivalent to dot-product similarity — standard for ANN search.
    """
    start = max(0, bar_idx - EMBEDDING_WINDOW_BARS + 1)
    window = prices[start : bar_idx + 1]   # shape: (≤ EMBEDDING_WINDOW_BARS,)

    n = len(window)
    features = np.zeros(EMBEDDING_DIM, dtype=np.float32)

    # ── Feature block 1: normalised price shape (first DIM/2 dims) ──────────
    # Resample window to DIM/2 points via linear interpolation so the feature
    # vector is always the same length regardless of how much history exists.
    half = EMBEDDING_DIM // 2
    indices = np.linspace(0, n - 1, half)
    resampled = np.interp(indices, np.arange(n), window).astype(np.float32)
    w_min, w_max = resampled.min(), resampled.max()
    if w_max > w_min:
        features[:half] = (resampled - w_min) / (w_max - w_min)

    # ── Feature block 2: rolling statistics (second DIM/2 dims) ─────────────
    log_rets = np.diff(np.log(window + 1e-8)).astype(np.float32)
    if len(log_rets) > 0:
        quarter = EMBEDDING_DIM // 4
        # Momentum at multiple lookbacks (1d, 5d, 20d in bars)
        for i, lb in enumerate([BARS_PER_DAY, BARS_PER_DAY * 5, BARS_PER_DAY * 20]):
            lb = min(lb, len(log_rets))
            features[half + i] = float(log_rets[-lb:].mean())
            features[half + i + quarter] = float(log_rets[-lb:].std())

    # ── Noise ────────────────────────────────────────────────────────────────
    features += rng.normal(0, 0.03, size=EMBEDDING_DIM).astype(np.float32)

    # ── L2 normalise ─────────────────────────────────────────────────────────
    norm = np.linalg.norm(features)
    if norm > 0:
        features /= norm

    return features


# ── Main builder ──────────────────────────────────────────────────────────────

def build_database() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    trading_days = _trading_days(START_DATE, END_DATE)
    n_days = len(trading_days)

    print(f"Simulating 30-min price series for {len(TICKERS)} tickers …")
    print(f"  {n_days} trading days × {BARS_PER_DAY} bars/day = {n_days * BARS_PER_DAY:,} bars per ticker")
    print(f"  Embedding window : {EMBEDDING_WINDOW_BARS} bars ({EMBEDDING_WINDOW_BARS // BARS_PER_DAY} trading days)")
    print(f"  Embedding dim    : {EMBEDDING_DIM}")
    print(f"  Labels           : fwd_ret_30min, fwd_ret_1d, fwd_ret_5d\n")

    # Build a flat list of all bar timestamps (all days × 13 bars)
    all_bar_times: list[datetime] = []
    for day in trading_days:
        all_bar_times.extend(_intraday_bars(day))
    n_bars_total = len(all_bar_times)

    # Index of each day's close bar (bar 12, 0-indexed within the day)
    close_bar_indices = [d * BARS_PER_DAY + (BARS_PER_DAY - 1) for d in range(n_days)]

    records = []

    for ticker in TICKERS:
        start_price = float(rng.uniform(50, 500))
        prices = _simulate_30min_series(n_bars_total, start_price)

        for day_idx, close_idx in enumerate(close_bar_indices):
            ts = all_bar_times[close_idx]  # e.g. 2020-01-02T16:00:00

            # ── Forward returns ──────────────────────────────────────────────
            # fwd_ret_30min: return of the very next bar (first bar of next session)
            next_bar_idx = close_idx + 1
            fwd_ret_30min = (
                float((prices[next_bar_idx] - prices[close_idx]) / prices[close_idx])
                if next_bar_idx < n_bars_total else 0.0
            )

            # fwd_ret_1d: next trading day's close vs. today's close
            next_close_idx = close_idx + BARS_PER_DAY
            fwd_ret_1d = (
                float((prices[next_close_idx] - prices[close_idx]) / prices[close_idx])
                if next_close_idx < n_bars_total else 0.0
            )

            # fwd_ret_5d: close 5 trading days from now vs. today's close
            close_5d_idx = close_idx + BARS_PER_DAY * 5
            fwd_ret_5d = (
                float((prices[close_5d_idx] - prices[close_idx]) / prices[close_idx])
                if close_5d_idx < n_bars_total else 0.0
            )

            embedding = _make_embedding(prices, close_idx)
            regime    = _regime(prices, close_idx)

            # Rolling vol proxy for schema compatibility
            lookback = min(BARS_PER_DAY * 5, close_idx)
            if lookback > 1:
                seg = prices[close_idx - lookback : close_idx + 1]
                log_rets = np.diff(np.log(seg + 1e-8))
                realized_vol = float(np.std(log_rets) * np.sqrt(252 * BARS_PER_DAY))
            else:
                realized_vol = 0.2

            records.append({
                "vector":        embedding,
                "ticker":        ticker,
                "timestamp":     ts.isoformat(),
                "price":         float(prices[close_idx]),
                "realized_vol":  realized_vol,
                "fwd_ret_30min": fwd_ret_30min,
                "fwd_ret_1d":    fwd_ret_1d,
                "fwd_ret_5d":    fwd_ret_5d,
                "regime":        regime,
            })

        print(f"  ✓ {ticker}  ({len(close_bar_indices)} records)")

    # ── Write to LanceDB (cosine IVF-PQ index) ───────────────────────────────
    store = VectorStore(db_path=DB_PATH, table_name=TABLE_NAME)
    store.create_table(records, embedding_dim=EMBEDDING_DIM, replace=True, build_index=True)

    print(f"\n✓ Database written to : {DB_PATH}")
    print(f"  Table  : '{TABLE_NAME}'")
    print(f"  Rows   : {len(records):,}  ({len(TICKERS)} tickers × {n_days} days)")
    print(f"  Index  : cosine IVF-PQ  (metric=cosine, dim={EMBEDDING_DIM})")


if __name__ == "__main__":
    build_database()
