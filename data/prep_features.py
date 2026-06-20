"""
data/prep_features.py — Preprocess raw Alpaca bars into continuous per-ticker
time series tensors ready for TNCDataset.

Outputs
-------
data/processed/TNC_features_<YYYYMMDD>.npz
    x           : float32 (N, F, T_max)   — intraday feature tensor for TNC
                    N     = number of tickers
                    F     = len(FEATURE_COLS) = 15
                    T_max = longest ticker series; shorter ones zero-padded
    lengths     : int32   (N,)             — unpadded length of each ticker
    tickers     : object  (N,)             — ticker symbol strings
    macro       : float32 (D, M)           — z-scored daily macro snapshot
                    D     = number of unique trading dates in the dataset
                    cols  = vix_close, yield_spread, credit_spread, dxy,
                            fed_funds, unemployment, oil_wti, gold
    macro_dates : object  (D,)             — ISO date strings (YYYY-MM-DD)
                    maps rows of `macro` to calendar dates

Design notes
------------
- No pre-slicing. TNCDataset receives the full continuous series and does
  its own random sub-window sampling internally.
- Price features are normalised with a *causal* rolling min-max window of
  size NORM_WINDOW bars so no future bar values influence the scaled price
  seen at time t.
- Intraday scale-free features (ret, rel_volume, log_hl_range, etc.) need no
  further normalisation.
- Macro features are NOT fed into the TNC encoder. They change only once per
  day (13 identical bars), so they carry no contrastive signal at 30-min
  resolution. Instead they are exported as a separate (D, 4) array for use
  as conditioning context in the downstream task head.
- Macro normalisation uses a *causal expanding z-score*: day t's mean and std
  are computed from days [0, t-1] only (min 63 days). This mirrors the causal
  min-max used for OHLC and prevents any future macro regime leaking into the
  representation.
- Tickers with fewer than MIN_BARS bars are skipped.
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR     = Path(__file__).resolve().parent.parent
RAW_DIR      = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Intraday feature channels fed into the TNC encoder.
# Must match FEATURE_COLS in models/utils.py  (F = 15).
FEATURE_COLS = [
    # ── Causal OHLC (rolling min-max scaled) ──────────────────────────────
    "open_scaled",
    "high_scaled",
    "low_scaled",
    "close_scaled",
    # ── Scale-free intraday features ──────────────────────────────────────
    "ret",           # log return bar-over-bar
    "rel_volume",    # volume / rolling-20-bar mean
    "log_hl_range",  # log(high/low)  — intrabar vol proxy
    "log_oc_gap",    # log(close/open) — directional bar move
    "log_vwap_dev",  # log(close/vwap) — price vs avg trade price
    "ret_vs_spy",    # ticker ret minus SPY ret (relative alpha)
    "realized_vol",  # rolling-20-bar annualised vol of ret
    # ── SPY market context (30-min, same bar) ─────────────────────────────
    "spy_ret",
    "spy_realized_vol",
    # ── Time-of-day encoding ──────────────────────────────────────────────
    "sin_time",
    "cos_time",
]

from data.macro import MACRO_COLS

# Rolling window for causal min-max normalisation of OHLC prices.
# 65 bars = one full trading week of 30-min bars.
NORM_WINDOW = 65

# Tickers shorter than this are dropped (too little data for TNC).
# 4 * TNC window_size minimum; we require at least 5× for ADF to be useful.
MIN_BARS = 650   # ~50 trading days of 30-min bars


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    timestamps = pd.to_datetime(df["timestamp"])
    minutes_since_midnight = timestamps.dt.hour * 60 + timestamps.dt.minute
    df["sin_time"] = np.sin(2 * np.pi * minutes_since_midnight / 1440)
    df["cos_time"] = np.cos(2 * np.pi * minutes_since_midnight / 1440)
    return df


def causal_minmax_scale(series: pd.Series, window: int) -> pd.Series:
    """
    Scale each bar using only the rolling min/max of the past `window` bars.
    This is strictly causal — bar t only sees bars [t-window, t-1].
    Falls back to expanding window at the start of the series.
    """
    rolling_min = series.shift(1).rolling(window, min_periods=1).min()
    rolling_max = series.shift(1).rolling(window, min_periods=1).max()
    denom = (rolling_max - rolling_min).replace(0, 1.0)
    return (series - rolling_min) / denom


# Minimum number of prior days required before the expanding z-score is
# considered stable.  One quarter of trading days (~63) is a reasonable floor.
_MACRO_ZSCORE_MIN_PERIODS = 63


def causal_zscore_scale(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a causal expanding z-score to each column of `df` (a date-indexed
    daily DataFrame).  Day t is normalised using only the mean and std of
    days [0, t-1] — no future data leaks into the scaling.

    Rows before `_MACRO_ZSCORE_MIN_PERIODS` have NaN stats and are filled
    forward from the first valid scaled value (then backward for any leading
    NaNs), so the output is always fully populated.
    """
    shifted = df.shift(1)  # exclude current day from its own stats
    expanding_mean = shifted.expanding(min_periods=_MACRO_ZSCORE_MIN_PERIODS).mean()
    expanding_std  = shifted.expanding(min_periods=_MACRO_ZSCORE_MIN_PERIODS).std()
    expanding_std  = expanding_std.where(expanding_std > 0, other=1.0)  # avoid /0
    scaled = (df - expanding_mean) / expanding_std
    return scaled.ffill().bfill()  # fill warm-up rows with first valid value


def build_ticker_series(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Given a raw per-ticker DataFrame (from pull_alpaca_merged), compute all
    intraday features and return a cleaned DataFrame sorted by timestamp.
    Returns None if the ticker is too short or has fatally bad data.

    Macro columns (MACRO_COLS) are intentionally left untouched here — they
    are extracted separately in __main__ and exported as a distinct array.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    if len(df) < MIN_BARS:
        return None

    # Time-of-day features
    df = add_time_features(df)

    # ── Causal OHLC scaling ────────────────────────────────────────────────
    for col, src in [("open_scaled",  "open"),
                     ("high_scaled",  "high"),
                     ("low_scaled",   "low"),
                     ("close_scaled", "close")]:
        df[col] = causal_minmax_scale(df[src], NORM_WINDOW).clip(0, 2)

    df[["open_scaled", "high_scaled", "low_scaled", "close_scaled"]] = (
        df[["open_scaled", "high_scaled", "low_scaled", "close_scaled"]]
        .ffill().bfill()
    )

    # ── Scale-free intraday features (already computed by puller) ──────────
    df["ret"]              = df["ret"].fillna(0.0)
    df["rel_volume"]       = df["rel_volume"].fillna(1.0).clip(0, 10)
    df["log_hl_range"]     = df["log_hl_range"].fillna(0.0)
    df["log_oc_gap"]       = df["log_oc_gap"].fillna(0.0)
    df["log_vwap_dev"]     = df["log_vwap_dev"].fillna(0.0)
    df["ret_vs_spy"]       = df["ret_vs_spy"].fillna(0.0)
    df["realized_vol"]     = df["realized_vol"].fillna(0.0)
    df["spy_ret"]          = df["spy_ret"].fillna(0.0)
    df["spy_realized_vol"] = df["spy_realized_vol"].fillna(0.0)

    # Sanity check: drop if any intraday feature column is still NaN
    if df[FEATURE_COLS].isnull().any().any():
        return None

    return df


def df_to_feature_array(df: pd.DataFrame) -> np.ndarray:
    """Return float32 array of shape (F, T) for one ticker."""
    return df[FEATURE_COLS].to_numpy(dtype=np.float32).T  # (F, T)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prep continuous TNC features")
    parser.add_argument("--date", type=str, default=None,
                        help="Bar folder date YYYYMMDD (default: most recent)")
    args = parser.parse_args()

    # ── Locate bar folder ──────────────────────────────────────────────────
    bars_root = RAW_DIR / "bars"
    if args.date:
        daily_dir = bars_root / args.date
    else:
        dated_dirs = sorted([d for d in bars_root.iterdir() if d.is_dir()])
        if not dated_dirs:
            print(f"No bar folders found under {bars_root}")
            raise SystemExit(1)
        daily_dir = dated_dirs[-1]

    print(f"Processing bars from: {daily_dir}")

    parquet_files = sorted(daily_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"No parquet files in {daily_dir}")
        raise SystemExit(1)

    print(f"Found {len(parquet_files)} ticker files")

    # ── Per-ticker intraday processing ─────────────────────────────────────
    feat_dfs = []   # cleaned DataFrames (still contain macro cols for extraction)
    lengths  = []   # series length per ticker
    tickers  = []   # ticker symbol strings
    macro_frames = []  # raw macro slices collected across all tickers

    skipped = 0
    for file_path in parquet_files:
        ticker_name = file_path.stem
        df_raw = pd.read_parquet(file_path)
        if "symbol" not in df_raw.columns:
            df_raw["symbol"] = ticker_name

        df_feat = build_ticker_series(df_raw)
        if df_feat is None:
            skipped += 1
            continue

        feat_dfs.append(df_feat)
        lengths.append(len(df_feat))
        tickers.append(ticker_name)

        # Collect raw macro rows — macro values are identical across tickers
        # on the same date, so we only need one ticker's worth.
        if ticker_name == tickers[0]:
            avail = [c for c in MACRO_COLS if c in df_feat.columns]
            if avail:
                macro_slice = df_feat[["timestamp"] + avail].copy()
                macro_frames.append(macro_slice)

    print(f"Processed {len(feat_dfs)} tickers  ({skipped} skipped — too short or bad data)")

    if not feat_dfs:
        print("No usable tickers. Exiting.")
        raise SystemExit(1)

    # ── Build macro table: one row per trading date ────────────────────────
    # All tickers share the same daily macro values, so we de-duplicate by
    # date and z-score normalise globally.
    if macro_frames:
        macro_all = pd.concat(macro_frames, ignore_index=True)
        macro_all["date"] = pd.to_datetime(macro_all["timestamp"]).dt.normalize()
        # Take last bar of each day (forward-fill already applied by puller)
        macro_daily = (
            macro_all.groupby("date")[[c for c in MACRO_COLS if c in macro_all.columns]]
            .last()
            .sort_index()
        )
        # Fill any missing dates forward then back
        macro_daily = macro_daily.ffill().bfill().fillna(0.0)
        # Causal expanding z-score — each day normalised by strictly prior stats
        macro_scaled = causal_zscore_scale(macro_daily)
        print("\nMacro feature stats (raw, full-range for reference only):")
        for col in macro_daily.columns:
            print(f"  {col:20s}  mean={macro_daily[col].mean():.4f}  "
                  f"std={macro_daily[col].std():.4f}")
        macro_arr   = macro_scaled.to_numpy(dtype=np.float32)        # (D, M)
        macro_dates = macro_scaled.index.strftime("%Y-%m-%d").values  # (D,)
    else:
        print("Warning: no macro columns found in parquet files — macro array will be empty.")
        macro_arr   = np.zeros((0, len(MACRO_COLS)), dtype=np.float32)
        macro_dates = np.array([], dtype=object)

    # ── Pad intraday arrays to (N, F, T_max) ──────────────────────────────
    feature_arrays = [df_to_feature_array(df) for df in feat_dfs]
    T_max = max(lengths)
    F     = len(FEATURE_COLS)
    N     = len(feature_arrays)
    x = np.zeros((N, F, T_max), dtype=np.float32)
    for i, arr in enumerate(feature_arrays):
        x[i, :, :arr.shape[1]] = arr

    # ── Save ───────────────────────────────────────────────────────────────
    export_path = PROCESSED_DIR / f"TNC_features_{daily_dir.name}.npz"
    np.savez_compressed(
        export_path,
        x           = x,
        lengths     = np.array(lengths, dtype=np.int32),
        tickers     = np.array(tickers, dtype=object),
        macro       = macro_arr,
        macro_dates = macro_dates,
    )

    print(f"\nSaved → {export_path}")
    print(f"  Intraday : {x.shape}  (N={N}, F={F}, T_max={T_max})")
    print(f"  Macro    : {macro_arr.shape}  (D dates × {len(MACRO_COLS)} features)")
    print(f"  Min len  : {min(lengths)} bars")
    print(f"  Max len  : {max(lengths)} bars")
    print(f"  Median   : {int(np.median(lengths))} bars")
    print(f"  Features : {FEATURE_COLS}")
    print(f"  Macro cols: {MACRO_COLS}")
