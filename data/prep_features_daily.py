"""
data/prep_features_daily.py — Preprocess daily Alpaca bars into TNC-ready tensors.

Outputs
-------
data/processed/TNC_features_daily_<YYYYMMDD>.npz
    x           : float32 (N, F, T_max)
    lengths     : int32   (N,)
    tickers     : object  (N,)
    dates       : object  (T_max,)         — shared trading calendar
    macro       : float32 (T_max, M)       — causal z-scored macro per date
    fwd_ret_1d  : float32 (N, T_max)
    fwd_ret_5d  : float32 (N, T_max)

Usage
-----
  uv run python data/prep_features_daily.py
  uv run python data/prep_features_daily.py --date 20260611
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.macro import MACRO_COLS

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "open_scaled",
    "high_scaled",
    "low_scaled",
    "close_scaled",
    "ret",
    "rel_volume",
    "log_hl_range",
    "log_oc_gap",
    "ret_vs_spy",
    "realized_vol",
    "spy_ret",
    "spy_realized_vol",
]

NORM_WINDOW = 20
MIN_BARS = 126
_MACRO_ZSCORE_MIN_PERIODS = 63


def causal_minmax_scale(series: pd.Series, window: int) -> pd.Series:
    rolling_min = series.shift(1).rolling(window, min_periods=1).min()
    rolling_max = series.shift(1).rolling(window, min_periods=1).max()
    denom = (rolling_max - rolling_min).replace(0, 1.0)
    return (series - rolling_min) / denom


def causal_zscore_scale(df: pd.DataFrame) -> pd.DataFrame:
    shifted = df.shift(1)
    expanding_mean = shifted.expanding(min_periods=_MACRO_ZSCORE_MIN_PERIODS).mean()
    expanding_std = shifted.expanding(min_periods=_MACRO_ZSCORE_MIN_PERIODS).std()
    expanding_std = expanding_std.where(expanding_std > 0, other=1.0)
    scaled = (df - expanding_mean) / expanding_std
    return scaled.ffill().bfill()


def build_ticker_series(df: pd.DataFrame) -> pd.DataFrame | None:
    df = df.sort_values("timestamp").reset_index(drop=True)
    if len(df) < MIN_BARS:
        return None

    for col, src in [
        ("open_scaled", "open"),
        ("high_scaled", "high"),
        ("low_scaled", "low"),
        ("close_scaled", "close"),
    ]:
        df[col] = causal_minmax_scale(df[src], NORM_WINDOW).clip(0, 2)

    df[["open_scaled", "high_scaled", "low_scaled", "close_scaled"]] = (
        df[["open_scaled", "high_scaled", "low_scaled", "close_scaled"]].ffill().bfill()
    )

    for col in FEATURE_COLS[4:]:
        df[col] = df[col].fillna(0.0)

    if df[FEATURE_COLS].isnull().any().any():
        return None

    df["fwd_ret_1d"] = (df["close"].shift(-1) / df["close"] - 1.0).fillna(0.0)
    df["fwd_ret_5d"] = (df["close"].shift(-5) / df["close"] - 1.0).fillna(0.0)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prep daily TNC features")
    parser.add_argument("--date", type=str, default=None, help="Run folder YYYYMMDD")
    args = parser.parse_args()

    daily_root = RAW_DIR / "daily"
    if args.date:
        daily_dir = daily_root / args.date
    else:
        dated_dirs = sorted(d for d in daily_root.iterdir() if d.is_dir())
        if not dated_dirs:
            print(f"No daily folders under {daily_root}. Run pull_alpaca_daily.py first.")
            raise SystemExit(1)
        daily_dir = dated_dirs[-1]

    print(f"Processing daily bars from: {daily_dir}")
    parquet_files = sorted(daily_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"No parquet files in {daily_dir}")
        raise SystemExit(1)

    feat_dfs: list[pd.DataFrame] = []
    tickers: list[str] = []
    macro_raw_frames: list[pd.DataFrame] = []
    skipped = 0

    for file_path in parquet_files:
        ticker_name = file_path.stem
        df_raw = pd.read_parquet(file_path)
        df_feat = build_ticker_series(df_raw)
        if df_feat is None:
            skipped += 1
            continue
        feat_dfs.append(df_feat)
        tickers.append(ticker_name)
        if not macro_raw_frames:
            avail = [c for c in MACRO_COLS if c in df_feat.columns]
            if avail:
                macro_raw_frames.append(df_feat[["timestamp"] + avail].copy())

    print(f"Processed {len(feat_dfs)} tickers  ({skipped} skipped)")
    if not feat_dfs:
        raise SystemExit("No usable tickers.")

    ref_idx = tickers.index("SPY") if "SPY" in tickers else int(np.argmax([len(d) for d in feat_dfs]))
    dates = pd.to_datetime(feat_dfs[ref_idx]["timestamp"]).dt.strftime("%Y-%m-%d").values
    T_max = len(dates)

    if macro_raw_frames:
        macro_all = pd.concat(macro_raw_frames, ignore_index=True)
        macro_all["date"] = pd.to_datetime(macro_all["timestamp"]).dt.normalize()
        macro_daily = (
            macro_all.groupby("date")[[c for c in MACRO_COLS if c in macro_all.columns]]
            .last()
            .sort_index()
        )
        macro_daily = macro_daily.reindex(pd.to_datetime(dates)).ffill().bfill().fillna(0.0)
        macro_arr = causal_zscore_scale(macro_daily).to_numpy(dtype=np.float32)
    else:
        macro_arr = np.zeros((T_max, len(MACRO_COLS)), dtype=np.float32)

    F, N = len(FEATURE_COLS), len(feat_dfs)
    x = np.zeros((N, F, T_max), dtype=np.float32)
    fwd_1d = np.zeros((N, T_max), dtype=np.float32)
    fwd_5d = np.zeros((N, T_max), dtype=np.float32)
    lengths = []

    for i, df in enumerate(feat_dfs):
        df = df.copy()
        df["date_str"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d")
        merged = pd.DataFrame({"date_str": dates}).merge(df, on="date_str", how="left")
        valid = merged[FEATURE_COLS[0]].notna()
        lengths.append(int(valid.sum()))
        block = merged[FEATURE_COLS].fillna(0.0).to_numpy(dtype=np.float32).T
        x[i, :, : block.shape[1]] = block[:, :T_max]
        fwd_1d[i, : len(merged)] = merged["fwd_ret_1d"].fillna(0.0).to_numpy(dtype=np.float32)[:T_max]
        fwd_5d[i, : len(merged)] = merged["fwd_ret_5d"].fillna(0.0).to_numpy(dtype=np.float32)[:T_max]

    export_path = PROCESSED_DIR / f"TNC_features_daily_{daily_dir.name}.npz"
    np.savez_compressed(
        export_path,
        x=x,
        lengths=np.array(lengths, dtype=np.int32),
        tickers=np.array(tickers, dtype=object),
        dates=dates,
        macro=macro_arr,
        macro_cols=np.array(MACRO_COLS, dtype=object),
        fwd_ret_1d=fwd_1d,
        fwd_ret_5d=fwd_5d,
    )

    print(f"\nSaved → {export_path}")
    print(f"  Shape    : {x.shape}")
    print(f"  Macro    : {macro_arr.shape}  ({len(MACRO_COLS)} indicators)")
    print(f"  Features : {FEATURE_COLS}")
