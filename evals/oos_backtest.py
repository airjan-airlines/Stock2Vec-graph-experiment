"""
evals/oos_backtest.py — Out-of-sample IC ranking and k-NN retrieval strategy backtest.

Runs temporal train/test split, cross-sectional rank IC time series, and strategy
metrics: Sharpe, max drawdown, VaR, Expected Shortfall (ES/CVaR).

Usage
-----
  uv run python evals/oos_backtest.py
  uv run python evals/oos_backtest.py --ckpt ckpt/stock2vec_statiocl --k 10
  uv run python evals/oos_backtest.py --daily --ckpt ckpt/stock2vec_daily_macro
  uv run python evals/oos_backtest.py --walk-forward --n-folds 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from evals.encode import (
    WINDOW_DAILY,
    WINDOW_INTRADAY,
    build_embedding_table,
    load_bundle_and_ckpt,
    resolve_device,
)
from evals.metrics import (
    geometric_drift,
    linear_probe,
    neighbor_rank_ic,
    nn_precision_at_k,
    strategy_backtest,
)

RESULTS_DIR = ROOT / "evals" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def temporal_split(
    dates: np.ndarray,
    test_frac: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return boolean train/test masks based on sorted unique dates."""
    uniq = sorted(pd.unique(dates))
    cutoff_idx = int(len(uniq) * (1 - test_frac))
    cutoff = uniq[cutoff_idx] if cutoff_idx < len(uniq) else uniq[-1]
    tr = dates < cutoff
    te = dates >= cutoff
    return tr, te, str(cutoff)


def walk_forward_folds(dates: np.ndarray, n_folds: int = 5) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Expanding-window walk-forward: each fold trains on prior dates, tests on next chunk."""
    uniq = sorted(pd.unique(dates))
    if len(uniq) < n_folds + 2:
        n_folds = max(1, len(uniq) - 2)
    fold_size = max(1, len(uniq) // (n_folds + 1))
    folds = []
    for i in range(1, n_folds + 1):
        test_start_idx = i * fold_size
        test_end_idx = min((i + 1) * fold_size, len(uniq))
        if test_start_idx >= len(uniq):
            break
        train_dates = set(uniq[:test_start_idx])
        test_dates = set(uniq[test_start_idx:test_end_idx])
        if not test_dates:
            continue
        tr = np.isin(dates, list(train_dates))
        te = np.isin(dates, list(test_dates))
        folds.append((tr, te, str(uniq[test_start_idx])))
    return folds


def run_single_split(
    X: np.ndarray,
    y_vol: np.ndarray,
    y_ret: np.ndarray,
    dates: np.ndarray,
    tickers: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
    k: int,
    long_threshold: float,
    short_threshold: float,
    cost_bps: float,
    ann_factor: float,
) -> dict:
    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y_vol[tr], y_vol[te]
    ret_tr, ret_te = y_ret[tr], y_ret[te]
    dates_tr, dates_te = dates[tr], dates[te]
    tickers_tr, tickers_te = tickers[tr], tickers[te]

    ic = neighbor_rank_ic(
        X_tr, ret_tr, X_te, ret_te, dates_te, k=k,
        tickers_tr=tickers_tr, dates_tr=dates_tr, tickers_te=tickers_te,
    )
    strat = strategy_backtest(
        X_tr, ret_tr, dates_tr,
        X_te, ret_te, dates_te,
        tickers_te=tickers_te,
        tickers_tr=tickers_tr,
        k=k,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        cost_bps=cost_bps,
        ann_factor=ann_factor,
    )

    return {
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
        "vol_probe": linear_probe(X_tr, y_tr, X_te, y_te),
        "nn_precision_at_k": nn_precision_at_k(X_tr, y_tr, X_te, y_te, k=k),
        "geometric_drift": geometric_drift(X_tr, y_tr, X_te, y_te),
        "rank_ic": {
            "mean_ic": ic["mean_ic"],
            "std_ic": ic["std_ic"],
            "t_stat": ic["t_stat"],
            "n_dates": ic["n_dates"],
        },
        "ic_series": ic["ic_series"],
        "strategy": {k: v for k, v in strat.items() if k != "daily_pnl" and k != "ic_on_signals"},
        "strategy_ic": strat["ic_on_signals"],
        "daily_pnl": strat["daily_pnl"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="OOS IC + k-NN strategy backtest")
    parser.add_argument("--daily", action="store_true")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--test-frac", type=float, default=0.20)
    parser.add_argument("--long-threshold", type=float, default=0.002)
    parser.add_argument("--short-threshold", type=float, default=-0.002)
    parser.add_argument("--cost-bps", type=float, default=0.0, help="Per-turnover cost in bps")
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--bars-dir", default=None, help="Raw bars dir for fwd returns if not in npz")
    args = parser.parse_args()

    device = resolve_device(args.device)
    bundle, enc, use_macro, ckpt_path, ckpt_meta = load_bundle_and_ckpt(args.daily, args.ckpt, device)

    bars_dir = Path(args.bars_dir) if args.bars_dir else None
    if bars_dir is None and not bundle.is_daily:
        tag = bundle.path.stem.split("_")[-1]
        candidate = ROOT / "data" / "raw" / "bars" / tag
        if candidate.exists():
            bars_dir = candidate
    elif bars_dir is None and bundle.is_daily:
        tag = bundle.path.stem.split("_")[-1]
        candidate = ROOT / "data" / "raw" / "daily" / tag
        if candidate.exists():
            bars_dir = candidate

    print(f"Checkpoint : {ckpt_path}")
    print(f"Encoder    : macro={use_macro}  daily={bundle.is_daily}")
    print(f"Bars dir   : {bars_dir}")

    X, y_vol, y_ret, dates, tickers, _ = build_embedding_table(
        bundle, enc, use_macro, device, bars_dir=bars_dir
    )

    ann_factor = 252.0 if bundle.is_daily else 252.0 * (6.5 * 2)  # ~252 trading days of 30min bars scaled

    if args.walk_forward:
        folds = walk_forward_folds(dates, n_folds=args.n_folds)
        fold_results = []
        for i, (tr, te, cutoff) in enumerate(folds):
            print(f"\n── Fold {i + 1}/{len(folds)}  cutoff={cutoff} ──")
            res = run_single_split(
                X, y_vol, y_ret, dates, tickers, tr, te,
                args.k, args.long_threshold, args.short_threshold,
                args.cost_bps, ann_factor,
            )
            res["cutoff"] = cutoff
            fold_results.append(res)
            print(f"  IC={res['rank_ic']['mean_ic']:.4f}  t={res['rank_ic']['t_stat']:.2f}  "
                  f"Sharpe={res['strategy']['sharpe']:.2f}  VaR={res['strategy']['var_95']:.4f}")

        ic_means = [f["rank_ic"]["mean_ic"] for f in fold_results]
        sharpes = [f["strategy"]["sharpe"] for f in fold_results]
        results = {
            "mode": "walk_forward",
            "ckpt": str(ckpt_path),
            "daily": args.daily,
            "n_folds": len(fold_results),
            "aggregate": {
                "mean_ic": float(np.mean(ic_means)) if ic_means else 0.0,
                "mean_sharpe": float(np.mean(sharpes)) if sharpes else 0.0,
            },
            "folds": fold_results,
        }
    else:
        tr, te, cutoff = temporal_split(dates, test_frac=args.test_frac)
        print(f"\nTemporal split  cutoff={cutoff}  train={tr.sum():,}  test={te.sum():,}")
        split_res = run_single_split(
            X, y_vol, y_ret, dates, tickers, tr, te,
            args.k, args.long_threshold, args.short_threshold,
            args.cost_bps, ann_factor,
        )
        results = {
            "mode": "temporal_holdout",
            "ckpt": str(ckpt_path),
            "daily": args.daily,
            "cutoff": cutoff,
            "n_windows": len(X),
            **split_res,
        }
        print(f"\nRank IC     : {results['rank_ic']['mean_ic']:.4f}  "
              f"t={results['rank_ic']['t_stat']:.2f}  ({results['rank_ic']['n_dates']} dates)")
        print(f"Strategy    : Sharpe={results['strategy']['sharpe']:.2f}  "
              f"MDD={results['strategy']['max_drawdown']:.4f}  "
              f"Hit={results['strategy']['hit_rate']:.1%}")
        print(f"Risk        : VaR(95%)={results['strategy']['var_95']:.4f}  "
              f"ES(95%)={results['strategy']['es_95']:.4f}")

    out = RESULTS_DIR / f"oos_{'daily' if args.daily else 'intraday'}_{ckpt_path.parent.name}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
