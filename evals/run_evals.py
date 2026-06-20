"""
evals/run_evals.py — Stock2Vec evaluation framework.

Runs embedding-quality probes and OOS strategy metrics via shared evals/metrics.py.

Usage
-----
  uv run python evals/run_evals.py
  uv run python evals/run_evals.py --daily --ckpt ckpt/stock2vec_daily_macro
  uv run python evals/run_evals.py --ckpt ckpt/stock2vec_statiocl --oos
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from evals.encode import build_embedding_table, load_bundle_and_ckpt, resolve_device
from evals.metrics import (
    geometric_drift,
    linear_probe,
    neighbor_rank_ic,
    nn_precision_at_k,
    strategy_backtest,
)
from evals.oos_backtest import temporal_split

RESULTS_DIR = ROOT / "evals" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TEST_FRAC = 0.20
KNN_K = 10


def main() -> None:
    parser = argparse.ArgumentParser(description="Stock2Vec evaluation framework")
    parser.add_argument("--daily", action="store_true", help="Use daily feature npz")
    parser.add_argument("--ckpt", default=None, help="Checkpoint path or ckpt/ subfolder name")
    parser.add_argument("--device", default=None)
    parser.add_argument("--k", type=int, default=KNN_K)
    parser.add_argument("--test-frac", type=float, default=TEST_FRAC)
    parser.add_argument("--plot", action="store_true", help="Also run UMAP plot via evals/plots.py")
    parser.add_argument("--oos", action="store_true",
                        help="Run full OOS backtest harness (evals/oos_backtest.py)")
    args = parser.parse_args()

    if args.oos:
        cmd = [sys.executable, str(ROOT / "evals" / "oos_backtest.py")]
        if args.daily:
            cmd.append("--daily")
        if args.ckpt:
            cmd += ["--ckpt", args.ckpt]
        if args.device:
            cmd += ["--device", args.device]
        cmd += ["--k", str(args.k), "--test-frac", str(args.test_frac)]
        raise SystemExit(subprocess.call(cmd))

    device = resolve_device(args.device)
    bundle, enc, use_macro, ckpt_path, _ = load_bundle_and_ckpt(args.daily, args.ckpt, device)
    print(f"Loaded encoder from {ckpt_path}  macro={use_macro}  daily={bundle.is_daily}")

    tag = bundle.path.stem.split("_")[-1]
    bars_dir = (
        ROOT / "data" / "raw" / ("daily" if bundle.is_daily else "bars") / tag
    )
    if not bars_dir.exists():
        bars_dir = None

    X, y_vol, y_ret, dates, tickers, _ = build_embedding_table(
        bundle, enc, use_macro, device, bars_dir=bars_dir
    )

    tr, te, cutoff = temporal_split(dates, test_frac=args.test_frac)
    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y_vol[tr], y_vol[te]
    ret_tr, ret_te = y_ret[tr], y_ret[te]
    dates_tr, dates_te = dates[tr], dates[te]
    tickers_tr, tickers_te = tickers[tr], tickers[te]

    ann_factor = 252.0 if bundle.is_daily else 252.0 * (6.5 * 2)
    ic = neighbor_rank_ic(
        X_tr, ret_tr, X_te, ret_te, dates_te, k=args.k,
        tickers_tr=tickers_tr, dates_tr=dates_tr, tickers_te=tickers_te,
    )
    strat = strategy_backtest(
        X_tr, ret_tr, dates_tr, X_te, ret_te, dates_te,
        tickers_te=tickers_te, tickers_tr=tickers_tr, k=args.k, ann_factor=ann_factor,
    )

    results = {
        "ckpt": str(ckpt_path),
        "daily": args.daily,
        "cutoff": cutoff,
        "n_windows": len(X),
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
        "vol_probe": linear_probe(X_tr, y_tr, X_te, y_te),
        "nn_precision_at_k": nn_precision_at_k(X_tr, y_tr, X_te, y_te, k=args.k),
        "rank_ic": {
            "mean_ic": ic["mean_ic"],
            "std_ic": ic["std_ic"],
            "t_stat": ic["t_stat"],
            "n_dates": ic["n_dates"],
        },
        "geometric_drift": geometric_drift(X_tr, y_tr, X_te, y_te),
        "strategy": {k: v for k, v in strat.items() if k not in ("daily_pnl", "ic_on_signals")},
    }

    out_path = RESULTS_DIR / f"eval_{'daily' if args.daily else 'intraday'}_{ckpt_path.parent.name}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")
    print(json.dumps(results, indent=2))

    if args.plot:
        subprocess.run([sys.executable, "evals/plots.py"], cwd=ROOT, check=False)


if __name__ == "__main__":
    main()
