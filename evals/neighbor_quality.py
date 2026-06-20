"""
evals/neighbor_quality.py — Neighbor quality analysis for cosine vector search.

Tests whether closer cosine neighbors produce more accurate forward-return
predictions than distant neighbors, and whether that holds across:

  - Horizons : next bar (30min), EOD (1d), week (5d)
  - Slices   : all neighbors, cross-ticker only, per vol regime

Core hypothesis
---------------
If embeddings capture meaningful market state, then:
  1. Low cosine distance  → neighbors' forward returns ≈ query's actual return
  2. High cosine distance → weaker return alignment
  3. Cross-ticker matches should still beat random (generalization)

Metrics per distance quintile (Q1=closest, Q5=farthest)
-------------------------------------------------------
  mae          : mean |weighted_pred_return - actual_return|
  rmse         : root mean squared error
  rank_ic      : Spearman(pred_return, actual_return) across queries
  hit_rate     : fraction of queries where sign(pred) == sign(actual)
  ret_spread   : mean neighbor return std (closer should be tighter)

Usage
-----
  uv run python evals/neighbor_quality.py
  uv run python evals/neighbor_quality.py --k 30 --n-queries 800
  uv run python evals/neighbor_quality.py --daily
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
from scipy.stats import spearmanr

from vector_db.store import DEFAULT_DB_PATH, VectorStore, db_path_for, normalize_vector

RESULTS_DIR = ROOT / "evals" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = {
    "30min": {
        "col": "fwd_ret_30min",
        "label": "Next bar (~30 min)",
    },
    "eod": {
        "col": "fwd_ret_1d",
        "label": "End of day / next close (1d)",
    },
    "week": {
        "col": "fwd_ret_5d",
        "label": "Next week (5 trading days)",
    },
}

REGIME_ORDER = ["low", "mid", "high", "bull", "bear", "sideways", "unknown"]


def _cosine_distances(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Cosine distances from one query to all candidates (1-dot, clipped)."""
    dots = query @ candidates.T
    return np.clip(1.0 - dots, 0.0, 2.0).astype(np.float32)


def _similarity_weights(distances: np.ndarray) -> np.ndarray:
    """Distance → weight (same formula as strategy/outcome.py)."""
    sim = np.clip(1.0 - distances / 2.0, 1e-8, None)
    return sim / sim.sum()


def _date_prefix(ts: str) -> str:
    return str(ts)[:10]


def _eligible_mask(
    tickers: np.ndarray,
    dates: np.ndarray,
    query_idx: int,
    cross_ticker_only: bool,
) -> np.ndarray:
    """Boolean mask of valid neighbors for query i."""
    n = len(tickers)
    mask = np.ones(n, dtype=bool)
    mask[query_idx] = False
    q_ticker = tickers[query_idx]
    q_date = dates[query_idx]
    # Exclude same ticker on same calendar day (production search rule)
    same_day = (tickers == q_ticker) & (dates == q_date)
    mask &= ~same_day
    if cross_ticker_only:
        mask &= tickers != q_ticker
    return mask


def _quintile_bins(distances: np.ndarray, k_neighbors: int) -> list[np.ndarray]:
    """Split sorted neighbor distances into 5 equal-count bins."""
    order = np.argsort(distances)
    if len(order) < 5:
        return [order]
    bins = np.array_split(order[:k_neighbors], 5)
    return bins


def _slice_metrics(
    pred: np.ndarray,
    actual: np.ndarray,
) -> dict:
    err = pred - actual
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    if len(pred) > 5 and np.std(pred) > 1e-12 and np.std(actual) > 1e-12:
        ic, _ = spearmanr(pred, actual)
        rank_ic = float(ic) if not np.isnan(ic) else 0.0
    else:
        rank_ic = 0.0
    hit = float(np.mean(np.sign(pred) == np.sign(actual))) if len(pred) else 0.0
    return {"mae": mae, "rmse": rmse, "rank_ic": rank_ic, "hit_rate": hit, "n": len(pred)}


def analyze_neighbor_quality(
    df: pd.DataFrame,
    *,
    k_neighbors: int = 25,
    n_queries: int | None = 500,
    seed: int = 42,
) -> dict:
    """
    Run distance-stratified neighbor return analysis on indexed LanceDB records.
    """
    vectors = np.stack(df["vector"].apply(lambda v: np.asarray(v, dtype=np.float32)))
    unit = np.stack([normalize_vector(v) for v in vectors])
    tickers = df["ticker"].to_numpy()
    dates = df["timestamp"].apply(_date_prefix).to_numpy()
    regimes = df["regime"].fillna("unknown").astype(str).to_numpy()

    n = len(df)
    rng = np.random.default_rng(seed)
    query_indices = (
        np.arange(n)
        if n_queries is None or n_queries >= n
        else rng.choice(n, size=min(n_queries, n), replace=False)
    )

    results: dict = {
        "n_records": n,
        "n_queries": len(query_indices),
        "k_neighbors": k_neighbors,
        "horizons": {},
    }

    for horizon_key, horizon_meta in HORIZONS.items():
        col = horizon_meta["col"]
        if col not in df.columns:
            continue
        actual_all = df[col].to_numpy(dtype=np.float64)

        horizon_out: dict = {
            "label": horizon_meta["label"],
            "slices": {},
        }

        for slice_name, cross_only in [
            ("all_neighbors", False),
            ("cross_ticker", True),
        ]:
            slice_out: dict = {
                "quintiles": {},
                "top_vs_bottom": {},
                "by_regime": {},
            }

            # Collect per-query predictions for each quintile
            quintile_preds: dict[int, list[float]] = {q: [] for q in range(1, 6)}
            quintile_actuals: dict[int, list[float]] = {q: [] for q in range(1, 6)}
            top_preds, top_actuals = [], []
            bot_preds, bot_actuals = [], []
            all_weighted_preds, all_actuals = [], []
            monotonic_flags = []

            for qi in query_indices:
                mask = _eligible_mask(tickers, dates, qi, cross_only)
                nbr_idx = np.where(mask)[0]
                if len(nbr_idx) < k_neighbors:
                    continue

                dists = _cosine_distances(unit[qi], unit[nbr_idx])
                order = np.argsort(dists)[:k_neighbors]
                sel = nbr_idx[order]
                sel_dists = dists[order]
                sel_rets = actual_all[sel]
                actual = float(actual_all[qi])

                # Full top-k weighted prediction
                w = _similarity_weights(sel_dists)
                pred_full = float(np.dot(w, sel_rets))
                all_weighted_preds.append(pred_full)
                all_actuals.append(actual)

                # Top vs bottom half
                half = max(1, k_neighbors // 2)
                top_pred = float(np.dot(_similarity_weights(sel_dists[:half]), sel_rets[:half]))
                bot_pred = float(np.dot(_similarity_weights(sel_dists[-half:]), sel_rets[-half:]))
                top_preds.append(top_pred)
                top_actuals.append(actual)
                bot_preds.append(bot_pred)
                bot_actuals.append(actual)

                # Quintile bins
                bins = _quintile_bins(sel_dists, k_neighbors)
                q_maes = []
                for q, bin_idx in enumerate(bins, start=1):
                    if len(bin_idx) == 0:
                        continue
                    bd = sel_dists[bin_idx]
                    br = sel_rets[bin_idx]
                    pred_q = float(np.dot(_similarity_weights(bd), br))
                    quintile_preds[q].append(pred_q)
                    quintile_actuals[q].append(actual)
                    q_maes.append(abs(pred_q - actual))

                if len(q_maes) >= 2:
                    monotonic_flags.append(q_maes[-1] >= q_maes[0])

            # Aggregate quintile metrics
            for q in range(1, 6):
                if quintile_preds[q]:
                    slice_out["quintiles"][f"Q{q}"] = _slice_metrics(
                        np.array(quintile_preds[q]),
                        np.array(quintile_actuals[q]),
                    )
                    if q == 1:
                        slice_out["quintiles"][f"Q{q}"]["role"] = "closest"
                    elif q == 5:
                        slice_out["quintiles"][f"Q{q}"]["role"] = "farthest"

            slice_out["overall"] = _slice_metrics(
                np.array(all_weighted_preds), np.array(all_actuals)
            )
            slice_out["top_vs_bottom"] = {
                "closest_half": _slice_metrics(np.array(top_preds), np.array(top_actuals)),
                "farthest_half": _slice_metrics(np.array(bot_preds), np.array(bot_actuals)),
                "mae_improvement_close_vs_far": (
                    _slice_metrics(np.array(bot_preds), np.array(bot_actuals))["mae"]
                    - _slice_metrics(np.array(top_preds), np.array(top_actuals))["mae"]
                ),
            }
            slice_out["monotonic_mae_fraction"] = (
                float(np.mean(monotonic_flags)) if monotonic_flags else 0.0
            )

            # Per-regime (query regime)
            for regime in sorted(set(regimes)):
                r_preds, r_actuals = [], []
                for qi in query_indices:
                    if regimes[qi] != regime:
                        continue
                    mask = _eligible_mask(tickers, dates, qi, cross_only)
                    nbr_idx = np.where(mask)[0]
                    if len(nbr_idx) < k_neighbors:
                        continue
                    dists = _cosine_distances(unit[qi], unit[nbr_idx])
                    order = np.argsort(dists)[:k_neighbors]
                    sel = nbr_idx[order]
                    sel_dists = dists[order]
                    sel_rets = actual_all[sel]
                    w = _similarity_weights(sel_dists)
                    r_preds.append(float(np.dot(w, sel_rets)))
                    r_actuals.append(float(actual_all[qi]))
                if len(r_preds) >= 10:
                    slice_out["by_regime"][regime] = _slice_metrics(
                        np.array(r_preds), np.array(r_actuals)
                    )

            horizon_out["slices"][slice_name] = slice_out

        results["horizons"][horizon_key] = horizon_out

    return results


def _print_report(results: dict) -> None:
    print("\n" + "=" * 72)
    print("  NEIGHBOR QUALITY REPORT — cosine distance vs forward return accuracy")
    print("=" * 72)
    print(f"  Records: {results['n_records']:,}   Queries sampled: {results['n_queries']:,}   k={results['k_neighbors']}")

    for hkey, hdata in results.get("horizons", {}).items():
        print(f"\n── Horizon: {hdata['label']} ({hkey}) ──")
        for slice_name, sdata in hdata.get("slices", {}).items():
            print(f"\n  [{slice_name}]")
            overall = sdata.get("overall", {})
            print(
                f"    Overall weighted k-NN:  MAE={overall.get('mae', 0):.5f}  "
                f"Rank IC={overall.get('rank_ic', 0):.4f}  "
                f"Hit rate={overall.get('hit_rate', 0):.1%}  (n={overall.get('n', 0)})"
            )
            print(f"    Monotonic MAE (far ≥ close): {sdata.get('monotonic_mae_fraction', 0):.1%} of queries")

            tb = sdata.get("top_vs_bottom", {})
            close = tb.get("closest_half", {})
            far = tb.get("farthest_half", {})
            imp = tb.get("mae_improvement_close_vs_far", 0)
            print(
                f"    Closest half MAE={close.get('mae', 0):.5f}  vs  "
                f"Farthest half MAE={far.get('mae', 0):.5f}  "
                f"(Δ MAE close better by {imp:+.5f})"
            )

            print("    Distance quintiles (Q1=closest):")
            for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
                qm = sdata.get("quintiles", {}).get(q, {})
                if not qm:
                    continue
                role = qm.get("role", "")
                print(
                    f"      {q} {role:10s}  MAE={qm.get('mae', 0):.5f}  "
                    f"IC={qm.get('rank_ic', 0):.4f}  Hit={qm.get('hit_rate', 0):.1%}"
                )

            regimes = sdata.get("by_regime", {})
            if regimes:
                print("    By query regime:")
                for regime, rm in sorted(regimes.items()):
                    print(
                        f"      {regime:10s}  MAE={rm.get('mae', 0):.5f}  "
                        f"IC={rm.get('rank_ic', 0):.4f}  Hit={rm.get('hit_rate', 0):.1%}  n={rm.get('n', 0)}"
                    )


def load_lancedb_df(db_path: Path) -> pd.DataFrame:
    store = VectorStore(db_path=db_path)
    store.connect()
    n = store.row_count()
    return store.table.search().limit(n).to_pandas()


def main() -> None:
    parser = argparse.ArgumentParser(description="Neighbor quality analysis (cosine vs forward returns)")
    parser.add_argument("--db", default=None, help="LanceDB path (default: intraday or daily store)")
    parser.add_argument("--daily", action="store_true", help="Use daily LanceDB path")
    parser.add_argument("--k", type=int, default=25, help="Neighbors per query")
    parser.add_argument("--n-queries", type=int, default=500, help="Random queries to sample (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="Output JSON path")
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else db_path_for(args.daily)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        print("Run: uv run python data/create_dummy_db.py")
        sys.exit(1)

    print(f"Loading LanceDB from {db_path} …")
    df = load_lancedb_df(db_path)
    print(f"  {len(df):,} records")

    n_queries = None if args.n_queries <= 0 else args.n_queries
    results = analyze_neighbor_quality(df, k_neighbors=args.k, n_queries=n_queries, seed=args.seed)

    out_path = Path(args.out) if args.out else RESULTS_DIR / "neighbor_quality.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out_path}")

    _print_report(results)


if __name__ == "__main__":
    main()
