"""
evals/metrics.py — Shared evaluation metrics for Stock2Vec.

Canonical implementations used by run_evals.py, oos_backtest.py, and embed flows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors


def similarity_weights(distances: np.ndarray) -> np.ndarray:
    """Cosine distance → normalized similarity weights (matches strategy/outcome.py)."""
    sim = np.clip(1.0 - distances / 2.0, 1e-8, None)
    return sim / sim.sum()


def _knn_with_exclusion(
    nn: NearestNeighbors,
    X_te: np.ndarray,
    k: int,
    tickers_tr: np.ndarray | None,
    dates_tr: np.ndarray | None,
    tickers_te: np.ndarray | None,
    dates_te: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return k nearest train neighbors per test row, excluding same ticker on same day
    (matches production vector search rules in strategy/search.py).
    """
    n_train = nn.n_samples_fit_
    overfetch = min(n_train, max(k * 5, k + 20))
    dists_all, idx_all = nn.kneighbors(X_te, n_neighbors=overfetch)

    use_exclusion = (
        tickers_tr is not None
        and dates_tr is not None
        and tickers_te is not None
    )

    n_query = len(X_te)
    dists_out = np.zeros((n_query, k), dtype=np.float64)
    idx_out = np.zeros((n_query, k), dtype=int)

    for i in range(n_query):
        eligible: list[tuple[float, int]] = []
        q_ticker = tickers_te[i] if use_exclusion else None
        q_date = dates_te[i]

        for dist, tr_idx in zip(dists_all[i], idx_all[i]):
            if use_exclusion:
                if tickers_tr[tr_idx] == q_ticker:
                    continue
                if dates_tr[tr_idx] == q_date:
                    continue
            eligible.append((float(dist), int(tr_idx)))
            if len(eligible) >= k:
                break

        if not eligible:
            eligible = [(float(dists_all[i, 0]), int(idx_all[i, 0]))]

        while len(eligible) < k:
            eligible.append(eligible[-1])

        for j in range(k):
            dists_out[i, j], idx_out[i, j] = eligible[j]

    return dists_out, idx_out


def linear_probe(X_tr, y_tr, X_te, y_te) -> dict:
    if len(np.unique(y_tr)) < 2:
        return {"accuracy": 0.0, "macro_f1": 0.0}
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    return {
        "accuracy": float(accuracy_score(y_te, preds)),
        "macro_f1": float(f1_score(y_te, preds, average="macro", zero_division=0)),
    }


def nn_precision_at_k(X_tr, y_tr, X_te, y_te, k: int = 10) -> float:
    if len(X_tr) < k or len(X_te) == 0:
        return 0.0
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(X_tr)
    _, idx = nn.kneighbors(X_te)
    hits = np.mean([np.mean(y_tr[nbrs] == lbl) for lbl, nbrs in zip(y_te, idx)])
    return float(hits)


def neighbor_rank_ic(
    X_tr: np.ndarray,
    fwd_tr: np.ndarray,
    X_te: np.ndarray,
    fwd_te: np.ndarray,
    dates_te: np.ndarray,
    k: int = 10,
    min_names: int = 5,
    tickers_tr: np.ndarray | None = None,
    dates_tr: np.ndarray | None = None,
    tickers_te: np.ndarray | None = None,
) -> dict:
    """
    Cross-sectional rank IC: k-NN predicted forward return vs actual, by date.

    Neighbors exclude the query ticker entirely and any train rows on the same
    calendar day (matches production vector_search.py / vector_db/store.py rules).
    """
    if len(X_tr) < k or len(X_te) == 0:
        return {"mean_ic": 0.0, "std_ic": 0.0, "t_stat": 0.0, "n_dates": 0, "ic_series": []}

    nn = NearestNeighbors(metric="cosine")
    nn.fit(X_tr)
    dists, idx = _knn_with_exclusion(
        nn, X_te, k, tickers_tr, dates_tr, tickers_te, dates_te,
    )

    pred = np.array([np.dot(similarity_weights(d), fwd_tr[i]) for d, i in zip(dists, idx)])

    df = pd.DataFrame({"date": dates_te, "pred": pred, "actual": fwd_te})
    ic_vals: list[float] = []
    for _, grp in df.groupby("date"):
        if len(grp) < min_names:
            continue
        if grp["pred"].std() < 1e-12 or grp["actual"].std() < 1e-12:
            continue
        ic, _ = spearmanr(grp["pred"], grp["actual"])
        if not np.isnan(ic):
            ic_vals.append(float(ic))

    if not ic_vals:
        return {"mean_ic": 0.0, "std_ic": 0.0, "t_stat": 0.0, "n_dates": 0, "ic_series": []}

    mean_ic = float(np.mean(ic_vals))
    std_ic = float(np.std(ic_vals))
    t_stat = float(mean_ic / (std_ic / np.sqrt(len(ic_vals)) + 1e-9))
    return {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "t_stat": t_stat,
        "n_dates": len(ic_vals),
        "ic_series": ic_vals,
    }


def geometric_drift(X_tr, y_tr, X_te, y_te, hi_label: int = 2) -> float:
    hi_tr = X_tr[y_tr == hi_label]
    hi_te = X_te[y_te == hi_label]
    if len(hi_tr) == 0 or len(hi_te) == 0:
        return 0.0
    return float(np.linalg.norm(hi_tr.mean(0) - hi_te.mean(0)))


def var_es(returns: np.ndarray, alpha: float = 0.05) -> dict:
    """Historical VaR and Expected Shortfall (CVaR) at confidence 1-alpha."""
    if len(returns) == 0:
        return {"var": 0.0, "es": 0.0, "alpha": alpha}
    r = np.asarray(returns, dtype=np.float64)
    var = float(np.percentile(r, alpha * 100))
    tail = r[r <= var]
    es = float(np.mean(tail)) if len(tail) else var
    return {"var": var, "es": es, "alpha": alpha}


def strategy_backtest(
    X_tr: np.ndarray,
    fwd_tr: np.ndarray,
    dates_tr: np.ndarray,
    X_te: np.ndarray,
    fwd_te: np.ndarray,
    dates_te: np.ndarray,
    tickers_te: np.ndarray | None = None,
    tickers_tr: np.ndarray | None = None,
    k: int = 10,
    long_threshold: float = 0.002,
    short_threshold: float = -0.002,
    cost_bps: float = 0.0,
    ann_factor: float = 252.0,
) -> dict:
    """
    OOS k-NN retrieval strategy backtest with cross-sectional daily PnL.

    Signal per window: distance-weighted mean neighbor forward return on train set.
    Daily PnL: equal-weight mean of active window returns (long=+1, short=-1).
    """
    if len(X_tr) < k or len(X_te) == 0:
        return _empty_strategy_result()

    nn = NearestNeighbors(metric="cosine")
    nn.fit(X_tr)
    dists, idx = _knn_with_exclusion(
        nn, X_te, k, tickers_tr, dates_tr, tickers_te, dates_te,
    )

    signals = np.zeros(len(X_te))
    pred_rets = np.zeros(len(X_te))
    for i, (d, nbrs) in enumerate(zip(dists, idx)):
        w = similarity_weights(d)
        pred = float(np.dot(w, fwd_tr[nbrs]))
        pred_rets[i] = pred
        if pred >= long_threshold:
            signals[i] = 1.0
        elif pred <= short_threshold:
            signals[i] = -1.0

    cost = cost_bps / 10_000.0
    window_pnl = signals * fwd_te - (signals != 0).astype(np.float64) * cost

    df = pd.DataFrame({
        "date": dates_te,
        "ticker": tickers_te if tickers_te is not None else "all",
        "signal": signals,
        "pred_ret": pred_rets,
        "fwd_ret": fwd_te,
        "pnl": window_pnl,
    })

    daily = df.groupby("date").agg(
        pnl=("pnl", "mean"),
        n_active=("signal", lambda s: int((s != 0).sum())),
        n_long=("signal", lambda s: int((s > 0).sum())),
        n_short=("signal", lambda s: int((s < 0).sum())),
    ).reset_index()

    daily_returns = daily["pnl"].to_numpy(dtype=np.float64)
    cum = np.cumsum(daily_returns)
    peak = np.maximum.accumulate(cum) if len(cum) else np.array([])
    max_dd = float(np.min(cum - peak)) if len(cum) else 0.0

    active = signals != 0
    hit_rate = float(np.mean(window_pnl[active] > 0)) if active.any() else 0.0
    mean_ret = float(np.mean(daily_returns)) if len(daily_returns) else 0.0
    std_ret = float(np.std(daily_returns)) if len(daily_returns) else 1.0
    sharpe = float(mean_ret / (std_ret + 1e-8) * np.sqrt(ann_factor))

    risk = var_es(daily_returns)

    return {
        "sharpe": sharpe,
        "mean_daily_return": mean_ret,
        "std_daily_return": std_ret,
        "max_drawdown": max_dd,
        "hit_rate": hit_rate,
        "total_return": float(cum[-1]) if len(cum) else 0.0,
        "n_days": len(daily),
        "n_windows": len(df),
        "n_active_windows": int(active.sum()),
        "var_95": risk["var"],
        "es_95": risk["es"],
        "daily_pnl": daily_returns.tolist(),
        "ic_on_signals": neighbor_rank_ic(
            X_tr, fwd_tr, X_te, fwd_te, dates_te, k=k,
            tickers_tr=tickers_tr, dates_tr=dates_tr, tickers_te=tickers_te,
        ),
    }


def structural_shape_contraction(
    X_tr: np.ndarray, reg_tr: np.ndarray, X_te: np.ndarray, reg_te: np.ndarray
) -> float:
    """Ratio of test/train within-regime dispersion (1.0 = stable geometry)."""
    dispersions_tr, dispersions_te = [], []
    for regime in ('low', 'mid', 'high'):
        mask_tr = reg_tr == regime
        mask_te = reg_te == regime
        if mask_tr.any() and mask_te.any():
            c_tr = X_tr[mask_tr].mean(axis=0)
            c_te = X_te[mask_te].mean(axis=0)
            dispersions_tr.append(float(np.mean(np.linalg.norm(X_tr[mask_tr] - c_tr, axis=1))))
            dispersions_te.append(float(np.mean(np.linalg.norm(X_te[mask_te] - c_te, axis=1))))
    if not dispersions_tr:
        return 1.0
    return float(np.mean(dispersions_te) / np.mean(dispersions_tr))


def regime_separation_ratio(X: np.ndarray, regimes: np.ndarray) -> float:
    """Between-regime distance / within-regime variance (higher = sharper boundaries)."""
    unique_regimes = [r for r in ('low', 'mid', 'high') if (regimes == r).any()]
    if len(unique_regimes) < 2:
        return 0.0
    centroids = {r: X[regimes == r].mean(axis=0) for r in unique_regimes}
    global_centroid = X.mean(axis=0)
    ss_between = sum(float((regimes == r).sum()) * float(np.linalg.norm(centroids[r] - global_centroid) ** 2) for r in unique_regimes)
    ss_within = sum(float(np.sum(np.linalg.norm(X[regimes == r] - centroids[r], axis=1) ** 2)) for r in unique_regimes)
    return float(ss_between / (ss_within + 1e-9))


def _empty_strategy_result() -> dict:
    return {
        "sharpe": 0.0,
        "mean_daily_return": 0.0,
        "std_daily_return": 0.0,
        "max_drawdown": 0.0,
        "hit_rate": 0.0,
        "total_return": 0.0,
        "n_days": 0,
        "n_windows": 0,
        "n_active_windows": 0,
        "var_95": 0.0,
        "es_95": 0.0,
        "daily_pnl": [],
        "ic_on_signals": {"mean_ic": 0.0, "std_ic": 0.0, "t_stat": 0.0, "n_dates": 0, "ic_series": []},
    }
