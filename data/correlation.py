from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def compute_rolling_returns(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).rolling(window, min_periods=2).sum()


def compute_correlation_matrix(returns: pd.DataFrame, method: str = "pearson") -> np.ndarray:
    return returns.corr(method=method).to_numpy(dtype=np.float32)


def compute_rolling_correlation(
    prices: pd.DataFrame,
    return_window: int = 20,
    corr_window: int = 60,
    min_periods: int = 30,
) -> np.ndarray:
    returns = prices.pct_change().iloc[1:]
    n_stocks = prices.shape[1]
    n_dates = len(prices)
    corr_matrices = np.zeros((n_dates, n_stocks, n_stocks), dtype=np.float32)
    for i in range(corr_window, n_dates):
        window_rets = returns.iloc[i - corr_window : i]
        valid = window_rets.dropna(how="all")
        if len(valid) < min_periods:
            corr_matrices[i] = np.eye(n_stocks, dtype=np.float32)
        else:
            corr = valid.corr().to_numpy(dtype=np.float32)
            corr = np.nan_to_num(corr, nan=0.0)
            corr_matrices[i] = corr
    corr_matrices[:corr_window] = np.eye(n_stocks, dtype=np.float32)
    return corr_matrices


def correlation_to_adjacency(
    corr: np.ndarray | torch.Tensor,
    threshold: float = 0.3,
    top_k: int | None = 20,
    self_loop: bool = True,
    symmetric: bool = True,
) -> np.ndarray | torch.Tensor:
    is_torch = isinstance(corr, torch.Tensor)
    xp = torch if is_torch else np

    adj = xp.abs(corr)
    if threshold > 0:
        adj = adj * (adj > threshold).astype(xp.float32) if not is_torch else adj * (adj > threshold).float()
    if top_k is not None:
        k = min(top_k, adj.shape[-1])
        idx = xp.argsort(adj, axis=-1)[:, :-k] if is_torch else xp.argpartition(adj, -k, axis=-1)[:, :-k]
        mask = xp.ones_like(adj, dtype=xp.float32 if not is_torch else adj.dtype)
        if is_torch:
            mask.scatter_(1, idx, 0.0)
        else:
            xp.put_along_axis(mask, idx, 0.0, axis=1)
        adj = adj * mask
    if self_loop:
        eye = xp.eye(adj.shape[0], dtype=adj.dtype)
        adj = adj + eye if not is_torch else adj + eye

    if symmetric:
        adj = (adj + adj.T) / 2 if not is_torch else (adj + adj.T) / 2
    d_inv_sqrt = xp.diag(adj.sum(axis=-1).clip(1e-8) ** (-0.5))
    if is_torch:
        d_inv_sqrt = d_inv_sqrt.to(adj.dtype)
        return d_inv_sqrt @ adj @ d_inv_sqrt
    return d_inv_sqrt @ adj @ d_inv_sqrt


def build_sparse_adjacency(
    ticker_prices: dict[str, np.ndarray],
    return_window: int = 20,
    corr_threshold: float = 0.3,
    top_k: int | None = 20,
    dates_idx: int = -1,
) -> tuple[np.ndarray, list[str]]:
    tickers = sorted(ticker_prices.keys())
    prices = np.column_stack([ticker_prices[t] for t in tickers])
    df = pd.DataFrame(prices, columns=tickers)
    returns = df.pct_change().iloc[1:]
    window_rets = returns.iloc[max(0, dates_idx - return_window) : dates_idx]
    valid = window_rets.dropna(how="all")
    if len(valid) < 2:
        adj = np.eye(len(tickers), dtype=np.float32)
    else:
        corr = valid.corr().to_numpy(dtype=np.float32)
        corr = np.nan_to_num(corr, nan=0.0)
        adj = correlation_to_adjacency(corr, threshold=corr_threshold, top_k=top_k)
    return adj, tickers
