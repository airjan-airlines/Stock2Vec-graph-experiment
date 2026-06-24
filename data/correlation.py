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
            np.fill_diagonal(corr, 1.0)
            corr = (corr + corr.T) / 2
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
    n_stocks = corr.shape[-1]
    adj = xp.abs(corr)
    
    if threshold > 0:
        adj = adj * (adj > threshold).astype(xp.float32) if not is_torch else adj * (adj > threshold).float()
        
    if top_k is not None:
        k = min(top_k, n_stocks)
        idx = xp.argsort(adj, axis=-1)[..., :-k] if is_torch else xp.argpartition(adj, -k, axis=-1)[..., :-k]
        mask = xp.ones_like(adj, dtype=xp.float32 if not is_torch else adj.dtype)
        
        if is_torch:
            mask.scatter_(-1, idx, 0.0)
        else:
            xp.put_along_axis(mask, idx, 0.0, axis=-1)
        adj = adj * mask
        
    if self_loop:
        if len(adj.shape) == 3:
            eye = xp.eye(n_stocks, dtype=adj.dtype)
            if not is_torch:
                eye = np.expand_dims(eye, axis=0)
            else:
                eye = eye.unsqueeze(0)
            adj = adj + eye
        else:
            eye = xp.eye(n_stocks, dtype=adj.dtype)
            adj = adj + eye

    if symmetric:
        if len(adj.shape) == 3:
            adj = (adj + xp.swapaxes(adj, -1, -2)) / 2 if not is_torch else (adj + adj.transpose(-1, -2)) / 2
        else:
            adj = (adj + adj.T) / 2 if not is_torch else (adj + adj.T) / 2
    deg_sum = adj.sum(axis=-1)
    if is_torch:
        d_inv_sqrt = torch.diag_embed(deg_sum.clamp(min=1e-8) ** (-0.5))
        return d_inv_sqrt @ adj @ d_inv_sqrt
    else:
        if len(adj.shape) == 3:
            t_dim = adj.shape[0]
            out = np.zeros_like(adj)
            for t in range(t_dim):
                d_inv = np.diag(np.clip(deg_sum[t], 1e-8, None) ** (-0.5))
                out[t] = d_inv @ adj[t] @ d_inv
            return out
        else:
            d_inv_sqrt = np.diag(np.clip(deg_sum, 1e-8, None) ** (-0.5))
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
    start_idx = max(0, dates_idx - return_window)
    window_rets = returns.iloc[start_idx:dates_idx]
    valid = window_rets.dropna(how="all")
    if len(valid) < 2:
        adj = np.eye(len(tickers), dtype=np.float32)
    else:
        corr = valid.corr().to_numpy(dtype=np.float32)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        corr = (corr + corr.T) / 2
        adj = correlation_to_adjacency(corr, threshold=corr_threshold, top_k=top_k)
    return adj, tickers
