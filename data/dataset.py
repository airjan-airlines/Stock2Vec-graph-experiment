from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from models.utils import NPZBundle, load_npz_bundle, latest_npz

FEATURE_COLS_INTRADAY = [
    "open_scaled", "high_scaled", "low_scaled", "close_scaled",
    "ret", "rel_volume", "log_hl_range", "log_oc_gap",
    "log_vwap_dev", "ret_vs_spy", "realized_vol", "spy_ret",
    "spy_realized_vol", "sin_time", "cos_time",
]

FEATURE_COLS_DAILY = [
    "open_scaled", "high_scaled", "low_scaled", "close_scaled",
    "ret", "rel_volume", "log_hl_range", "log_oc_gap",
    "ret_vs_spy", "realized_vol", "spy_ret", "spy_realized_vol",
]

WINDOW_INTRADAY = 65
WINDOW_DAILY = 20


class TNCWindowDataset(Dataset):
    def __init__(self, bundle: NPZBundle, window: int | None = None,
                 step: int = 1, positive_w: int = 5, negative_w: int = 50):
        self.bundle = bundle
        self.window = window or (WINDOW_DAILY if bundle.is_daily else WINDOW_INTRADAY)
        self.step = step
        self.positive_w = positive_w
        self.negative_w = negative_w
        self.samples: list[tuple[int, int, int]] = []

        for i, L in enumerate(bundle.lengths):
            L = int(L)
            if L < self.window + self.negative_w:
                continue
            max_start = L - self.window
            for t in range(0, max_start, self.step):
                pos_low = max(0, t - self.positive_w)
                pos_high = min(max_start, t + self.positive_w)
                neg_low = max(0, t - self.negative_w)
                neg_high = min(max_start, t + self.negative_w)
                positive_pool = [p for p in range(pos_low, pos_high + 1) if p != t]
                negative_pool = [n for n in range(neg_low, neg_high + 1) if n != t]
                if positive_pool and negative_pool:
                    self.samples.append((int(i), int(t), 0))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i, t, _ = self.samples[idx]
        L = int(self.bundle.lengths[i])
        max_start = L - self.window
        pos_low = max(0, t - self.positive_w)
        pos_high = min(max_start, t + self.positive_w)
        neg_low = max(0, t - self.negative_w)
        neg_high = min(max_start, t + self.negative_w)

        rng = np.random.default_rng()
        t_pos = int(rng.integers(pos_low, pos_high))
        while t_pos == t and pos_high - pos_low > 1:
            t_pos = int(rng.integers(pos_low, pos_high))
        t_neg = int(rng.integers(neg_low, neg_high))
        while t_neg == t and neg_high - neg_low > 1:
            t_neg = int(rng.integers(neg_low, neg_high))

        x_anchor = self.bundle.x[i, :, t:t + self.window]
        x_pos = self.bundle.x[i, :, t_pos:t_pos + self.window]
        x_neg = self.bundle.x[i, :, t_neg:t_neg + self.window]
        return x_anchor.clone(), x_pos.clone(), x_neg.clone()


class GraphWindowDataset(Dataset):
    def __init__(self, bundle: NPZBundle, adj_matrices: np.ndarray | torch.Tensor | None = None,
                 window: int | None = None, step: int = 1, pred_horizon: int = 1):
        self.bundle = bundle
        self.window = window or (WINDOW_DAILY if bundle.is_daily else WINDOW_INTRADAY)
        self.step = step
        self.pred_horizon = pred_horizon
        self.adj = adj_matrices
        self.samples: list[tuple[int, int]] = []

        for i, L in enumerate(bundle.lengths):
            L = int(L)
            if L < self.window + pred_horizon:
                continue
            max_start = L - self.window - pred_horizon
            for t in range(0, max_start, self.step):
                self.samples.append((int(i), int(t)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        i, t = self.samples[idx]
        x = self.bundle.x[i, :, t:t + self.window].clone()
        target = self.bundle.x[i, :, t + self.window + self.pred_horizon - 1].clone() \
            if self.pred_horizon > 0 else torch.zeros(1)
        return {"x": x, "ticker_idx": i, "time_idx": t, "target": target}


def collate_tnc(batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    anc, pos, neg = zip(*batch)
    return torch.stack(anc), torch.stack(pos), torch.stack(neg)


def create_tnc_dataloader(bundle: NPZBundle, batch_size: int = 128, shuffle: bool = True,
                          window: int | None = None, step: int = 1,
                          positive_w: int = 5, negative_w: int = 50) -> DataLoader:
    ds = TNCWindowDataset(bundle, window=window, step=step,
                           positive_w=positive_w, negative_w=negative_w)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_tnc)


def load_dataset(npz_path: str | Path | None = None, daily: bool = False) -> NPZBundle:
    if npz_path:
        return load_npz_bundle(npz_path)
    return load_npz_bundle(latest_npz(daily=daily))
