from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


@dataclass
class NPZBundle:
    x: torch.Tensor
    lengths: torch.Tensor
    tickers: np.ndarray
    dates: np.ndarray | None
    macro: torch.Tensor | None
    fwd_ret_1d: torch.Tensor | None
    fwd_ret_5d: torch.Tensor | None
    path: Path
    is_daily: bool


def latest_npz(daily: bool = False) -> Path:
    pattern = "TNC_features_daily_*.npz" if daily else "TNC_features_*.npz"
    candidates = sorted(PROCESSED_DIR.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No NPZ files matching '{pattern}' in {PROCESSED_DIR}")
    return candidates[-1]


def load_npz_bundle(path: str | Path) -> NPZBundle:
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    macro = torch.from_numpy(data["macro"]).float() if "macro" in data and data["macro"].size > 0 else None
    fwd_1d = torch.from_numpy(data["fwd_ret_1d"]).float() if "fwd_ret_1d" in data else None
    fwd_5d = torch.from_numpy(data["fwd_ret_5d"]).float() if "fwd_ret_5d" in data else None
    dates = data["dates"] if "dates" in data else None
    return NPZBundle(
        x=torch.from_numpy(data["x"]).float(),
        lengths=torch.from_numpy(data["lengths"]).long(),
        tickers=data["tickers"],
        dates=dates,
        macro=macro,
        fwd_ret_1d=fwd_1d,
        fwd_ret_5d=fwd_5d,
        path=path,
        is_daily="daily" in path.stem,
    )
