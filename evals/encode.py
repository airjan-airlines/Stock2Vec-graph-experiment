"""
evals/encode.py — Shared encoder loading and embedding generation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from data.prep_features import FEATURE_COLS as INTRADAY_FEATURE_COLS
from data.prep_features_daily import FEATURE_COLS as DAILY_FEATURE_COLS
from models.models import (
    CNNStockEncoder,
    MacroConditionedEncoder,
    TCNEncoder,
    TemporalTransformerEncoder,
)
from models.utils import NPZBundle, latest_npz, load_npz_bundle

WINDOW_INTRADAY = 65
WINDOW_DAILY = 20
STEP_INTRADAY = 13
STEP_DAILY = 1


class _AllWindowsDataset(Dataset):
    """Lazy dataset of all windows across all tickers."""

    def __init__(self, bundle: NPZBundle, window: int, step: int):
        self.bundle = bundle
        self.window = window
        self.samples: list[tuple[int, int, int]] = []  # (ticker_idx, center, start)
        for i in range(len(bundle.tickers)):
            L = int(bundle.lengths[i])
            if L < window:
                continue
            starts = range(0, max(L - window + 1, 0), step)
            for s in starts:
                center = min(s + window // 2, L - 1)
                self.samples.append((i, center, s))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        ticker_idx, center, start = self.samples[idx]
        w = self.bundle.x[ticker_idx, :, start:start + self.window].float()
        return w, ticker_idx, center


def resolve_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_ckpt_path(ckpt: str | Path | None, daily: bool) -> Path:
    if ckpt:
        p = Path(ckpt)
        return p / "checkpoint_0.pth.tar" if p.is_dir() else p
    default = "stock2vec_daily_macro" if daily else "stock2vec"
    root = Path(__file__).resolve().parent.parent
    return root / "ckpt" / default / "checkpoint_0.pth.tar"


def load_encoder(ckpt_path: Path, in_channels: int, n_macro: int, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["encoder_state_dict"]
    cleaned = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    use_macro = any(k.startswith("macro_head") or k.startswith("fusion") for k in cleaned)

    if use_macro:
        temporal_keys = {k for k in cleaned if not k.startswith("macro_head") and not k.startswith("fusion")}
        if "input_proj.weight" in temporal_keys:
            d_model = cleaned["input_proj.weight"].shape[0]
            nhead = 4 if d_model <= 64 else 8
            n_layers = (
                max(int(k.split(".")[2]) for k in cleaned if k.startswith("transformer.layers.")) + 1
                if any(k.startswith("transformer.layers.") for k in cleaned)
                else 2
            )
            temporal = TemporalTransformerEncoder(
                in_channels=in_channels, encoding_size=64,
                d_model=d_model, nhead=nhead, num_layers=n_layers,
            )
        elif any(k.startswith("tcn.") for k in cleaned):
            num_channels = [
                v.shape[0] for k, v in cleaned.items()
                if k.startswith("tcn.") and k.endswith(".conv1.weight_v")
            ] or [64, 64, 128]
            temporal = TCNEncoder(in_channels=in_channels, encoding_size=64, num_channels=num_channels)
        else:
            temporal = CNNStockEncoder(encoding_size=64, input_channels=in_channels)
        enc = MacroConditionedEncoder(temporal, n_macro=n_macro, encoding_size=64)
    elif "input_proj.weight" in cleaned:
        d_model = cleaned["input_proj.weight"].shape[0]
        nhead = 4 if d_model <= 64 else 8
        n_layers = max(int(k.split(".")[2]) for k in cleaned if k.startswith("transformer.layers.")) + 1
        enc = TemporalTransformerEncoder(
            in_channels=in_channels, encoding_size=64,
            d_model=d_model, nhead=nhead, num_layers=n_layers,
        )
    elif any(k.startswith("tcn.") for k in cleaned):
        num_channels = [
            v.shape[0] for k, v in cleaned.items()
            if k.startswith("tcn.") and k.endswith(".conv1.weight_v")
        ] or [64, 64, 128, 128]
        enc = TCNEncoder(in_channels=in_channels, encoding_size=64, num_channels=num_channels)
    else:
        enc = CNNStockEncoder(encoding_size=64, input_channels=in_channels)

    enc.load_state_dict(cleaned, strict=False)
    enc.to(device).eval()
    return enc, use_macro, ckpt


@torch.no_grad()
def encode_series(
    enc,
    series: np.ndarray,
    length: int,
    macro: np.ndarray | None,
    window: int,
    step: int,
    use_macro: bool,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[np.ndarray, list[int]]:
    starts = list(range(0, max(length - window + 1, 0), step))
    if not starts:
        return np.empty((0, 64), dtype=np.float32), []

    windows = np.stack([series[:, s : s + window] for s in starts])
    loader = DataLoader(TensorDataset(torch.from_numpy(windows)), batch_size=batch_size, shuffle=False)
    parts = []
    offset = 0
    for (batch,) in loader:
        batch = batch.to(device)
        if use_macro and macro is not None:
            centers = [
                min(s + window // 2, macro.shape[0] - 1)
                for s in starts[offset : offset + len(batch)]
            ]
            m = torch.from_numpy(macro[centers]).to(device)
            z = enc(batch, m)
        else:
            z = enc(batch)
        parts.append(F.normalize(z, dim=-1).cpu().numpy())
        offset += len(batch)
    return np.concatenate(parts, axis=0), starts


def global_vol_regime_labels(rvols: np.ndarray) -> np.ndarray:
    """Assign low/mid/high (0/1/2) by global tercile of realized vol across all windows."""
    q33, q67 = np.nanpercentile(rvols, [33.33, 66.67])
    return np.where(rvols <= q33, 0, np.where(rvols <= q67, 1, 2))


def _forward_return_step(is_daily: bool) -> int:
    return 1 if is_daily else 13


def build_embedding_table(
    bundle: NPZBundle,
    enc,
    use_macro: bool,
    device: torch.device,
    bars_dir: Path | None = None,
    batch_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Encode all tickers and return aligned arrays:
      embeddings, vol_regimes, fwd_rets, dates, tickers, window_indices
    """
    window = WINDOW_DAILY if bundle.is_daily else WINDOW_INTRADAY
    step = STEP_DAILY if bundle.is_daily else STEP_INTRADAY
    rvol_idx = (
        DAILY_FEATURE_COLS.index("realized_vol")
        if bundle.is_daily
        else INTRADAY_FEATURE_COLS.index("realized_vol")
    )
    macro_np = bundle.macro.numpy() if bundle.macro is not None else None
    fwd_step = _forward_return_step(bundle.is_daily)

    if bundle.fwd_ret_1d is None and bars_dir is None:
        bars_kind = "daily" if bundle.is_daily else "bars"
        tag = bundle.path.stem.split("_")[-1]
        raise FileNotFoundError(
            f"Cannot compute forward returns: NPZ has no fwd_ret_1d and raw {bars_kind} "
            f"directory is missing. Run the data pull step or point --bars-dir at "
            f"data/raw/{bars_kind}/{tag}/."
        )

    # Build one large dataset of all windows across all tickers
    all_ds = _AllWindowsDataset(bundle, window, step)
    if len(all_ds) == 0:
        raise ValueError("No valid windows found in NPZ bundle.")

    loader = DataLoader(all_ds, batch_size=batch_size, shuffle=False)

    all_embs: list[torch.Tensor] = []
    all_tidxs: list[torch.Tensor] = []
    all_centers: list[torch.Tensor] = []
    if str(device) == "cuda":
        torch.cuda.empty_cache()

    with torch.no_grad():
        for batch in loader:
            windows, ticker_idxs, batch_centers = batch
            windows = windows.to(device)
            if use_macro and macro_np is not None:
                m = torch.from_numpy(macro_np[batch_centers.numpy()]).to(device)
                z = enc(windows, m)
            else:
                z = enc(windows)
            all_embs.append(F.normalize(z, dim=-1).cpu())
            all_tidxs.append(ticker_idxs)
            all_centers.append(batch_centers)

    embeddings = torch.cat(all_embs, dim=0).numpy()
    ticker_idx_arr = torch.cat(all_tidxs).numpy()
    center_arr = torch.cat(all_centers).numpy()

    rows: list[dict] = []

    for i, ticker in enumerate(bundle.tickers):
        mask = ticker_idx_arr == i
        if not mask.any():
            continue
        order = np.argsort(center_arr[mask])
        tick_embs = embeddings[mask][order]
        tick_centers = center_arr[mask][order].tolist()

        rvols = bundle.x[i, rvol_idx, tick_centers].numpy().astype(np.float64)

        if bundle.fwd_ret_1d is not None:
            fwd = bundle.fwd_ret_1d[i][tick_centers]
            if bundle.dates is not None:
                dates = [str(d)[:10] for d in bundle.dates[tick_centers]]
            else:
                raise ValueError(
                    "NPZ provides fwd_ret_1d but no dates array — cannot group cross-sectional IC."
                )
        elif bars_dir and (bars_dir / f"{ticker}.parquet").exists():
            import pandas as pd

            L = int(bundle.lengths[i])
            df = pd.read_parquet(bars_dir / f"{ticker}.parquet").sort_values("timestamp").iloc[:L]
            closes = df["close"].to_numpy(dtype=np.float64)
            date_arr = pd.to_datetime(df["timestamp"], utc=True).dt.strftime("%Y-%m-%d").to_numpy()
            fwd = np.array([
                (closes[min(c + fwd_step, L - 1)] - closes[c]) / (closes[c] + 1e-12)
                for c in tick_centers
            ])
            dates = [date_arr[c] for c in tick_centers]
        else:
            continue

        for j in range(len(tick_centers)):
            rows.append({
                "emb": tick_embs[j],
                "rvol": rvols[j],
                "fwd": float(fwd[j]),
                "date": dates[j],
                "ticker": str(ticker),
            })

    if not rows:
        bars_kind = "daily" if bundle.is_daily else "bars"
        tag = bundle.path.stem.split("_")[-1]
        raise ValueError(
            "No embeddings with valid forward returns. "
            f"Ensure per-ticker parquet files exist under data/raw/{bars_kind}/{tag}/ "
            "or use a daily NPZ that includes fwd_ret_1d and dates."
        )

    all_rvol = np.array([r["rvol"] for r in rows], dtype=np.float64)
    regimes = global_vol_regime_labels(all_rvol)

    return (
        np.stack([r["emb"] for r in rows]),
        regimes,
        np.array([r["fwd"] for r in rows], dtype=np.float64),
        np.array([r["date"] for r in rows]),
        np.array([r["ticker"] for r in rows]),
        np.arange(len(rows)),
    )


def load_bundle_and_ckpt(daily: bool, ckpt: str | Path | None, device: torch.device):
    bundle = load_npz_bundle(latest_npz(daily=daily))
    ckpt_path = resolve_ckpt_path(ckpt, daily)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    enc, use_macro, ckpt_meta = load_encoder(
        ckpt_path, bundle.x.shape[1], bundle.macro.shape[-1], device
    )
    return bundle, enc, use_macro, ckpt_path, ckpt_meta
