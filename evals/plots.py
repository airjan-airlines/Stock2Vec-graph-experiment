"""
evals/plots.py — UMAP projection of latest Stock2Vec embeddings per ticker.

Uses the shared encoder loader from evals.encode (supports transformer/TCN/CNN/macro).

Usage
-----
  uv run python evals/plots.py
  uv run python evals/plots.py --ckpt ckpt/stock2vec_statiocl
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.encode import WINDOW_INTRADAY, load_encoder, resolve_device
from models.utils import latest_npz, load_npz_bundle

WINDOW_SIZE = WINDOW_INTRADAY
PLOT_DPI = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UMAP plot of Stock2Vec embeddings")
    parser.add_argument("--ckpt", default=None, help="Checkpoint path or ckpt/ subfolder")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    bundle = load_npz_bundle(latest_npz(daily=False))
    ckpt_path = Path(args.ckpt) if args.ckpt else ROOT / "ckpt" / "stock2vec" / "checkpoint_0.pth.tar"
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "checkpoint_0.pth.tar"
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    in_channels = bundle.x.shape[1]
    n_macro = bundle.macro.shape[-1] if bundle.macro is not None else 8
    enc, use_macro, _ = load_encoder(ckpt_path, in_channels, n_macro, device)
    print(f"Loaded {len(bundle.tickers)} tickers from {bundle.path.name}  macro={use_macro}")

    macro_np = bundle.macro.numpy() if use_macro and bundle.macro is not None else None
    windows, valid_tickers, macro_centers = [], [], []

    for i, ticker in enumerate(bundle.tickers):
        L = int(bundle.lengths[i])
        if L < WINDOW_SIZE:
            continue
        series = bundle.x[i].numpy()
        start = L - WINDOW_SIZE
        windows.append(series[:, start:L])
        valid_tickers.append(str(ticker))
        macro_centers.append(min(start + WINDOW_SIZE // 2, macro_np.shape[0] - 1) if macro_np is not None else 0)

    if not windows:
        print("No tickers with sufficient history for UMAP.")
        sys.exit(1)

    windows_t = torch.from_numpy(np.stack(windows)).to(device)
    with torch.no_grad():
        if use_macro and macro_np is not None:
            m = torch.from_numpy(macro_np[macro_centers]).to(device)
            embeddings = enc(windows_t, m)
        else:
            embeddings = enc(windows_t)
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu().numpy()

    try:
        import umap
    except ImportError:
        print("umap-learn not installed — run: uv add umap-learn")
        sys.exit(1)

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)

    random.seed(42)
    plt.figure(figsize=(15, 12), dpi=PLOT_DPI)
    style = "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    plt.style.use(style)

    core_symbols = ["AAPL", "NVDA", "MSFT", "AMZN", "TSLA", "META", "GOOGL", "JPM", "SPY", "QQQ"]
    num_to_highlight = min(100, len(valid_tickers))
    selected = [t for t in core_symbols if t in valid_tickers]
    others = [t for t in valid_tickers if t not in selected]
    random.shuffle(others)
    selected.extend(others[: num_to_highlight - len(selected)])

    try:
        cmap = plt.colormaps.get_cmap("turbo")
    except AttributeError:
        cmap = plt.cm.get_cmap("turbo")
    colors = [cmap(v) for v in np.linspace(0, 0.95, len(selected))]

    x_coords, y_coords = embeddings_2d[:, 0], embeddings_2d[:, 1]
    selected_set = set(selected)
    bg_mask = np.array([t not in selected_set for t in valid_tickers])
    plt.scatter(x_coords[bg_mask], y_coords[bg_mask], c="#cbd5e1", alpha=0.25, s=6, edgecolors="none")

    for idx_select, ticker in enumerate(selected):
        idx = valid_tickers.index(ticker)
        color = colors[idx_select]
        is_core = ticker in core_symbols
        plt.scatter(
            x_coords[idx], y_coords[idx], color=color,
            s=130 if is_core else 65, zorder=10 if is_core else 5,
            edgecolors="white", linewidths=1.0,
        )
        plt.annotate(
            ticker, (x_coords[idx], y_coords[idx]),
            textcoords="offset points", xytext=(0, 6 if is_core else 4), ha="center",
            fontsize=8 if is_core else 6, fontweight="bold" if is_core else "normal",
            color="#0f172a", zorder=20 if is_core else 15,
            bbox=dict(
                boxstyle="round,pad=0.12", fc="white",
                alpha=0.9 if is_core else 0.75,
                ec=color if is_core else "#cbd5e1",
                lw=1.5 if is_core else 0.5,
            ),
        )

    run_tag = ckpt_path.parent.name
    plt.title(f"UMAP — {run_tag}\n(latest {WINDOW_SIZE}-bar window per ticker)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")

    plots_dir = ROOT / "plots"
    plots_dir.mkdir(exist_ok=True)
    out_path = plots_dir / f"umap_tickers_{run_tag}.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=PLOT_DPI)
    plt.close()
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
