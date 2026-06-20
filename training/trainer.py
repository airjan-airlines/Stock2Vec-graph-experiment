from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.utils import NPZBundle

log = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        device: torch.device | str = "cpu",
        ckpt_dir: str | Path = "ckpt",
        run_name: str = "stock2vec",
        grad_clip: float = 1.0,
        log_interval: int = 10,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device)
        self.ckpt_dir = Path(ckpt_dir) / run_name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.grad_clip = grad_clip
        self.log_interval = log_interval

        self.optimizer = optimizer or AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = scheduler

        self.model.to(self.device)
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self.best_val_loss = float("inf")
        self.start_epoch = 0

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> Path:
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "best_val_loss": self.best_val_loss,
        }
        path = self.ckpt_dir / ("checkpoint_best.pth.tar" if is_best else f"checkpoint_{epoch}.pth.tar")
        torch.save(ckpt, path)
        log.info(f"Saved checkpoint → {path}")
        return path

    def load_checkpoint(self, path: str | Path) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.history = ckpt.get("history", self.history)
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.start_epoch = ckpt.get("epoch", 0) + 1
        log.info(f"Loaded checkpoint from {path} (epoch {ckpt.get('epoch', 0)})")
        return ckpt.get("epoch", 0)

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.perf_counter()

        for batch_idx, batch in enumerate(self.train_loader):
            batch = self._to_device(batch)
            loss = self._compute_loss(batch)

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if batch_idx % self.log_interval == 0:
                log.info(f"  [{batch_idx}/{len(self.train_loader)}] loss={loss.item():.4f}")

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self) -> float:
        if self.val_loader is None:
            return float("inf")
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        for batch in self.val_loader:
            batch = self._to_device(batch)
            loss = self._compute_loss(batch)
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    def _to_device(self, batch: Any) -> Any:
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, (list, tuple)):
            return tuple(self._to_device(b) for b in batch)
        elif isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        return batch

    def _compute_loss(self, batch: Any) -> torch.Tensor:
        if isinstance(batch, tuple) and len(batch) == 3:
            anc, pos, neg = batch
            z_anc, z_pos, z_neg = self.model(anc), self.model(pos), self.model(neg)
            return self.loss_fn(z_anchor=z_anc, z_positive=z_pos, z_negative=z_neg)
        elif isinstance(batch, dict):
            x = batch["x"]
            z = self.model(x)
            if "target" in batch and hasattr(self.loss_fn, "forward") and hasattr(self.loss_fn, "lambda_s"):
                return self.loss_fn(z_anchor=z, z_positive=z, z_negative=z,
                                     pred_returns=z, target_returns=batch["target"])
            return self.loss_fn(z, batch.get("target"))
        else:
            z = self.model(batch)
            return self.loss_fn(z)

    def fit(self, epochs: int = 100) -> dict[str, list[float]]:
        for epoch in range(self.start_epoch, self.start_epoch + epochs):
            t0 = time.perf_counter()
            train_loss = self.train_epoch()
            val_loss = self.validate()
            elapsed = time.perf_counter() - t0

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            log.info(f"Epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  ({elapsed:.1f}s)")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)

            if self.scheduler:
                self.scheduler.step()

            if epoch % 10 == 0:
                self.save_checkpoint(epoch)

        self.save_history()
        return self.history

    def save_history(self) -> None:
        path = self.ckpt_dir / "history.json"
        path.write_text(json.dumps(self.history, indent=2))
        log.info(f"Saved training history → {path}")


def train_tnc(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    ckpt_dir: str = "ckpt",
    run_name: str = "stock2vec_tnc",
    **trainer_kwargs,
) -> Trainer:
    loss_fn = _build_tnc_loss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    trainer = Trainer(
        model=model, loss_fn=loss_fn,
        train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, device=device,
        ckpt_dir=ckpt_dir, run_name=run_name,
        **trainer_kwargs,
    )
    trainer.fit(epochs=epochs)
    return trainer


def _build_tnc_loss() -> nn.Module:
    from training.losses import TNCLoss
    return TNCLoss()
