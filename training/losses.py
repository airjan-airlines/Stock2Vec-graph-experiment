from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TNCLoss(nn.Module):
    def __init__(self, temperature: float = 0.5, lambda_1: float = 1.0, lambda_2: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def forward(self, z_anchor: torch.Tensor, z_positive: torch.Tensor,
                z_negative: torch.Tensor) -> torch.Tensor:
        z_anc = F.normalize(z_anchor, dim=-1)
        z_pos = F.normalize(z_positive, dim=-1)
        z_neg = F.normalize(z_negative, dim=-1)

        pos_sim = (z_anc * z_pos).sum(dim=-1) / self.temperature
        neg_sim = (z_anc * z_neg).sum(dim=-1) / self.temperature

        all_sim = torch.cat([z_anc @ z_pos.T, z_anc @ z_neg.T], dim=1) / self.temperature
        pos_logits = torch.arange(z_anc.size(0), device=z_anc.device)
        nce_loss = F.cross_entropy(all_sim, pos_logits)

        pos_exp = torch.exp(pos_sim)
        neg_exp = torch.exp(neg_sim)
        disc_loss = -torch.log(pos_exp / (pos_exp + neg_exp)).mean()

        return self.lambda_1 * nce_loss + self.lambda_2 * disc_loss


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        z = F.normalize(z, dim=-1)
        sim = z @ z.T / self.temperature
        n = z.size(0)
        if labels is None:
            pos_mask = torch.eye(n, dtype=torch.bool, device=z.device)
        else:
            pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
            pos_mask.fill_diagonal_(0)
            if not pos_mask.any():
                pos_mask = torch.eye(n, dtype=torch.bool, device=z.device)
        neg_mask = ~pos_mask
        pos_sim = sim[pos_mask].view(n, -1)
        neg_sim = sim[neg_mask].view(n, -1)
        logits = torch.cat([pos_sim.mean(dim=1, keepdim=True), neg_sim], dim=1)
        labels = torch.zeros(n, dtype=torch.long, device=z.device)
        return F.cross_entropy(logits / self.temperature, labels)


class ForwardReturnLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_returns: torch.Tensor, target_returns: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred_returns, target_returns)


class CombinedLoss(nn.Module):
    def __init__(self, contrastive_loss: nn.Module, supervised_loss: nn.Module,
                 lambda_contrastive: float = 1.0, lambda_supervised: float = 0.1):
        super().__init__()
        self.contrastive_loss = contrastive_loss
        self.supervised_loss = supervised_loss
        self.lambda_c = lambda_contrastive
        self.lambda_s = lambda_supervised

    def forward(self, z: torch.Tensor, pred_returns: torch.Tensor | None = None,
                target_returns: torch.Tensor | None = None,
                z_anchor: torch.Tensor | None = None,
                z_positive: torch.Tensor | None = None,
                z_negative: torch.Tensor | None = None) -> torch.Tensor:
        total = 0.0
        if all(x is not None for x in (z_anchor, z_positive, z_negative)):
            total += self.lambda_c * self.contrastive_loss(z_anchor, z_positive, z_negative)
        if pred_returns is not None and target_returns is not None:
            total += self.lambda_s * self.supervised_loss(pred_returns, target_returns)
        return total
