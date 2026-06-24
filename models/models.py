from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class MacroConditionedEncoder(nn.Module):
    def __init__(self, temporal_encoder: nn.Module, n_macro: int = 8, encoding_size: int = 64):
        super().__init__()
        self.temporal_encoder = temporal_encoder
        self.macro_head = nn.Sequential(
            nn.Linear(n_macro, encoding_size),
            nn.ReLU(),
            nn.Linear(encoding_size, encoding_size))
        self.fusion = nn.Linear(encoding_size * 2, encoding_size)
    def forward(self, x: torch.Tensor, macro: torch.Tensor | None = None) -> torch.Tensor:
        z = self.temporal_encoder(x)
        if macro is not None:
            m = self.macro_head(macro)
            if len(z.shape) == 3 and len(m.shape) == 2:
                m = m.unsqueeze(1).expand(-1, z.size(1), -1)
            z = self.fusion(torch.cat([z, m], dim=-1))
        return F.normalize(z, dim=-1)
