from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.temporal import build_temporal_encoder
from models.gnn import CorrelationGNN, correlation_to_adj


class Stock2VecEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        encoding_size: int = 64,
        temporal_type: Literal["tcn", "transformer", "cnn"] = "tcn",
        use_graph: bool = True,
        gnn_hidden_dim: int = 128,
        gnn_layers: int = 3,
        gnn_layer_type: Literal["gcn", "gat"] = "gcn",
        gat_heads: int = 4,
        adj_threshold: float = 0.3,
        adj_top_k: int = 20,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.use_graph = use_graph
        self.adj_threshold = adj_threshold
        self.adj_top_k = adj_top_k

        temporal_kwargs = {"dropout": dropout}
        if temporal_type == "transformer":
            temporal_kwargs["d_model"] = encoding_size
            temporal_kwargs["nhead"] = 4 if encoding_size <= 64 else 8

        self.temporal_encoder = build_temporal_encoder(
            temporal_type, in_channels, encoding_size, **temporal_kwargs
        )

        if use_graph:
            self.gnn = CorrelationGNN(
                node_dim=encoding_size,
                hidden_dim=gnn_hidden_dim,
                encoding_size=encoding_size,
                num_layers=gnn_layers,
                layer_type=gnn_layer_type,
                gat_heads=gat_heads,
                dropout=dropout,
            )

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_graph and adj is not None:
            node_embs = self.temporal_encoder(x)
            return self.gnn(node_embs, adj)
        return F.normalize(self.temporal_encoder(x), dim=-1)

    @torch.no_grad()
    def encode_batch(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        self.eval()
        return self.forward(x, adj)


class MacroStock2VecEncoder(nn.Module):
    def __init__(self, base_encoder: Stock2VecEncoder, n_macro: int = 8, encoding_size: int = 64):
        super().__init__()
        self.base_encoder = base_encoder
        self.use_graph = base_encoder.use_graph
        self.macro_head = nn.Sequential(
            nn.Linear(n_macro, encoding_size),
            nn.ReLU(),
            nn.Linear(encoding_size, encoding_size),
        )
        self.fusion = nn.Linear(encoding_size * 2, encoding_size)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None,
                macro: torch.Tensor | None = None) -> torch.Tensor:
        z = self.base_encoder.temporal_encoder(x)
        if self.base_encoder.use_graph and adj is not None:
            z = self.base_encoder.gnn(z, adj)
        if macro is not None:
            m = self.macro_head(macro)
            z = self.fusion(torch.cat([z, m], dim=-1))
        return F.normalize(z, dim=-1)


def build_encoder(
    in_channels: int,
    encoding_size: int = 64,
    temporal_type: Literal["tcn", "transformer", "cnn"] = "tcn",
    use_graph: bool = True,
    n_macro: int = 0,
    **kwargs,
) -> nn.Module:
    encoder = Stock2VecEncoder(
        in_channels=in_channels,
        encoding_size=encoding_size,
        temporal_type=temporal_type,
        use_graph=use_graph,
        **kwargs,
    )
    if n_macro > 0:
        return MacroStock2VecEncoder(encoder, n_macro=n_macro, encoding_size=encoding_size)
    return encoder
