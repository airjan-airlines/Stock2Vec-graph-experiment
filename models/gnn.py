from __future__ import annotations

from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = torch.matmul(x, self.weight)
        out = torch.matmul(adj, support)
        if len(out.shape) == 3:
            out = out + self.bias.unsqueeze(0).unsqueeze(0)
        else:
            out = out + self.bias
        return self.dropout(F.relu(out))


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.W = nn.Parameter(torch.empty(in_dim, out_dim * heads))
        self.a = nn.Parameter(torch.empty(2 * out_dim, heads))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        is_batched = len(x.shape) == 3
        if is_batched:
            b, n, _ = x.shape
        else:
            b, n = 1, x.shape[0]
            x = x.unsqueeze(0)
            adj = adj.unsqueeze(0)
        h = torch.matmul(x, self.W)
        h = h.view(b, n, self.heads, self.out_dim)
        h_i = h.unsqueeze(2).expand(-1, -1, n, -1, -1) 
        h_j = h.unsqueeze(1).expand(-1, n, -1, -1, -1) 
        a_input = torch.cat([h_i, h_j], dim=-1)         
        e = self.leaky_relu(torch.einsum("bijhd,dh->bijh", a_input, self.a))
        mask = adj.unsqueeze(-1).expand(-1, -1, -1, self.heads)
        att = torch.where(mask > 0, e, torch.full_like(e, -1e9))
        att = F.softmax(att, dim=2)  
        att = self.dropout(att)
        h_out = torch.einsum("bijh,bjhd->bihd", att, h)
        h_out = h_out.mean(dim=2)  
        if not is_batched:
            h_out = h_out.squeeze(0)    
        return F.elu(h_out)


class CorrelationGNN(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int = 128, encoding_size: int = 64,
                 num_layers: int = 3, layer_type: Literal["gcn", "gat"] = "gcn",
                 gat_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.num_layers = num_layers
        layers = []
        dims = [node_dim] + [hidden_dim] * (num_layers - 1) + [encoding_size]
        for i in range(num_layers):
            if layer_type == "gat":
                layers.append(GraphAttentionLayer(dims[i], dims[i + 1],heads=gat_heads, dropout=dropout))
            else:
                layers.append(GraphConvLayer(dims[i], dims[i + 1], dropout=dropout))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, adj)
        return F.normalize(x, dim=-1)


class TemporalGNNEncoder(nn.Module):
    def __init__(self, temporal_encoder: nn.Module, gnn: CorrelationGNN):
        super().__init__()
        self.temporal_encoder = temporal_encoder
        self.gnn = gnn

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 4:
            b, n, f, w = x.shape
            x_reshaped = x.view(b * n, f, w)
            node_embs = self.temporal_encoder(x_reshaped)  
            node_embs = node_embs.view(b, n, -1)
        else:
            node_embs = self.temporal_encoder(x) 
        return self.gnn(node_embs, adj)


def correlation_to_adj(corr: torch.Tensor, threshold: float = 0.3, top_k: int | None = 20,
                       self_loop: bool = True) -> torch.Tensor:
    adj = corr.abs()
    if threshold > 0:
        adj = adj * (adj > threshold).float()
    if top_k is not None:
        topk_vals, _ = adj.topk(min(top_k, adj.size(-1)), dim=-1)
        min_vals = topk_vals[..., -1:]
        adj = adj * (adj >= min_vals).float()
    if self_loop:
        n_stocks = adj.size(-1)
        eye = torch.eye(n_stocks, device=adj.device)
        if len(adj.shape) == 3:
            eye = eye.unsqueeze(0)
        adj = adj + eye
    deg_sum = adj.sum(dim=-1).clamp(min=1e-8)
    d_inv = torch.diag_embed(deg_sum.pow(-0.5))
    return d_inv @ adj @ d_inv
