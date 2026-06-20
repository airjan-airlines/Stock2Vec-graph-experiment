from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                                     padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size,
                                                     padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.drop1,
                                  self.conv2, self.chomp2, self.relu2, self.drop2)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNEncoder(nn.Module):
    def __init__(self, in_channels: int, encoding_size: int = 64, num_channels: list[int] | None = None,
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        channels = num_channels or [64, 64, 128, 128]
        layers = []
        for i, out_ch in enumerate(channels):
            layers.append(TemporalBlock(in_channels if i == 0 else channels[i - 1],
                                         out_ch, kernel_size, dilation=2 ** i, dropout=dropout))
        self.tcn = nn.Sequential(*layers)
        self.proj = nn.Linear(channels[-1], encoding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.tcn(x)
        out = out[:, :, -1]
        return self.proj(out)


class CNNStockEncoder(nn.Module):
    def __init__(self, input_channels: int, encoding_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, encoding_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalTransformerEncoder(nn.Module):
    def __init__(self, in_channels: int, encoding_size: int = 64,
                 d_model: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4, dropout=dropout,
                                            activation="relu", batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers)
        self.output_proj = nn.Linear(d_model, encoding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        return self.output_proj(x[:, -1])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class MacroConditionedEncoder(nn.Module):
    def __init__(self, temporal_encoder: nn.Module, n_macro: int = 8, encoding_size: int = 64):
        super().__init__()
        self.temporal_encoder = temporal_encoder
        self.macro_head = nn.Sequential(
            nn.Linear(n_macro, encoding_size),
            nn.ReLU(),
            nn.Linear(encoding_size, encoding_size),
        )
        self.fusion = nn.Linear(encoding_size * 2, encoding_size)

    def forward(self, x: torch.Tensor, macro: torch.Tensor | None = None) -> torch.Tensor:
        z = self.temporal_encoder(x)
        if macro is not None:
            m = self.macro_head(macro)
            z = self.fusion(torch.cat([z, m], dim=-1))
        return F.normalize(z, dim=-1)
