"""
Stages 7-8: Model architectures (Table 4 and Table A.9), implemented in PyTorch.

  GRU         : GRU(50, return_sequences=True) -> Dropout -> GRU(50) -> Dropout -> Dense(1)
                Target parameter count from the paper: 24,351 (Table 5/A.9).
  LSTM        : LSTM(43, RS=True) -> LSTM(43) -> Dense(1), dropout 0.1 per layer.
                Target parameter count from the paper: 24,100 (Table 5).
  Transformer : linear embedding -> positional encoding -> encoder(3 layers, 4 heads,
                d_model=64) -> decoder(3 layers) -> Linear(1).

All recurrent models take input of shape (batch, LOOKBACK, n_features) and output
a single scalar (next-step pH).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from config import (
    GRU_DROPOUT,
    GRU_UNITS,
    LSTM_DROPOUT,
    LSTM_UNITS,
    TRANSFORMER_D_MODEL,
    TRANSFORMER_DROPOUT,
    TRANSFORMER_HEADS,
    TRANSFORMER_LAYERS,
)


# ----------------------------------------------------------------------------
# GRU (the proposed model)
# ----------------------------------------------------------------------------
class GRUModel(nn.Module):
    def __init__(self, n_features: int, units: int = GRU_UNITS,
                 dropout: float = GRU_DROPOUT):
        super().__init__()
        # Two stacked GRU layers, each with `units` hidden size.
        # Keras "return_sequences=True" then a second GRU == two nn.GRU layers.
        self.gru1 = nn.GRU(input_size=n_features, hidden_size=units,
                           batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.gru2 = nn.GRU(input_size=units, hidden_size=units,
                           batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc = nn.Linear(units, 1)

    def forward(self, x):
        out, _ = self.gru1(x)          # (batch, seq, units) -> return sequences
        out = self.drop1(out)
        out, h = self.gru2(out)        # take last hidden state
        last = out[:, -1, :]           # (batch, units)
        last = self.drop2(last)
        return self.fc(last).squeeze(-1)


# ----------------------------------------------------------------------------
# LSTM baseline
# ----------------------------------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, n_features: int, units: int = LSTM_UNITS,
                 dropout: float = LSTM_DROPOUT):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=units,
                             batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(input_size=units, hidden_size=units,
                             batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc = nn.Linear(units, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        last = out[:, -1, :]
        last = self.drop2(last)
        return self.fc(last).squeeze(-1)


# ----------------------------------------------------------------------------
# Transformer baseline (encoder-decoder, Table 4)
# ----------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerModel(nn.Module):
    """
    Encoder input: the LOOKBACK window (seq_len=10, d features).
    Decoder input: a single learned query token (target seq len = 1), following
    the paper's decoder input (1, 1). Output: Linear(1) -> next-step pH.
    """
    def __init__(self, n_features: int,
                 d_model: int = TRANSFORMER_D_MODEL,
                 nhead: int = TRANSFORMER_HEADS,
                 num_layers: int = TRANSFORMER_LAYERS,
                 dropout: float = TRANSFORMER_DROPOUT):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers, num_decoder_layers=num_layers,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True,
        )
        # Learned decoder query (single step).
        self.dec_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        b = x.size(0)
        src = self.pos_enc(self.input_proj(x))          # (b, seq, d_model)
        tgt = self.dec_token.expand(b, 1, -1)           # (b, 1, d_model)
        out = self.transformer(src, tgt)                # (b, 1, d_model)
        return self.fc(out[:, -1, :]).squeeze(-1)


# ----------------------------------------------------------------------------
# Factory + utilities
# ----------------------------------------------------------------------------
def build_model(name: str, n_features: int) -> nn.Module:
    name = name.lower()
    if name == "gru":
        return GRUModel(n_features)
    if name == "lstm":
        return LSTMModel(n_features)
    if name == "transformer":
        return TransformerModel(n_features)
    raise ValueError(f"unknown model '{name}'")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Paper param counts (Table 5 / A.9) correspond to 8 input features:
    #   GRU 24,351 ; LSTM 24,100 both match n_features=8 exactly.
    print("With n_features=8 (matches Table A.9 (batch,10,8)):")
    for name, paper in [("gru", 24351), ("lstm", 24100),
                        ("transformer", 251585)]:
        m = build_model(name, n_features=8)
        print(f"  {name:>12s}: {count_parameters(m):,} params "
              f"(paper: {paper:,})")
    print("\nWith n_features=6 (all sensors, our primary config):")
    for name in ("gru", "lstm", "transformer"):
        m = build_model(name, n_features=6)
        print(f"  {name:>12s}: {count_parameters(m):,} params")
