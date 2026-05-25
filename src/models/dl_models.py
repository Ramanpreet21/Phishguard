"""
src/models/dl_models.py
=======================
PyTorch architectures for URL-level phishing detection.

All three models accept the same input:
    x : LongTensor [batch, MAX_URL_LEN]  (character IDs, 0=PAD)

All three output:
    logits : FloatTensor [batch]  (raw scores, apply sigmoid for probability)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.features import VOCAB_SIZE, MAX_URL_LEN


# ────────────────────────────────────────────────────────────────
# Bidirectional LSTM
# ────────────────────────────────────────────────────────────────

class URLLSTMClassifier(nn.Module):
    """
    Char-level bidirectional LSTM.
    Embeddings → BiLSTM → concat last hidden states → Linear.
    """

    def __init__(
        self,
        vocab_size: int   = VOCAB_SIZE,
        embed_dim:  int   = 64,
        hidden_dim: int   = 128,
        num_layers: int   = 2,
        dropout:    float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb          = self.drop(self.embedding(x))                     # [B, L, E]
        _, (h, _)    = self.lstm(emb)                                   # h: [2*layers, B, H]
        # Take last layer forward + backward
        h_fwd = h[-2]   # forward  last layer
        h_bwd = h[-1]   # backward last layer
        h_cat = torch.cat([h_fwd, h_bwd], dim=1)                       # [B, 2H]
        return self.fc(self.drop(h_cat)).squeeze(1)                     # [B]


# ────────────────────────────────────────────────────────────────
# Character-level CNN  (Kim 2014 multi-kernel style)
# ────────────────────────────────────────────────────────────────

class URLCNNClassifier(nn.Module):
    """
    Char-level CNN with multiple kernel widths.
    Embeddings → parallel Conv1d + max-pool → concat → Linear.
    """

    def __init__(
        self,
        vocab_size:   int       = VOCAB_SIZE,
        embed_dim:    int       = 64,
        num_filters:  int       = 128,
        kernel_sizes: list[int] = (3, 5, 7),
        dropout:      float     = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(num_filters * len(kernel_sizes), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb     = self.embedding(x).transpose(1, 2)                    # [B, E, L]
        pooled  = [F.relu(conv(emb)).max(dim=2)[0] for conv in self.convs]
        cat     = torch.cat(pooled, dim=1)                              # [B, F*K]
        return self.fc(self.drop(cat)).squeeze(1)                       # [B]


# ────────────────────────────────────────────────────────────────
# Transformer encoder
# ────────────────────────────────────────────────────────────────

class URLTransformerClassifier(nn.Module):
    """
    Char-level Transformer encoder with learned positional embeddings.
    Masked mean-pooling → Linear classification head.
    """

    def __init__(
        self,
        vocab_size:     int   = VOCAB_SIZE,
        embed_dim:      int   = 64,
        nhead:          int   = 4,
        num_layers:     int   = 2,
        ff_dim:         int   = 256,
        dropout:        float = 0.1,
        max_len:        int   = MAX_URL_LEN,
    ):
        super().__init__()
        self.char_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_emb  = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,        # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L     = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0)      # [1, L]
        emb      = self.char_emb(x) + self.pos_emb(positions)          # [B, L, E]
        emb      = self.drop(emb)

        pad_mask = (x == 0)                                             # [B, L]  True=ignore
        out      = self.transformer(emb, src_key_padding_mask=pad_mask) # [B, L, E]

        # Masked mean-pool over non-padding positions
        non_pad  = (~pad_mask).unsqueeze(-1).float()                    # [B, L, 1]
        pooled   = (out * non_pad).sum(dim=1) / non_pad.sum(dim=1).clamp(min=1e-9)
        return self.fc(self.drop(pooled)).squeeze(1)                    # [B]


# ────────────────────────────────────────────────────────────────
# Dataset helper
# ────────────────────────────────────────────────────────────────

class URLDataset(torch.utils.data.Dataset):
    """Simple dataset wrapper for URL char-ID sequences + labels."""

    def __init__(self, ids: list[list[int]], labels: list[int]):
        self.ids    = torch.tensor(ids,    dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.ids[idx], self.labels[idx]
