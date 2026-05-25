"""
src/models/dl_models.py
=======================
Multimodal PyTorch architectures for URL-level phishing detection.

Each model is a **two-branch network**:
  Branch 1 (Text):    char_ids   → Embedding → [LSTM / CNN / Transformer] → text_repr
  Branch 2 (Tabular): url_feats → MLP(22 → 64 → 32)                     → tab_repr
  Fusion:             concat(text_repr, tab_repr) → FusionHead            → logit

All three models accept the same inputs:
    x    : LongTensor  [batch, MAX_URL_LEN]    (character IDs, 0=PAD)
    tab  : FloatTensor [batch, N_TAB_FEATURES] (scaled URL-derived features)

All three output:
    logits : FloatTensor [batch]  (raw scores, apply sigmoid for probability)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.features import VOCAB_SIZE, MAX_URL_LEN

# Default number of tabular features (URL-derived)
N_TAB_FEATURES = 22


# ────────────────────────────────────────────────────────────────
# Shared components
# ────────────────────────────────────────────────────────────────

class TabularBranch(nn.Module):
    """
    Small MLP that projects tabular features into a dense representation.
    30 → 64 → ReLU → Dropout → 32
    """

    def __init__(self, n_features: int = N_TAB_FEATURES, hidden: int = 64,
                 out_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.ReLU(),
        )

    def forward(self, tab: torch.Tensor) -> torch.Tensor:
        return self.net(tab)                                            # [B, out_dim]


class FusionHead(nn.Module):
    """
    Classification head that operates on the concatenated text + tabular
    representations.
    (text_dim + tab_dim) → 64 → ReLU → Dropout → 1
    """

    def __init__(self, text_dim: int, tab_dim: int = 32,
                 hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim + tab_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, text_repr: torch.Tensor, tab_repr: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([text_repr, tab_repr], dim=1)                # [B, T+Tab]
        return self.net(fused).squeeze(1)                               # [B]


# ────────────────────────────────────────────────────────────────
# Bidirectional LSTM (Multimodal)
# ────────────────────────────────────────────────────────────────

class URLLSTMClassifier(nn.Module):
    """
    Char-level bidirectional LSTM + tabular feature fusion.
    Text:    Embeddings → BiLSTM → concat last hidden states → text_repr [B, 2H]
    Tabular: URL feats → TabularBranch                       → tab_repr  [B, 32]
    Fusion:  concat → FusionHead → logit
    """

    def __init__(
        self,
        vocab_size:     int   = VOCAB_SIZE,
        embed_dim:      int   = 64,
        hidden_dim:     int   = 128,
        num_layers:     int   = 2,
        dropout:        float = 0.3,
        n_tab_features: int   = N_TAB_FEATURES,
        tab_out_dim:    int   = 32,
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

        text_dim = hidden_dim * 2   # bidirectional
        self.tab_branch  = TabularBranch(n_tab_features, out_dim=tab_out_dim, dropout=dropout)
        self.fusion_head = FusionHead(text_dim, tab_out_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        # Text branch
        emb          = self.drop(self.embedding(x))                     # [B, L, E]
        _, (h, _)    = self.lstm(emb)                                   # h: [2*layers, B, H]
        h_fwd = h[-2]   # forward  last layer
        h_bwd = h[-1]   # backward last layer
        text_repr = torch.cat([h_fwd, h_bwd], dim=1)                   # [B, 2H]
        text_repr = self.drop(text_repr)

        # Tabular branch
        tab_repr = self.tab_branch(tab)                                 # [B, 32]

        # Fusion
        return self.fusion_head(text_repr, tab_repr)                    # [B]


# ────────────────────────────────────────────────────────────────
# Character-level CNN (Multimodal)
# ────────────────────────────────────────────────────────────────

class URLCNNClassifier(nn.Module):
    """
    Char-level CNN with multiple kernel widths + tabular feature fusion.
    Text:    Embeddings → parallel Conv1d + max-pool → concat → text_repr [B, F*K]
    Tabular: URL feats → TabularBranch                        → tab_repr  [B, 32]
    Fusion:  concat → FusionHead → logit
    """

    def __init__(
        self,
        vocab_size:     int       = VOCAB_SIZE,
        embed_dim:      int       = 64,
        num_filters:    int       = 128,
        kernel_sizes:   list[int] = (3, 5, 7),
        dropout:        float     = 0.3,
        n_tab_features: int       = N_TAB_FEATURES,
        tab_out_dim:    int       = 32,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.drop = nn.Dropout(dropout)

        text_dim = num_filters * len(kernel_sizes)
        self.tab_branch  = TabularBranch(n_tab_features, out_dim=tab_out_dim, dropout=dropout)
        self.fusion_head = FusionHead(text_dim, tab_out_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        # Text branch
        emb     = self.embedding(x).transpose(1, 2)                    # [B, E, L]
        pooled  = [F.relu(conv(emb)).max(dim=2)[0] for conv in self.convs]
        text_repr = torch.cat(pooled, dim=1)                            # [B, F*K]
        text_repr = self.drop(text_repr)

        # Tabular branch
        tab_repr = self.tab_branch(tab)                                 # [B, 32]

        # Fusion
        return self.fusion_head(text_repr, tab_repr)                    # [B]


# ────────────────────────────────────────────────────────────────
# Transformer encoder (Multimodal)
# ────────────────────────────────────────────────────────────────

class URLTransformerClassifier(nn.Module):
    """
    Char-level Transformer encoder + tabular feature fusion.
    Text:    Embeddings + PosEmb → TransformerEncoder → mean-pool → text_repr [B, E]
    Tabular: URL feats → TabularBranch                            → tab_repr  [B, 32]
    Fusion:  concat → FusionHead → logit
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
        n_tab_features: int   = N_TAB_FEATURES,
        tab_out_dim:    int   = 32,
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

        text_dim = embed_dim
        self.tab_branch  = TabularBranch(n_tab_features, out_dim=tab_out_dim, dropout=dropout)
        self.fusion_head = FusionHead(text_dim, tab_out_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        # Text branch
        B, L     = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0)      # [1, L]
        emb      = self.char_emb(x) + self.pos_emb(positions)          # [B, L, E]
        emb      = self.drop(emb)

        pad_mask = (x == 0)                                             # [B, L]  True=ignore
        out      = self.transformer(emb, src_key_padding_mask=pad_mask) # [B, L, E]

        # Masked mean-pool over non-padding positions
        non_pad  = (~pad_mask).unsqueeze(-1).float()                    # [B, L, 1]
        text_repr = (out * non_pad).sum(dim=1) / non_pad.sum(dim=1).clamp(min=1e-9)  # [B, E]
        text_repr = self.drop(text_repr)

        # Tabular branch
        tab_repr = self.tab_branch(tab)                                 # [B, 32]

        # Fusion
        return self.fusion_head(text_repr, tab_repr)                    # [B]


# ────────────────────────────────────────────────────────────────
# Dataset helper (multimodal)
# ────────────────────────────────────────────────────────────────

class URLDataset(torch.utils.data.Dataset):
    """Dataset wrapper for URL char-ID sequences + tabular features + labels."""

    def __init__(self, ids: list[list[int]], tab_features: list | None = None,
                 labels: list[int] = None):
        self.ids    = torch.tensor(ids, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32) if labels is not None \
                      else torch.zeros(len(ids), dtype=torch.float32)

        if tab_features is not None:
            if isinstance(tab_features, torch.Tensor):
                self.tab = tab_features.float()
            else:
                self.tab = torch.tensor(tab_features, dtype=torch.float32)
        else:
            # Fallback: zeros (no tabular features available)
            self.tab = torch.zeros(len(ids), N_TAB_FEATURES, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.ids[idx], self.tab[idx], self.labels[idx]
