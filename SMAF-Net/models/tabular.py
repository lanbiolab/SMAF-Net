from __future__ import annotations

import torch
import torch.nn as nn


class NumericalFeatureTokenizer(nn.Module):
    def __init__(self, input_dim: int, feature_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.feature_dim = int(feature_dim)
        self.weight = nn.Parameter(torch.empty(self.input_dim, self.feature_dim))
        self.bias = nn.Parameter(torch.zeros(self.input_dim, self.feature_dim))
        self.feature_embed = nn.Parameter(torch.empty(self.input_dim, self.feature_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        nn.init.normal_(self.feature_embed, std=0.02)

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(-1)
        return x * self.weight.unsqueeze(0) + self.bias.unsqueeze(0) + self.feature_embed.unsqueeze(0)


class AttentiveFeaturePool(nn.Module):
    def __init__(self, feature_dim: int, dropout: float = 0.1):
        super().__init__()
        hidden = max(feature_dim // 2, 32)
        self.score = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, tokens: torch.Tensor):
        logits = self.score(tokens).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        pooled = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class TabularTokenEncoder(nn.Module):
    """FT-Transformer style encoder for structured text/tabular features."""

    def __init__(
        self,
        input_dim: int,
        feature_dim: int,
        depth: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.feature_dim = int(feature_dim)
        self.tokenizer = NumericalFeatureTokenizer(self.input_dim, self.feature_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.feature_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=num_heads,
            dim_feedforward=self.feature_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.pool = AttentiveFeaturePool(self.feature_dim, dropout=dropout)
        self.norm = nn.LayerNorm(self.feature_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor):
        feat_tokens = self.tokenizer(x)
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls_token, feat_tokens], dim=1)
        tokens = self.transformer(tokens)

        cls_feat = self.norm(tokens[:, 0])
        feature_tokens = self.norm(tokens[:, 1:])
        pooled_feat, weights = self.pool(feature_tokens)
        global_feat = self.norm(0.5 * (cls_feat + pooled_feat))
        extras = {"tabular_feature_weights": weights.detach()}
        return global_feat, feature_tokens, extras
