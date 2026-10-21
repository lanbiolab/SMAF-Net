from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from .encoders import EfficientNetTokenEncoder, OCTSequenceEncoder
from .tabular import TabularTokenEncoder


class CrossModalAttentionBlock(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(feature_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor):
        h = self.norm1(x)
        attn_out, attn_weights = self.attn(
            h,
            h,
            h,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, attn_weights


class MaskedCrossModalFusion(nn.Module):
    """Mask-aware cross-modal attention fusion with residual masked-mean safeguard."""

    def __init__(
        self,
        feature_dim: int,
        num_modalities: int,
        num_heads: int = 4,
        depth: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Keep v2 stable: force shallow fusion by default.
        depth = int(depth)
        if depth <= 0:
            depth = 1

        self.feature_dim = int(feature_dim)
        self.num_modalities = int(num_modalities)

        self.modality_embed = nn.Parameter(torch.randn(1, num_modalities, feature_dim) * 0.02)
        self.missing_embed = nn.Parameter(torch.randn(1, num_modalities, feature_dim) * 0.02)

        self.layers = nn.ModuleList(
            [
                CrossModalAttentionBlock(feature_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(depth)
            ]
        )

        self.pool_query = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Initial sigmoid(0.35) ~= 0.586. Fusion is useful but residual stays strong.
        self.cross_gate = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))
        self.out_norm = nn.LayerNorm(feature_dim)

    def _masked_mean(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # tokens: [B, M, D], mask: [B, M], True=present
        mask_f = mask.float().unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        return (tokens * mask_f).sum(dim=1) / denom

    def forward(self, modality_tokens: torch.Tensor, modality_mask: torch.Tensor):
        modality_mask = modality_mask.bool()
        key_padding_mask = ~modality_mask

        x = modality_tokens + self.modality_embed[:, : modality_tokens.size(1), :]
        missing_token = self.missing_embed[:, : modality_tokens.size(1), :].expand(x.size(0), -1, -1)
        x = torch.where(modality_mask.unsqueeze(-1), x, missing_token)

        attn_maps = []
        for layer in self.layers:
            x, attn_weights = layer(x, key_padding_mask=key_padding_mask)
            attn_maps.append(attn_weights.detach())

        pool_q = self.pool_query.expand(x.size(0), -1, -1)
        pooled, pool_weights = self.pool_attn(
            pool_q,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )

        cross_feat = pooled.squeeze(1)
        residual_feat = self._masked_mean(modality_tokens, modality_mask)

        gate = torch.sigmoid(self.cross_gate)
        fused = self.out_norm(gate * cross_feat + (1.0 - gate) * residual_feat)

        extras = {
            "fusion_tokens": x,
            "fusion_attn_weights": attn_maps,
            "fusion_pool_weights": pool_weights.detach(),
            "fusion_cross_gate": gate.detach(),
            "fusion_cross_feat": cross_feat,
            "fusion_residual_feat": residual_feat,
        }
        return fused, extras


class BaselineConcatModel(nn.Module):
    """
    Robust masked cross-attention model.

    Main changes compared with your previous cross-attention model:
    1. Adds modality-specific auxiliary heads: fundus_head / oct_head / tabular_head.
    2. Returns auxiliary logits for auxiliary losses in trainer.py.
    3. Supports dynamic routing for single-modality inference:
       if final_logit_mode='single_when_one', samples with exactly one available modality
       use the corresponding modality-specific head as final output.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.feature_dim = int(cfg["feature_dim"])
        self.use_fundus = bool(cfg.get("use_fundus", True))
        self.use_oct = bool(cfg.get("use_oct", True))
        self.use_tabular = bool(cfg.get("use_tabular", True))
        self.dropout = float(cfg.get("dropout", 0.1))

        self.use_aux_heads = bool(cfg.get("use_aux_heads", True))
        self.final_logit_mode = str(cfg.get("final_logit_mode", "single_when_one"))
        self.double_modal_blend_weight = float(cfg.get("double_modal_blend_weight", 0.0))
        self.use_mask_aware_fusion = bool(cfg.get("use_mask_aware_fusion", True))
        self.fusion_type = str(cfg.get("fusion_type", "attention")).lower()
        if self.fusion_type not in {"attention", "mean", "sum", "concat"}:
            raise ValueError(
                f"Unsupported fusion_type={self.fusion_type!r}. "
                "Expected one of: attention, mean, sum, concat."
            )

        if not (self.use_fundus or self.use_oct or self.use_tabular):
            raise ValueError("At least one modality must be enabled.")

        # Fixed dataset mask order: [fundus, oct, tabular]
        self.modality_index_map = {"fundus": 0, "oct": 1, "tabular": 2}
        self.modality_names: List[str] = []

        if self.use_fundus:
            self.fundus_encoder = EfficientNetTokenEncoder(
                backbone_name=cfg.get("backbone", "efficientnet_b0"),
                pretrained=bool(cfg.get("pretrained", True)),
                feature_dim=self.feature_dim,
                token_grid=int(cfg.get("fundus_token_grid", 6)),
                dropout=self.dropout,
            )
            self.modality_names.append("fundus")

        if self.use_oct:
            self.oct_encoder = OCTSequenceEncoder(
                backbone_name=cfg.get("oct_backbone", cfg.get("backbone", "efficientnet_b0")),
                pretrained=bool(cfg.get("pretrained", True)),
                feature_dim=self.feature_dim,
                dropout=self.dropout,
                encode_chunk_size=int(cfg.get("oct_encode_chunk_size", 8)),
            )
            self.modality_names.append("oct")

        if self.use_tabular:
            self.tabular_encoder = TabularTokenEncoder(
                input_dim=int(cfg["tabular_input_dim"]),
                feature_dim=self.feature_dim,
                depth=int(cfg.get("tabular_depth", 2)),
                num_heads=int(cfg.get("tabular_heads", 4)),
                dropout=self.dropout,
            )
            self.modality_names.append("tabular")

        if bool(cfg.get("freeze_backbone_bn", True)):
            if self.use_fundus:
                self.fundus_encoder.freeze_bn()
            if self.use_oct:
                self.oct_encoder.freeze_bn()

        if self.fusion_type == "attention":
            self.fusion = MaskedCrossModalFusion(
                feature_dim=self.feature_dim,
                num_modalities=len(self.modality_names),
                num_heads=int(cfg.get("fusion_heads", 4)),
                depth=int(cfg.get("fusion_depth", 1)),
                dropout=float(cfg.get("fusion_dropout", self.dropout)),
            )
        else:
            self.fusion = None

        hidden_dim = int(cfg.get("classifier_hidden_dim", self.feature_dim))
        classifier_input_dim = (
            self.feature_dim * len(self.modality_names)
            if self.fusion_type == "concat"
            else self.feature_dim
        )

        def make_head(input_dim: int = self.feature_dim) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(hidden_dim, 1),
            )

        self.classifier = make_head(classifier_input_dim)

        # Auxiliary heads are separate lightweight heads on each global modality feature.
        self.fundus_head = make_head() if (self.use_aux_heads and self.use_fundus) else None
        self.oct_head = make_head() if (self.use_aux_heads and self.use_oct) else None
        self.tabular_head = make_head() if (self.use_aux_heads and self.use_tabular) else None

    def _collect_features(self, fundus, oct_volume, tabular, modality_mask):
        features: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        extras: Dict[str, torch.Tensor] = {}

        modality_mask = modality_mask.bool()

        if self.use_fundus:
            fundus_global, fundus_tokens = self.fundus_encoder(fundus)
            features.append(fundus_global)
            masks.append(modality_mask[:, self.modality_index_map["fundus"]])
            extras["fundus_global"] = fundus_global
            extras["fundus_tokens"] = fundus_tokens

        if self.use_oct:
            oct_global, oct_tokens = self.oct_encoder(oct_volume)
            features.append(oct_global)
            masks.append(modality_mask[:, self.modality_index_map["oct"]])
            extras["oct_global"] = oct_global
            extras["oct_tokens"] = oct_tokens

        if self.use_tabular:
            tab_global, tab_tokens, tab_extras = self.tabular_encoder(tabular)
            features.append(tab_global)
            masks.append(modality_mask[:, self.modality_index_map["tabular"]])
            extras["tabular_global"] = tab_global
            extras["tabular_tokens"] = tab_tokens
            extras.update(tab_extras)
        ##涓轰簡娑堣瀺 杩涜浜嗕慨鏀?        modality_tokens = torch.stack(features, dim=1)
        modality_tokens = torch.stack(features, dim=1)
        selected_mask = torch.stack(masks, dim=1).bool()

        if self.use_mask_aware_fusion:
            fusion_mask = selected_mask
        else:
            fusion_mask = torch.ones_like(selected_mask, dtype=torch.bool)

        if self.fusion_type == "attention":
            fused, fusion_extras = self.fusion(modality_tokens, fusion_mask)
        else:
            mask_f = fusion_mask.float().unsqueeze(-1)
            masked_tokens = modality_tokens * mask_f
            if self.fusion_type == "concat":
                fused = masked_tokens.flatten(start_dim=1)
            else:
                fused_sum = masked_tokens.sum(dim=1)
                if self.fusion_type == "mean":
                    fused = fused_sum / mask_f.sum(dim=1).clamp_min(1.0)
                else:
                    fused = fused_sum
            fusion_extras = {
                "fusion_tokens": modality_tokens,
                "fusion_attn_weights": [],
                "fusion_pool_weights": None,
                "fusion_cross_gate": None,
                "fusion_cross_feat": fused,
                "fusion_residual_feat": fused,
            }

        extras["selected_modality_mask"] = selected_mask
        extras["fusion_mask_used"] = fusion_mask
        extras["modality_tokens_before_fusion"] = modality_tokens
        extras["fusion_type"] = self.fusion_type
        extras.update(fusion_extras)
        return fused, extras

    def _route_logits(
        self,
        fusion_logits: torch.Tensor,
        aux_logits: Dict[str, torch.Tensor],
        selected_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dynamic routing based on available modalities.

        selected_mask is in self.modality_names order, e.g. ['fundus', 'oct', 'tabular'].
        """
        final_logits = fusion_logits.clone()

        if self.final_logit_mode == "fusion":
            return final_logits

        num_present = selected_mask.sum(dim=1)

        # Single-modality routing: use the corresponding auxiliary head.
        if self.final_logit_mode in {"single_when_one", "single_and_blend_double"}:
            for local_idx, name in enumerate(self.modality_names):
                key = f"{name}_logits"
                if key not in aux_logits:
                    continue
                idx = (num_present == 1) & selected_mask[:, local_idx]
                if idx.any():
                    final_logits[idx] = aux_logits[key][idx]

        # Optional double-modality blending:
        # final = (1-w) * fusion + w * mean(aux logits of available modalities)
        # Keep default w=0.0 unless you explicitly want to try it.
        if self.final_logit_mode == "single_and_blend_double" and self.double_modal_blend_weight > 0:
            w = max(0.0, min(float(self.double_modal_blend_weight), 1.0))
            double_idx = num_present == 2
            if double_idx.any():
                stacked = []
                for local_idx, name in enumerate(self.modality_names):
                    key = f"{name}_logits"
                    if key not in aux_logits:
                        continue
                    logits = aux_logits[key]
                    present_f = selected_mask[:, local_idx].float()
                    stacked.append(logits * present_f)
                if stacked:
                    aux_sum = torch.stack(stacked, dim=0).sum(dim=0)
                    denom = num_present.float().clamp_min(1.0)
                    aux_mean = aux_sum / denom
                    final_logits[double_idx] = (
                        (1.0 - w) * fusion_logits[double_idx]
                        + w * aux_mean[double_idx]
                    )

        return final_logits

    def forward(self, fundus, oct_volume, tabular, modality_mask):
        fused, extras = self._collect_features(fundus, oct_volume, tabular, modality_mask)

        fusion_logits = self.classifier(fused).squeeze(1)

        aux_logits: Dict[str, torch.Tensor] = {}
        if self.fundus_head is not None and "fundus_global" in extras:
            aux_logits["fundus_logits"] = self.fundus_head(extras["fundus_global"]).squeeze(1)
        if self.oct_head is not None and "oct_global" in extras:
            aux_logits["oct_logits"] = self.oct_head(extras["oct_global"]).squeeze(1)
        if self.tabular_head is not None and "tabular_global" in extras:
            aux_logits["tabular_logits"] = self.tabular_head(extras["tabular_global"]).squeeze(1)

        final_logits = self._route_logits(
            fusion_logits=fusion_logits,
            aux_logits=aux_logits,
            selected_mask=extras["selected_modality_mask"],
        )

        return {
            "logits": final_logits,
            "prob": torch.sigmoid(final_logits),
            "fusion_logits": fusion_logits,
            **aux_logits,
            "fused_feature": fused,
            **extras,
        }



