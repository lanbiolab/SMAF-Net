from __future__ import annotations

import os
os.environ.setdefault("TORCHVISION_DISABLE_NMS_EXPORT", "1")

import torch
import torch.nn as nn


def _patch_torchvision_register_fake():
    original = getattr(torch.library, "register_fake", None)
    if original is None:
        return lambda: None

    def safe_register_fake(op_name):
        decorator = original(op_name)

        def wrapper(fn):
            try:
                return decorator(fn)
            except RuntimeError as exc:
                if "does not exist" in str(exc):
                    return fn
                raise

        return wrapper

    torch.library.register_fake = safe_register_fake

    def restore():
        torch.library.register_fake = original

    return restore


_restore_torchvision_patch = _patch_torchvision_register_fake()
import torchvision.models as models
if _restore_torchvision_patch is not None:
    _restore_torchvision_patch()


def _build_efficientnet_b0(pretrained: bool = True):
    try:
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        return models.efficientnet_b0(weights=weights)
    except Exception:
        return models.efficientnet_b0(pretrained=pretrained)


class TokenRefiner(nn.Module):
    def __init__(self, feature_dim: int, dropout: float = 0.1):
        super().__init__()
        hidden_dim = max(feature_dim * 2, 64)
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.out_norm = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_norm(x + self.net(x))


class EfficientNetTokenEncoder(nn.Module):
    """Fundus encoder: EfficientNet-B0 + token pooling."""

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        feature_dim: int = 256,
        token_grid: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        if backbone_name != "efficientnet_b0":
            raise ValueError(f"Only efficientnet_b0 is supported, got {backbone_name}")

        backbone = _build_efficientnet_b0(pretrained=pretrained)
        self.features = backbone.features
        self.out_channels = backbone.classifier[1].in_features
        self.project = nn.Conv2d(self.out_channels, feature_dim, kernel_size=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.token_pool = nn.AdaptiveAvgPool2d((token_grid, token_grid))
        self.norm = nn.LayerNorm(feature_dim)
        self.refiner = TokenRefiner(feature_dim=feature_dim, dropout=dropout)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, x: torch.Tensor):
        fmap = self.features(x)
        proj = self.project(fmap)
        global_feat = self.pool(proj).flatten(1)
        tokens = self.token_pool(proj).flatten(2).transpose(1, 2)
        global_feat = self.refiner(self.norm(global_feat))
        tokens = self.refiner(self.norm(tokens))
        return global_feat, tokens


class OCTSequenceEncoder(nn.Module):
    """OCT encoder: EfficientNet-B0 per slice + mean pooling."""

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        feature_dim: int = 256,
        dropout: float = 0.1,
        encode_chunk_size: int = 8,
    ):
        super().__init__()
        if backbone_name != "efficientnet_b0":
            raise ValueError(f"Only efficientnet_b0 is supported, got {backbone_name}")

        backbone = _build_efficientnet_b0(pretrained=pretrained)
        self.features = backbone.features
        self.out_channels = backbone.classifier[1].in_features
        self.project = nn.Conv2d(self.out_channels, feature_dim, kernel_size=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(feature_dim)
        self.refiner = TokenRefiner(feature_dim=feature_dim, dropout=dropout)
        self.encode_chunk_size = max(1, int(encode_chunk_size))

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def _encode_slices(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.features(x)
        proj = self.project(fmap)
        return self.pool(proj).flatten(1)

    def forward(self, volume: torch.Tensor):
        b, s, c, h, w = volume.shape
        flat = volume.reshape(b * s, c, h, w)
        chunks = []
        for start in range(0, flat.size(0), self.encode_chunk_size):
            chunks.append(self._encode_slices(flat[start:start + self.encode_chunk_size]))
        slice_feats = torch.cat(chunks, dim=0).reshape(b, s, -1)
        slice_feats = self.refiner(self.norm(slice_feats))
        global_feat = self.refiner(self.norm(slice_feats.mean(dim=1)))
        return global_feat, slice_feats
