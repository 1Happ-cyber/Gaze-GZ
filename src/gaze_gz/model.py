from __future__ import annotations

import torch
import torch.nn as nn


class SmallCNNEncoder(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).flatten(1)
        return self.proj(x)


def build_encoder(use_resnet50: bool, embedding_dim: int) -> nn.Module:
    if use_resnet50:
        try:
            from torchvision.models import resnet50

            backbone = resnet50(weights=None)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Linear(in_features, embedding_dim)
            return backbone
        except Exception:
            pass
    return SmallCNNEncoder(embedding_dim)


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GazeGZModel(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, zone_scales: list[int], use_resnet50: bool = True):
        super().__init__()
        self.encoder = build_encoder(use_resnet50=use_resnet50, embedding_dim=embedding_dim)
        self.regressor = MLPHead(embedding_dim, hidden_dim, 2)
        self.zone_heads = nn.ModuleDict(
            {str(scale): MLPHead(embedding_dim, hidden_dim, scale) for scale in zone_scales}
        )

    def forward(self, x: torch.Tensor) -> dict:
        embedding = self.encoder(x)
        gaze = self.regressor(embedding)
        zones = {scale: head(embedding) for scale, head in self.zone_heads.items()}
        return {
            "embedding": embedding,
            "gaze": gaze,
            "zones": zones,
        }
