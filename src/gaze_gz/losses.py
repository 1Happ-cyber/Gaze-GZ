from __future__ import annotations

import torch
import torch.nn.functional as F


def l1_gaze_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


def mmd_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mean_x = x.mean(dim=0)
    mean_y = y.mean(dim=0)
    return ((mean_x - mean_y) ** 2).mean()


def multiscale_zone_loss(pred_zone_logits: dict[str, torch.Tensor], target_zones: dict[str, torch.Tensor]) -> torch.Tensor:
    losses = []
    for scale, logits in pred_zone_logits.items():
        losses.append(F.cross_entropy(logits, target_zones[scale]))
    return torch.stack(losses).mean()


def sample_triplets(embeddings: torch.Tensor, gazes: torch.Tensor):
    with torch.no_grad():
        dist = torch.cdist(gazes, gazes, p=2)
        eye = torch.eye(dist.size(0), device=dist.device, dtype=torch.bool)
        pos_dist = dist.masked_fill(eye, 1e9)
        neg_dist = dist.masked_fill(eye, -1.0)
        pos_idx = pos_dist.argmin(dim=1)
        neg_idx = neg_dist.argmax(dim=1)
    anchor = embeddings
    positive = embeddings[pos_idx]
    negative = embeddings[neg_idx]
    return anchor, positive, negative


def triplet_gaze_loss(embeddings: torch.Tensor, gazes: torch.Tensor, margin: float) -> torch.Tensor:
    if embeddings.size(0) < 3:
        return embeddings.new_tensor(0.0)
    anchor, positive, negative = sample_triplets(embeddings, gazes)
    return F.triplet_margin_loss(anchor, positive, negative, margin=margin, p=2)


def angles_to_unit_vectors(gaze: torch.Tensor) -> torch.Tensor:
    pitch = gaze[:, 0]
    yaw = gaze[:, 1]
    x = torch.cos(pitch) * torch.sin(yaw)
    y = torch.sin(pitch)
    z = torch.cos(pitch) * torch.cos(yaw)
    return torch.stack([x, y, z], dim=1)


def angular_error_deg(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_vec = angles_to_unit_vectors(pred)
    target_vec = angles_to_unit_vectors(target)
    cosine = F.cosine_similarity(pred_vec, target_vec, dim=1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))
