from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

from .losses import angular_error_deg, l1_gaze_loss, mmd_loss, multiscale_zone_loss, triplet_gaze_loss


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        "image": batch["image"].to(device),
        "gaze": batch["gaze"].to(device),
        "zones": {k: v.to(device) for k, v in batch["zones"].items()},
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    jitter,
    loss_cfg: dict,
    epoch: int = 0,
):
    model.train()
    running = {"loss": 0.0, "angular": 0.0, "count": 0}
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch in pbar:
        batch = move_batch_to_device(batch, device)
        images = batch["image"]
        gaze = batch["gaze"]

        disturbed = torch.stack([jitter(img) for img in images], dim=0)

        out_ori = model(images)
        out_con = model(disturbed)

        loss_ori = l1_gaze_loss(out_ori["gaze"], gaze)
        loss_gcon = l1_gaze_loss(out_con["gaze"], gaze)
        loss_feature = mmd_loss(out_ori["embedding"], out_con["embedding"])
        loss_con = loss_gcon + loss_cfg["lambda_feature"] * loss_feature
        loss_zone = multiscale_zone_loss(out_ori["zones"], batch["zones"])
        loss_triplet = triplet_gaze_loss(out_ori["embedding"], gaze, loss_cfg["triplet_margin"])
        loss_triplet = loss_triplet + triplet_gaze_loss(out_con["embedding"], gaze, loss_cfg["triplet_margin"])

        total = (
            loss_ori
            + loss_cfg["lambda_con"] * loss_con
            + loss_cfg["lambda_triplet"] * loss_triplet
            + loss_cfg["lambda_zone"] * loss_zone
        )

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()

        angular = angular_error_deg(out_ori["gaze"].detach(), gaze).mean().item()
        running["loss"] += total.item() * images.size(0)
        running["angular"] += angular * images.size(0)
        running["count"] += images.size(0)
        
        pbar.set_postfix({
            "loss": f"{total.item():.4f}",
            "ang": f"{angular:.2f}°"
        })

    denom = max(running["count"], 1)
    return {
        "loss": running["loss"] / denom,
        "angular_error_deg": running["angular"] / denom,
    }


@torch.no_grad()
def evaluate_model(model, loader, device, epoch: int = 0):
    model.eval()
    total_loss = 0.0
    total_ang = 0.0
    total_count = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]", leave=False)
    for batch in pbar:
        batch = move_batch_to_device(batch, device)
        out = model(batch["image"])
        loss = l1_gaze_loss(out["gaze"], batch["gaze"])
        ang = angular_error_deg(out["gaze"], batch["gaze"]).mean().item()
        batch_size = batch["image"].size(0)
        total_loss += loss.item() * batch_size
        total_ang += ang * batch_size
        total_count += batch_size
        
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "ang": f"{ang:.2f}°"
        })

    denom = max(total_count, 1)
    return {
        "loss": total_loss / denom,
        "angular_error_deg": total_ang / denom,
    }


def save_checkpoint(path: str | Path, model, optimizer, config: dict, labeler_state: dict, metrics: dict):
    if hasattr(model, "module"):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    torch.save(
        {
            "model": model_state,
            "optimizer": optimizer.state_dict(),
            "config": config,
            "labeler": labeler_state,
            "metrics": metrics,
        },
        path,
    )
