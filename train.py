from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.utils.data import DataLoader

from gaze_gz.config import load_config
from gaze_gz.dataset import GazeDataset, load_gaze_array
from gaze_gz.model import GazeGZModel
from gaze_gz.trainer import evaluate_model, save_checkpoint, train_one_epoch
from gaze_gz.utils import ensure_dir, set_seed
from gaze_gz.zones import MultiScaleZoneLabeler
# training dataset：gaze360 testing dataset:MPIIGaze

def build_jitter(cfg: dict):
    try:
        from torchvision.transforms import ColorJitter, Compose, Resize, ToTensor, Grayscale

        def tensor_transform(image):
            return Compose([
                Grayscale(num_output_channels=3),
                Resize((cfg["data"]["image_size"], cfg["data"]["image_size"])), 
                ToTensor()
            ])(image)

        jitter = ColorJitter(
            brightness=cfg["augment"]["brightness"],
            contrast=cfg["augment"]["contrast"],
            saturation=cfg["augment"]["saturation"],
            hue=cfg["augment"]["hue"],
        )
        return tensor_transform, jitter
    except Exception:
        print("error")
        from gaze_gz.dataset import _default_to_tensor

        def tensor_transform(image):
            return _default_to_tensor(image, cfg["data"]["image_size"])

        class IdentityJitter:
            def __call__(self, image_tensor):
                noise = torch.randn_like(image_tensor) * 0.02
                return (image_tensor + noise).clamp(0.0, 1.0)

        return tensor_transform, IdentityJitter()


def resolve_device(cfg: dict) -> torch.device:
    if cfg["device"] == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg["device"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpus", type=str, default=None, help="GPU IDs, e.g., '0,2,3'")
    args = parser.parse_args()

    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        print(f"Using GPUs: {args.gpus}")

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = resolve_device(cfg)

    train_root = ROOT / cfg["data"]["train_root"]
    test_root = ROOT / cfg["data"]["test_root"]
    out_dir = ensure_dir(ROOT / cfg["output"]["dir"])

    train_gaze = load_gaze_array(train_root)
    test_gaze = load_gaze_array(test_root)

    labeler = MultiScaleZoneLabeler(
        scales=cfg["zones"]["scales"],
        kmeans_iters=cfg["zones"]["kmeans_iters"],
        seed=cfg["seed"],
    )
    labeler.fit(train_gaze)
    train_zone_labels = labeler.encode(train_gaze)
    test_zone_labels = labeler.encode(test_gaze)

    tensor_transform, jitter = build_jitter(cfg)

    train_set = GazeDataset(train_root, cfg["data"]["image_size"], train_zone_labels, transform=tensor_transform)
    test_set = GazeDataset(test_root, cfg["data"]["image_size"], test_zone_labels, transform=tensor_transform)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )

    model = GazeGZModel(
        embedding_dim=cfg["model"]["embedding_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        zone_scales=cfg["zones"]["scales"],
        use_resnet50=cfg["model"]["use_resnet50"],
    ).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
        betas=(0.9, 0.95),
    )

    best_metric = float("inf")
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, jitter, cfg["loss"], epoch)
        val_metrics = evaluate_model(model, test_loader, device, epoch)
        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_ang={train_metrics['angular_error_deg']:.2f}° "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_ang={val_metrics['angular_error_deg']:.2f}°"
        )
        if val_metrics["angular_error_deg"] < best_metric:
            best_metric = val_metrics["angular_error_deg"]
            save_checkpoint(
                out_dir / "best.pt",
                model,
                optimizer,
                cfg,
                labeler.state_dict(),
                {"epoch": epoch, **val_metrics},
            )

    print(f"best_val_ang={best_metric:.2f}°")


if __name__ == "__main__":
    main()
