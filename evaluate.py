from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.utils.data import DataLoader

from gaze_gz.config import load_config
from gaze_gz.dataset import GazeDataset, load_gaze_array
from gaze_gz.model import GazeGZModel
from gaze_gz.trainer import evaluate_model
from gaze_gz.zones import MultiScaleZoneLabeler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_root = ROOT / cfg["data"]["test_root"]
    ckpt = torch.load(args.checkpoint, map_location=device)
    labeler = MultiScaleZoneLabeler.from_state_dict(ckpt["labeler"])
    test_zone_labels = labeler.encode(load_gaze_array(test_root))

    try:
        from torchvision.transforms import Compose, Resize, ToTensor

        transform = Compose([Resize((cfg["data"]["image_size"], cfg["data"]["image_size"])), ToTensor()])
    except Exception:
        from gaze_gz.dataset import _default_to_tensor

        def transform(image):
            return _default_to_tensor(image, cfg["data"]["image_size"])

    test_set = GazeDataset(test_root, cfg["data"]["image_size"], test_zone_labels, transform=transform)
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
    model.load_state_dict(ckpt["model"])

    metrics = evaluate_model(model, test_loader, device)
    print(f"test_loss={metrics['loss']:.4f} test_ang={metrics['angular_error_deg']:.2f}")


if __name__ == "__main__":
    main()
