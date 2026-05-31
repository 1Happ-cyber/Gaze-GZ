from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .utils import load_json


def _default_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    image = image.convert("RGB").resize((image_size, image_size))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


class GazeDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        image_size: int,
        zone_labels: dict[int, np.ndarray],
        transform=None,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.transform = transform
        self.samples = load_json(self.root / "annotations.json")
        self.zone_labels = zone_labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = Image.open(self.root / sample["image"])
        if self.transform is not None:
            tensor = self.transform(image)
        else:
            tensor = _default_to_tensor(image, self.image_size)
        gaze = torch.tensor([sample["pitch"], sample["yaw"]], dtype=torch.float32)
        zones = {str(scale): torch.tensor(labels[idx], dtype=torch.long) for scale, labels in self.zone_labels.items()}
        return {
            "image": tensor,
            "gaze": gaze,
            "zones": zones,
        }


def load_gaze_array(root: str | Path) -> np.ndarray:
    samples = load_json(Path(root) / "annotations.json")
    return np.asarray([[row["pitch"], row["yaw"]] for row in samples], dtype=np.float32)
