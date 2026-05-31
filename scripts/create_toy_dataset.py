from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image, ImageDraw

from gaze_gz.utils import ensure_dir, save_json, set_seed


def gaze_to_image(pitch: float, yaw: float, size: int = 224) -> Image.Image:
    pitch_n = (pitch + 0.7) / 1.4
    yaw_n = (yaw + 1.2) / 2.4

    x = np.linspace(0, 1, size, dtype=np.float32)
    y = np.linspace(0, 1, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    r = np.clip(255 * (0.25 + 0.75 * yaw_n * xx), 0, 255)
    g = np.clip(255 * (0.25 + 0.75 * pitch_n * yy), 0, 255)
    b = np.clip(255 * (0.2 + 0.8 * (1.0 - np.abs(xx - yy))), 0, 255)
    arr = np.stack([r, g, b], axis=2).astype(np.uint8)

    image = Image.fromarray(arr, mode="RGB")
    draw = ImageDraw.Draw(image)
    cx = int(size * (0.2 + 0.6 * yaw_n))
    cy = int(size * (0.2 + 0.6 * pitch_n))
    radius = max(8, size // 18)
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=(255, 255, 255), width=3)
    return image


def build_split(root: Path, num_samples: int, pitch_range: tuple[float, float], yaw_range: tuple[float, float]):
    image_dir = ensure_dir(root / "images")
    samples = []
    for idx in range(num_samples):
        pitch = float(np.random.uniform(*pitch_range))
        yaw = float(np.random.uniform(*yaw_range))
        image = gaze_to_image(pitch, yaw)
        name = f"sample_{idx:03d}.png"
        image.save(image_dir / name)
        samples.append({"image": f"images/{name}", "pitch": pitch, "yaw": yaw})
    save_json(samples, root / "annotations.json")


def main():
    set_seed(42)
    data_root = ROOT / "data"
    train_root = ensure_dir(data_root / "toy_train")
    test_root = ensure_dir(data_root / "toy_test")

    build_split(train_root, num_samples=48, pitch_range=(-0.45, 0.45), yaw_range=(-0.9, 0.9))
    build_split(test_root, num_samples=16, pitch_range=(-0.35, 0.35), yaw_range=(-0.7, 0.7))
    print(f"toy dataset written to: {data_root}")


if __name__ == "__main__":
    main()
