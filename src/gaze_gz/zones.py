from __future__ import annotations

import numpy as np


def _pairwise_sqdist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=2)


def simple_kmeans(points: np.ndarray, k: int, iters: int = 25, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if len(points) < k:
        reps = int(np.ceil(k / len(points)))
        points = np.tile(points, (reps, 1))[:k]
    init_ids = rng.choice(len(points), size=k, replace=False)
    centers = points[init_ids].copy()
    for _ in range(iters):
        dist = _pairwise_sqdist(points, centers)
        labels = dist.argmin(axis=1)
        new_centers = []
        for idx in range(k):
            mask = labels == idx
            if mask.any():
                new_centers.append(points[mask].mean(axis=0))
            else:
                new_centers.append(points[rng.integers(0, len(points))])
        new_centers = np.stack(new_centers, axis=0)
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return centers


class MultiScaleZoneLabeler:
    def __init__(self, scales: list[int], kmeans_iters: int = 25, seed: int = 42):
        self.scales = scales
        self.kmeans_iters = kmeans_iters
        self.seed = seed
        self.centers_by_scale: dict[int, np.ndarray] = {}

    def fit(self, gaze_labels: np.ndarray) -> None:
        for scale in self.scales:
            self.centers_by_scale[scale] = simple_kmeans(
                gaze_labels, k=scale, iters=self.kmeans_iters, seed=self.seed + scale
            )

    def encode(self, gaze_labels: np.ndarray) -> dict[int, np.ndarray]:
        outputs = {}
        for scale, centers in self.centers_by_scale.items():
            dist = _pairwise_sqdist(gaze_labels, centers)
            outputs[scale] = dist.argmin(axis=1).astype(np.int64)
        return outputs

    def state_dict(self) -> dict:
        return {
            "scales": self.scales,
            "kmeans_iters": self.kmeans_iters,
            "seed": self.seed,
            "centers_by_scale": {str(k): v.tolist() for k, v in self.centers_by_scale.items()},
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "MultiScaleZoneLabeler":
        obj = cls(state["scales"], state["kmeans_iters"], state["seed"])
        obj.centers_by_scale = {int(k): np.asarray(v, dtype=np.float32) for k, v in state["centers_by_scale"].items()}
        return obj
