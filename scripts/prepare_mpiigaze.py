from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

try:
    import scipy.io as sio
except ImportError as exc:
    raise ImportError("scipy is required: pip install scipy") from exc


TRAIN_PERSONS = ["p00", "p01", "p02", "p03", "p04", "p05"]
TEST_PERSONS = ["p06", "p07", "p08", "p09", "p10", "p11", "p12", "p13", "p14"]
EYE_NAMES = ("left", "right")


def vector_to_angles(vec: np.ndarray) -> tuple[float, float]:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
    pitch = math.asin(-y)
    yaw = math.atan2(-x, -z)
    return pitch, yaw


def unwrap(value):
    while isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        value = value.reshape(-1)[0]
    return value


def get_field(obj, name: str):
    obj = unwrap(obj)
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, np.void) and obj.dtype.names and name in obj.dtype.names:
        return obj[name]
    if isinstance(obj, np.ndarray) and obj.dtype.names and name in obj.dtype.names:
        return obj[name]
    return None


def to_image_list(value) -> list[np.ndarray]:
    value = unwrap(value)
    if value is None:
        return []

    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return [np.asarray(unwrap(item)) for item in value.reshape(-1)]

        if value.ndim == 2:
            return [value]

        if value.ndim == 3:
            if value.shape[-1] in (1, 3, 4):
                return [value]
            return [value[i] for i in range(value.shape[0])]

        if value.ndim == 4:
            return [value[i] for i in range(value.shape[0])]

    return [np.asarray(value)]


def to_gaze_list(value) -> list[np.ndarray]:
    value = unwrap(value)
    if value is None:
        return []

    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return [np.asarray(unwrap(item)).reshape(-1) for item in value.reshape(-1)]

        if value.ndim == 1:
            return [value.reshape(-1)]

        if value.ndim == 2:
            return [value[i].reshape(-1) for i in range(value.shape[0])]

    return [np.asarray(value).reshape(-1)]


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)

    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255.0).clip(0, 255).astype(np.uint8)
        else:
            image = image.clip(0, 255).astype(np.uint8)

    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))

    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[:, :, 0]

    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[:, :, :3]

    return image


def save_image(image: np.ndarray, output_path: Path) -> None:
    image = normalize_image(image)
    if image.ndim == 2:
        Image.fromarray(image, mode="L").save(output_path)
    else:
        Image.fromarray(image, mode="RGB").save(output_path)


def find_normalized_root(mpiigaze_root: Path) -> Path:
    candidates = [
        mpiigaze_root / "Data" / "Normalized",
        mpiigaze_root / "Normalized",
        mpiigaze_root,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find normalized MPIIGaze directory under {mpiigaze_root}")


def collect_mat_files(normalized_root: Path, persons: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for person in persons:
        person_dir = normalized_root / person
        if person_dir.is_dir():
            files.extend(sorted(person_dir.glob("*.mat")))
            continue

        single_mat = normalized_root / f"{person}.mat"
        if single_mat.exists():
            files.append(single_mat)
    return files


def load_day_struct(mat_path: Path):
    mat_data = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)

    for key in ("data", "Data"):
        if key in mat_data:
            return unwrap(mat_data[key])

    if "left" in mat_data or "right" in mat_data:
        return mat_data

    raise KeyError(f"Unsupported mat structure: {mat_path}")


def extract_eye_samples(day_struct, eye_name: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    eye_struct = get_field(day_struct, eye_name)
    if eye_struct is None:
        return [], []

    images = to_image_list(get_field(eye_struct, "image"))
    gazes = to_gaze_list(get_field(eye_struct, "gaze"))

    if not images or not gazes:
        return [], []

    usable = min(len(images), len(gazes))
    return images[:usable], gazes[:usable]


def process_normalized_mat(
    mat_path: Path,
    output_dir: Path,
    start_idx: int,
    max_samples: int | None = None,
) -> tuple[list[dict], int, int]:
    output_images = output_dir / "images"
    output_images.mkdir(parents=True, exist_ok=True)

    annotations: list[dict] = []
    saved = 0
    errors = 0

    try:
        day_struct = load_day_struct(mat_path)
    except Exception as exc:
        print(f"  failed to load {mat_path.name}: {exc}")
        return annotations, saved, errors + 1

    person_id = mat_path.parent.name
    day_id = mat_path.stem

    for eye_name in EYE_NAMES:
        images, gazes = extract_eye_samples(day_struct, eye_name)
        if not images:
            continue

        for local_index, (image, gaze) in enumerate(zip(images, gazes)):
            if max_samples is not None and start_idx + saved >= max_samples:
                return annotations, saved, errors

            try:
                if gaze.size < 3:
                    errors += 1
                    continue

                pitch, yaw = vector_to_angles(gaze[:3])
                file_name = f"{start_idx + saved:06d}.png"
                save_image(image, output_images / file_name)
                annotations.append(
                    {
                        "image": f"images/{file_name}",
                        "pitch": pitch,
                        "yaw": yaw,
                        "person": person_id,
                        "day": day_id,
                        "eye": eye_name,
                        "source_mat": mat_path.name,
                        "source_index": local_index,
                    }
                )
                saved += 1
            except Exception:
                errors += 1

    if saved == 0:
        print(f"  warning: no usable samples found in {mat_path.name}")

    return annotations, saved, errors


def process_mpiigaze_normalized(
    mpiigaze_root: Path,
    output_dir: Path,
    split: str,
    max_samples: int | None = None,
    train_persons: list[str] | None = None,
    test_persons: list[str] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if split == "train":
        persons = train_persons or TRAIN_PERSONS
    else:
        persons = test_persons or TEST_PERSONS
    normalized_root = find_normalized_root(mpiigaze_root)
    mat_files = collect_mat_files(normalized_root, persons)

    print(f"Preparing MPIIGaze normalized split: {split}")
    print(f"Normalized root: {normalized_root}")
    print(f"Persons: {persons}")
    print(f"Mat files: {len(mat_files)}")

    all_annotations: list[dict] = []
    total_saved = 0
    total_errors = 0

    for mat_file in mat_files:
        if max_samples is not None and total_saved >= max_samples:
            break

        print(f"\nProcessing {mat_file.relative_to(normalized_root)}")
        annotations, saved, errors = process_normalized_mat(
            mat_path=mat_file,
            output_dir=output_dir,
            start_idx=total_saved,
            max_samples=max_samples,
        )
        all_annotations.extend(annotations)
        total_saved += saved
        total_errors += errors
        print(f"  saved: {saved}, errors: {errors}")

    with open(output_dir / "annotations.json", "w", encoding="utf-8") as f:
        json.dump(all_annotations, f, indent=2)

    print("\nDone")
    print(f"  total saved: {total_saved}")
    print(f"  total errors: {total_errors}")
    print(f"  output dir: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MPIIGaze normalized data")
    parser.add_argument("--mpiigaze_root", type=str, required=True, help="MPIIGaze root directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="train uses p00-p05, test uses p06-p14")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap on saved samples")
    parser.add_argument("--train_persons", type=str, nargs="+", default=None, help="Optional custom train persons")
    parser.add_argument("--test_persons", type=str, nargs="+", default=None, help="Optional custom test persons")
    args = parser.parse_args()

    process_mpiigaze_normalized(
        mpiigaze_root=Path(args.mpiigaze_root),
        output_dir=Path(args.output_dir),
        split=args.split,
        max_samples=args.max_samples,
        train_persons=args.train_persons,
        test_persons=args.test_persons,
    )


if __name__ == "__main__":
    main()
