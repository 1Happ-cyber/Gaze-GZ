from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str]) -> bool:
    print(" ".join(cmd))
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def prepare_gaze360(gaze360_root: Path, output_dir: Path, split: str, max_samples: int | None) -> bool:
    cmd = [
        sys.executable,
        "scripts/prepare_gaze360.py",
        "--gaze360_root",
        str(gaze360_root),
        "--output_dir",
        str(output_dir),
        "--split",
        split,
        "--subsample",
        "5",
    ]
    if max_samples:
        cmd.extend(["--max_samples", str(max_samples)])
    return run_command(cmd)


def prepare_mpiigaze(mpiigaze_root: Path, output_dir: Path, split: str, max_samples: int | None) -> bool:
    cmd = [
        sys.executable,
        "scripts/prepare_mpiigaze.py",
        "--mpiigaze_root",
        str(mpiigaze_root),
        "--output_dir",
        str(output_dir),
        "--split",
        split,
    ]
    if max_samples:
        cmd.extend(["--max_samples", str(max_samples)])
    return run_command(cmd)


def prepare_ethxgaze(
    ethxgaze_root: Path,
    output_dir: Path,
    split: str,
    max_samples: int | None,
    subsample: int = 5,
) -> bool:
    cmd = [
        sys.executable,
        "scripts/prepare_ethxgaze.py",
        "--ethxgaze_root",
        str(ethxgaze_root),
        "--output_dir",
        str(output_dir),
        "--split",
        split,
        "--subsample",
        str(subsample),
    ]
    if max_samples:
        cmd.extend(["--max_samples", str(max_samples)])
    return run_command(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare gaze datasets")
    parser.add_argument("--train_dataset", choices=["gaze360", "mpiigaze", "ethxgaze", "toy"], default="mpiigaze")
    parser.add_argument("--test_dataset", choices=["gaze360", "mpiigaze", "ethxgaze", "toy"], default="mpiigaze")
    parser.add_argument("--gaze360_root", type=str, default=None)
    parser.add_argument("--mpiigaze_root", type=str, default=None)
    parser.add_argument("--ethxgaze_root", type=str, default=None)
    parser.add_argument("--train_max_samples", type=int, default=None)
    parser.add_argument("--test_max_samples", type=int, default=None)
    parser.add_argument("--subsample", type=int, default=5)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent

    print("=" * 60)
    print("Gaze-GZ data preparation")
    print("=" * 60)
    print(f"train dataset: {args.train_dataset}")
    print(f"test dataset: {args.test_dataset}")
    if args.train_dataset == "mpiigaze" or args.test_dataset == "mpiigaze":
        print("MPIIGaze format: normalized only")
        print("MPIIGaze split: train p00-p05, test p06-p14")
    print()

    success = True

    if args.train_dataset == "gaze360":
        if not args.gaze360_root:
            raise ValueError("--gaze360_root is required")
        success &= prepare_gaze360(Path(args.gaze360_root), root / "data" / "gaze360_train", "train", args.train_max_samples)
    elif args.train_dataset == "mpiigaze":
        if not args.mpiigaze_root:
            raise ValueError("--mpiigaze_root is required")
        success &= prepare_mpiigaze(Path(args.mpiigaze_root), root / "data" / "mpiigaze_train", "train", args.train_max_samples)
    elif args.train_dataset == "ethxgaze":
        if not args.ethxgaze_root:
            raise ValueError("--ethxgaze_root is required")
        success &= prepare_ethxgaze(Path(args.ethxgaze_root), root / "data" / "ethxgaze_train", "train", args.train_max_samples, args.subsample)
    else:
        print("Using existing toy train split")

    print()

    if args.test_dataset == "gaze360":
        if not args.gaze360_root:
            raise ValueError("--gaze360_root is required")
        success &= prepare_gaze360(Path(args.gaze360_root), root / "data" / "gaze360_test", "test", args.test_max_samples)
    elif args.test_dataset == "mpiigaze":
        if not args.mpiigaze_root:
            raise ValueError("--mpiigaze_root is required")
        success &= prepare_mpiigaze(Path(args.mpiigaze_root), root / "data" / "mpiigaze_test", "test", args.test_max_samples)
    elif args.test_dataset == "ethxgaze":
        if not args.ethxgaze_root:
            raise ValueError("--ethxgaze_root is required")
        success &= prepare_ethxgaze(Path(args.ethxgaze_root), root / "data" / "ethxgaze_test", "test", args.test_max_samples, args.subsample)
    else:
        print("Using existing toy test split")

    print()
    print("=" * 60)
    if success:
        print("Data preparation finished")
        print(f"Next: python train.py --config configs/{args.train_dataset}_{args.test_dataset}.yaml")
    else:
        print("Data preparation failed")
    print("=" * 60)


if __name__ == "__main__":
    main()
