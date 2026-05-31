"""
ETH-XGaze 数据集预处理脚本

ETH-XGaze 数据集结构:
ETH-XGaze/
├── data/
│   ├── participant000/
│   │   ├── session000/
│   │   │   ├── frame_000.jpg
│   │   │   ├── frame_001.jpg
│   │   │   └── ...
│   │   └── label.txt
│   ├── participant001/
│   └── ...
└── ...

输出格式:
data/ethxgaze_train/
├── images/
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
└── annotations.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:
    raise ImportError("请安装 opencv-python: pip install opencv-python")


def vector_to_angles(vec: np.ndarray) -> tuple[float, float]:
    """
    3D 注视向量转换为 pitch 和 yaw 角度
    """
    x, y, z = vec
    pitch = math.asin(-y)
    yaw = math.atan2(-x, -z)
    return pitch, yaw


def parse_ethxgaze_label(label_path: Path) -> list[dict[str, Any]]:
    """
    解析 ETH-XGaze 的 label.txt 文件
    
    格式通常是:
    frame_id, gaze_x, gaze_y, gaze_z, head_x, head_y, head_z, ...
    或者
    frame_id, pitch, yaw, ...
    """
    samples = []
    
    with open(label_path, "r") as f:
        lines = f.readlines()
    
    header = lines[0].strip().lower() if lines else ""
    has_header = "frame" in header or "gaze" in header
    
    for line in lines[has_header:]:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        
        try:
            frame_id = parts[0]
            
            if len(parts) >= 4:
                try:
                    gaze_x = float(parts[1])
                    gaze_y = float(parts[2])
                    gaze_z = float(parts[3])
                    
                    gaze_vec = np.array([gaze_x, gaze_y, gaze_z], dtype=np.float32)
                    norm = np.linalg.norm(gaze_vec)
                    if norm > 0:
                        gaze_vec = gaze_vec / norm
                    
                    pitch, yaw = vector_to_angles(gaze_vec)
                except ValueError:
                    pitch = float(parts[1])
                    yaw = float(parts[2])
            else:
                pitch = float(parts[1])
                yaw = float(parts[2])
            
            samples.append({
                "frame_id": frame_id,
                "pitch": pitch,
                "yaw": yaw,
            })
        except (ValueError, IndexError):
            continue
    
    return samples


def detect_and_crop_face(image: np.ndarray, target_size: int = 224) -> np.ndarray:
    """
    检测人脸并裁剪
    
    ETH-XGaze 图像通常已经是人脸区域，直接 resize
    """
    h, w = image.shape[:2]
    size = min(h, w)
    
    start_y = (h - size) // 2
    start_x = (w - size) // 2
    
    cropped = image[start_y:start_y+size, start_x:start_x+size]
    resized = cv2.resize(cropped, (target_size, target_size))
    
    return resized


def process_ethxgaze_participant(
    participant_dir: Path,
    output_dir: Path,
    start_idx: int,
    max_samples: int | None = None,
    subsample: int = 1,
) -> tuple[int, int, int]:
    """
    处理单个被试的数据
    
    Returns:
        (成功数量, 错误数量, 新的起始索引)
    """
    output_images = output_dir / "images"
    output_images.mkdir(parents=True, exist_ok=True)
    
    annotations = []
    count = 0
    errors = 0
    
    session_dirs = sorted([d for d in participant_dir.iterdir() if d.is_dir()])
    
    for session_dir in session_dirs:
        if max_samples and start_idx + count >= max_samples:
            break
        
        label_path = session_dir / "label.txt"
        if not label_path.exists():
            label_path = participant_dir / "label.txt"
        
        if not label_path.exists():
            continue
        
        samples = parse_ethxgaze_label(label_path)
        
        for i, sample in enumerate(samples):
            if max_samples and start_idx + count >= max_samples:
                break
            
            if i % subsample != 0:
                continue
            
            frame_path = session_dir / f"{sample['frame_id']}.jpg"
            if not frame_path.exists():
                frame_path = session_dir / f"frame_{int(sample['frame_id']):06d}.jpg"
            
            if not frame_path.exists():
                frame_path = session_dir / sample['frame_id']
            
            if not frame_path.exists():
                errors += 1
                continue
            
            try:
                image = cv2.imread(str(frame_path))
                if image is None:
                    errors += 1
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cropped = detect_and_crop_face(image)
                
                output_name = f"{start_idx + count:06d}.jpg"
                output_path = output_images / output_name
                cv2.imwrite(str(output_path), cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
                
                annotations.append({
                    "image": f"images/{output_name}",
                    "pitch": sample["pitch"],
                    "yaw": sample["yaw"],
                })
                
                count += 1
                
            except Exception:
                errors += 1
                continue
    
    return count, errors, start_idx + count


def process_ethxgaze(
    ethxgaze_root: Path,
    output_dir: Path,
    split: str = "train",
    max_samples: int | None = None,
    subsample: int = 1,
    train_ratio: float = 0.8,
):
    """
    处理 ETH-XGaze 数据集
    
    Args:
        ethxgaze_root: ETH-XGaze 数据集根目录
        output_dir: 输出目录
        split: train 或 test
        max_samples: 最大样本数
        subsample: 子采样间隔
        train_ratio: 训练集比例
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = ethxgaze_root / "data"
    if not data_dir.exists():
        data_dir = ethxgaze_root
    
    participant_dirs = sorted([
        d for d in data_dir.iterdir() 
        if d.is_dir() and (d.name.startswith("participant") or d.name.startswith("p"))
    ])
    
    n_participants = len(participant_dirs)
    n_train = int(n_participants * train_ratio)
    
    if split == "train":
        selected_dirs = participant_dirs[:n_train]
    else:
        selected_dirs = participant_dirs[n_train:]
    
    print(f"共 {n_participants} 个被试, {split} 集使用 {len(selected_dirs)} 个")
    
    all_annotations = []
    total_count = 0
    total_errors = 0
    
    for participant_dir in selected_dirs:
        if max_samples and total_count >= max_samples:
            break
        
        print(f"处理 {participant_dir.name}...")
        
        count, errors, _ = process_ethxgaze_participant(
            participant_dir=participant_dir,
            output_dir=output_dir,
            start_idx=total_count,
            max_samples=max_samples,
            subsample=subsample,
        )
        
        total_count += count
        total_errors += errors
        
        print(f"  成功: {count}, 错误: {errors}")
    
    annotations_path = output_dir / "annotations.json"
    with open(annotations_path, "w") as f:
        json.dump(all_annotations, f, indent=2)
    
    print(f"\n处理完成!")
    print(f"  总成功: {total_count}")
    print(f"  总错误: {total_errors}")
    print(f"  输出目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="ETH-XGaze 数据集预处理")
    parser.add_argument("--ethxgaze_root", type=str, required=True, help="ETH-XGaze 数据集根目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--max_samples", type=int, default=None, help="最大样本数")
    parser.add_argument("--subsample", type=int, default=5, help="子采样间隔")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")
    
    args = parser.parse_args()
    
    process_ethxgaze(
        ethxgaze_root=Path(args.ethxgaze_root),
        output_dir=Path(args.output_dir),
        split=args.split,
        max_samples=args.max_samples,
        subsample=args.subsample,
        train_ratio=args.train_ratio,
    )


if __name__ == "__main__":
    main()
