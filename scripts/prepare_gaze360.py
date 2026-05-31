"""
Gaze360 数据集预处理脚本

Gaze360 数据集结构:
Gaze360/
├── images/
│   ├── person_000/
│   │   ├── video_000/
│   │   │   ├── frame_000.jpg
│   │   │   ├── frame_001.jpg
│   │   │   └── ...
│   │   └── ...
│   └── ...
└── label.label

输出格式:
data/gaze360_train/
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
    
    Args:
        vec: (3,) 单位向量 [x, y, z]
    
    Returns:
        (pitch, yaw) 弧度
    """
    x, y, z = vec
    pitch = math.asin(-y)
    yaw = math.atan2(-x, -z)
    return pitch, yaw


def angles_to_vector(pitch: float, yaw: float) -> np.ndarray:
    """
    pitch 和 yaw 角度转换为 3D 向量
    """
    x = -math.cos(pitch) * math.sin(yaw)
    y = -math.sin(pitch)
    z = -math.cos(pitch) * math.cos(yaw)
    return np.array([x, y, z], dtype=np.float32)


def parse_gaze360_label(label_path: Path) -> list[dict[str, Any]]:
    """
    解析 Gaze360 的 label.label 文件
    
    格式: 每行包含
    frame_path, x3d, y3d, z3d, x2d, y2d, ...
    """
    samples = []
    
    with open(label_path, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        
        frame_path = parts[0]
        x3d, y3d, z3d = float(parts[1]), float(parts[2]), float(parts[3])
        
        gaze_vec = np.array([x3d, y3d, z3d], dtype=np.float32)
        norm = np.linalg.norm(gaze_vec)
        if norm > 0:
            gaze_vec = gaze_vec / norm
        
        pitch, yaw = vector_to_angles(gaze_vec)
        
        samples.append({
            "frame_path": frame_path,
            "pitch": pitch,
            "yaw": yaw,
            "gaze_vec": gaze_vec.tolist(),
        })
    
    return samples


def detect_and_crop_face(image: np.ndarray, target_size: int = 224) -> np.ndarray | None:
    """
    检测人脸并裁剪
    
    简化版本: 假设图像已经是人脸，直接 resize
    完整版本应该使用人脸检测器
    """
    h, w = image.shape[:2]
    size = min(h, w)
    
    start_y = (h - size) // 2
    start_x = (w - size) // 2
    
    cropped = image[start_y:start_y+size, start_x:start_x+size]
    resized = cv2.resize(cropped, (target_size, target_size))
    
    return resized


def process_gaze360(
    gaze360_root: Path,
    output_dir: Path,
    split: str = "train",
    max_samples: int | None = None,
    subsample: int = 1,
):
    """
    处理 Gaze360 数据集
    
    Args:
        gaze360_root: Gaze360 数据集根目录
        output_dir: 输出目录
        split: train 或 test
        max_samples: 最大样本数 (用于测试)
        subsample: 子采样间隔 (每隔几帧取一帧)
    """
    output_dir = Path(output_dir)
    output_images = output_dir / "images"
    output_images.mkdir(parents=True, exist_ok=True)
    
    label_path = gaze360_root / "label.label"
    if not label_path.exists():
        raise FileNotFoundError(f"找不到标签文件: {label_path}")
    
    print(f"解析标签文件: {label_path}")
    all_samples = parse_gaze360_label(label_path)
    print(f"共找到 {len(all_samples)} 个标注")
    
    if split == "train":
        samples = all_samples[::2]
    else:
        samples = all_samples[1::2]
    
    if max_samples:
        samples = samples[:max_samples]
    
    annotations = []
    count = 0
    errors = 0
    
    for i, sample in enumerate(samples):
        if i % subsample != 0:
            continue
        
        frame_path = gaze360_root / sample["frame_path"]
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
            
            output_name = f"{count:06d}.jpg"
            output_path = output_images / output_name
            cv2.imwrite(str(output_path), cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
            
            annotations.append({
                "image": f"images/{output_name}",
                "pitch": sample["pitch"],
                "yaw": sample["yaw"],
            })
            
            count += 1
            if count % 100 == 0:
                print(f"已处理 {count} 个样本...")
            
            if max_samples and count >= max_samples:
                break
                
        except Exception as e:
            errors += 1
            continue
    
    annotations_path = output_dir / "annotations.json"
    with open(annotations_path, "w") as f:
        json.dump(annotations, f, indent=2)
    
    print(f"\n处理完成!")
    print(f"  成功: {count}")
    print(f"  错误: {errors}")
    print(f"  输出目录: {output_dir}")
    print(f"  标注文件: {annotations_path}")


def main():
    parser = argparse.ArgumentParser(description="Gaze360 数据集预处理")
    parser.add_argument("--gaze360_root", type=str, required=True, help="Gaze360 数据集根目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--max_samples", type=int, default=None, help="最大样本数")
    parser.add_argument("--subsample", type=int, default=5, help="子采样间隔")
    
    args = parser.parse_args()
    
    process_gaze360(
        gaze360_root=Path(args.gaze360_root),
        output_dir=Path(args.output_dir),
        split=args.split,
        max_samples=args.max_samples,
        subsample=args.subsample,
    )


if __name__ == "__main__":
    main()
