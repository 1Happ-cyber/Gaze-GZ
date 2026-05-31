# Gaze-GZ Minimal and Personal Reproduction

This project is a minimal PyTorch reproduction of the paper:

`Gaze-GZ: Generalized Gaze Estimation with Multi-scale Gaze Zone Prediction`

It keeps the main ideas from the paper:

- gaze regression from face images
- feature-consistency branch with `ColorJitter`
- multi-scale gaze zone classification at scales `4/9/16/25`
- triplet loss on shared embeddings

This is a compact engineering reproduction, not a claim of exact benchmark parity with the paper. The original paper uses large real datasets such as ETH-XGaze, Gaze360, MPIIGaze, and EyeDiap. In this folder, a small train/test dataset generator is included so the full pipeline can run locally.

## Folder layout

- `configs/mpiigaze_mpiigaze.yaml`: default config
- `src/gaze_gz/`: model, losses, dataset, zone labeling, training helpers
- `train.py`: training entrypoint
- `evaluate.py`: evaluation entrypoint

## Data format

Each split contains:

- `images/`: face images (here actually is grey)
- `annotations.json`: list of samples

Sample annotation:

```json
{
  "image": "images/sample_000.png",
  "pitch": -0.15,
  "yaw": 0.32
}
```

Angles are in radians.

## Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create the toy dataset:

```bash
sh domain.sh
```

3. Train:

```bash
sh train.sh
```

4. Evaluate:

```bash
python evaluate.py --config configs/mpiigaze_mpiigaze.yaml --checkpoint outputs/best.pt
```

## Mapping from paper to code

- `L_ori`: `l1_loss` on pitch/yaw regression
- `L_con`: disturbed-image gaze loss + MMD feature alignment
- `L_zone`: average CE over `4/9/16/25` zone classifiers
- `L_t`: triplet loss on original and consistency embeddings

## Using Real Datasets

This reproduction supports real gaze estimation datasets like **Gaze360** and **MPIIGaze**.

### Supported Datasets

| Dataset | Description | Typical Use |
|---------|-------------|-------------|
| ETH-XGaze | Large-scale high-quality dataset | Training (推荐) |
| Gaze360 | Large-scale dataset with 360° gaze | Training |
| MPIIGaze | In-the-wild dataset from MPII | Testing/Evaluation |
| Toy | Synthetic dataset for testing code | Development |

### Training with Real Datasets

After preparing the data, use the appropriate config file:

```bash
sh train.sh
```

## Notes

- The paper reports `224 x 244` image size. This reproduction uses `224 x 224` by default to keep preprocessing simple.
- The paper uses ResNet-50. This repo uses ResNet-50 when `torchvision` is available, and falls back to a small CNN encoder otherwise.
- The paper describes KMeans + KNN to derive zone labels. This reproduction uses KMeans on training gaze labels and nearest-center assignment for all splits.
