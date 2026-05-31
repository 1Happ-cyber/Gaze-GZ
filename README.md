# Gaze-GZ Minimal Reproduction

This project is a minimal PyTorch reproduction of the paper:

`Gaze-GZ: Generalized Gaze Estimation with Multi-scale Gaze Zone Prediction`

It keeps the main ideas from the paper:

- gaze regression from face images
- feature-consistency branch with `ColorJitter`
- multi-scale gaze zone classification at scales `4/9/16/25`
- triplet loss on shared embeddings

This is a compact engineering reproduction, not a claim of exact benchmark parity with the paper. The original paper uses large real datasets such as ETH-XGaze, Gaze360, MPIIGaze, and EyeDiap. In this folder, a toy train/test dataset generator is included so the full pipeline can run locally.

## Folder layout

- `configs/minimal.yaml`: default config
- `src/gaze_gz/`: model, losses, dataset, zone labeling, training helpers
- `scripts/create_toy_dataset.py`: creates a tiny train/test dataset
- `train.py`: training entrypoint
- `evaluate.py`: evaluation entrypoint

## Data format

Each split contains:

- `images/`: RGB face images
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
python scripts/create_toy_dataset.py
```

3. Train:

```bash
python train.py --config configs/minimal.yaml
```

4. Evaluate:

```bash
python evaluate.py --config configs/minimal.yaml --checkpoint outputs/best.pt
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

### Data Preparation

#### Option 1: One-click preparation

```bash
# ETH-XGaze -> MPIIGaze (推荐)
python scripts/prepare_data.py \
    --train_dataset ethxgaze \
    --test_dataset mpiigaze \
    --ethxgaze_root /path/to/ETH-XGaze \
    --mpiigaze_root /path/to/MPIIGaze

# Gaze360 -> MPIIGaze
python scripts/prepare_data.py \
    --train_dataset gaze360 \
    --test_dataset mpiigaze \
    --gaze360_root /path/to/Gaze360 \
    --mpiigaze_root /path/to/MPIIGaze
```

#### Option 2: Step-by-step preparation

**Prepare ETH-XGaze (Training):**

```bash
python scripts/prepare_ethxgaze.py \
    --ethxgaze_root /path/to/ETH-XGaze \
    --output_dir data/ethxgaze_train \
    --split train \
    --subsample 5
```

**Prepare Gaze360 (Training):**

```bash
python scripts/prepare_gaze360.py \
    --gaze360_root /path/to/Gaze360 \
    --output_dir data/gaze360_train \
    --split train \
    --subsample 5
```

**Prepare MPIIGaze (Testing):**

```bash
python scripts/prepare_mpiigaze.py \
    --mpiigaze_root /path/to/MPIIGaze \
    --output_dir data/mpiigaze_test \
    --test_persons p00 p01
```

### Training with Real Datasets

After preparing the data, use the appropriate config file:

```bash
python train.py --config configs/gaze360_mpiigaze.yaml
```

### Dataset Download Links

- **ETH-XGaze**: [https://ait.ethz.ch/projects/2020/ETH-XGaze](https://ait.ethz.ch/projects/2020/ETH-XGaze)
- **Gaze360**: [http://gaze360.csail.mit.edu/](http://gaze360.csail.mit.edu/)
- **MPIIGaze**: [https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation/](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation/)

## Notes

- The paper reports `224 x 244` image size. This reproduction uses `224 x 224` by default to keep preprocessing simple.
- The paper uses ResNet-50. This repo uses ResNet-50 when `torchvision` is available, and falls back to a small CNN encoder otherwise.
- The paper describes KMeans + KNN to derive zone labels. This reproduction uses KMeans on training gaze labels and nearest-center assignment for all splits.
- For cross-domain evaluation (Gaze360 → MPIIGaze), the domain gap is significant. Consider using domain adaptation techniques for better performance.
