# BraTS 2023 — Fast-CWDM MRI Synthesis

Fast Conditional Wavelet Diffusion Model (Fast-CWDM) for synthesizing missing MRI modalities from BraTS 2023 brain tumor data, deployed on the NRP Nautilus Kubernetes cluster.

Based on [tsereda/brats-synthesis](https://github.com/tsereda/brats-synthesis).

---

## Overview

This project trains four independent diffusion models — one per MRI modality (T1n, T1c, T2w, T2-FLAIR) — to synthesize any missing modality from the remaining three. The model operates in the 3D Haar wavelet domain using a WaveUNet architecture and is trained with a hybrid MSE + SSIM loss.

**Key design choices:**
- **Wavelet domain diffusion**: all inputs and outputs are 3D Haar wavelet subbands (8 subbands per volume), keeping spatial resolution manageable while preserving high-frequency detail
- **32-channel conditioning**: 8 target subbands + 24 conditioning subbands (3 modalities × 8)
- **Fast sampling**: 100-step sampled DDPM schedule instead of the original 1000 steps
- **Parallel training**: one pod per modality launched simultaneously on the cluster

---

## Repository Structure

```
.
├── app/
│   ├── DWT_IDWT/                   # 3D Haar wavelet transform layers
│   ├── guided_diffusion/
│   │   ├── bratsloader.py          # BraTS dataset loader (NIfTI, handles all 4 modalities)
│   │   ├── gaussian_diffusion.py   # DDPM forward/reverse process
│   │   ├── train_util.py           # Training loop, WandB logging, checkpoint saving
│   │   ├── wunet.py                # WaveUNet — 3D U-Net with wavelet up/downsampling
│   │   └── script_util.py          # Model factory / argparse defaults
│   ├── scripts/
│   │   ├── train.py                # Training entrypoint
│   │   ├── complete_dataset.py     # Evaluation: synthesize val/test set, log metrics
│   │   ├── sample.py               # Single-case sampling
│   │   └── sample_auto.py          # Batch sampling without ground truth
│   └── utils/                      # Misc evaluation utilities
├── nautilius/
│   ├── jobs/
│   │   ├── prepare-brats23.yaml    # Data extraction + 70/10/20 split job
│   │   ├── train-cwdm.yaml         # 4-pod parallel training job
│   │   └── eval-cwdm.yaml          # Evaluation job (val split)
│   ├── pods/
│   │   └── brats23pvc.yaml         # 200 Gi ReadWriteMany PVC definition
│   └── wandb-secret.yaml           # WandB API key + entity secret
├── .github/workflows/
│   └── docker-build.yml            # CI/CD: build & push to ghcr.io on push to main
├── Dockerfile
├── requirements.txt
└── run.sh                          # Container entrypoint — trains one or all modalities
```

---

## Prerequisites

- Access to the **NRP Nautilus** cluster (`gai-lina-group` namespace)
- `kubectl` configured with your NRP credentials
- A [Weights & Biases](https://wandb.ai) account
- GitHub account (`kagozi`) with write access to the repository

---

## Setup

### 1. Create the PVC

```bash
kubectl apply -f nautilius/pods/brats23pvc.yaml
```

This creates a 200 Gi `ReadWriteMany` (`rook-cephfs`) PVC named `brats23-pvc` in `gai-lina-group`.

### 2. Create the WandB secret

Edit `nautilius/wandb-secret.yaml` and fill in your WandB API key and entity (username or team name):

```yaml
stringData:
  api_key: "your-wandb-api-key"
  entity: "your-wandb-username"
```

Apply it:

```bash
kubectl apply -f nautilius/wandb-secret.yaml
```

> The secret is referenced by all training and evaluation pods via `secretKeyRef`. Never commit actual credentials.

### 3. Prepare the data

The source data lives on the `brats2025-6` PVC (read-only). The preparation job extracts the BraTS 2023 GLI+MET training cases (all have segmentation masks) and splits them 70/10/20:

| Split | Cases | Path |
|-------|-------|------|
| Train | ~1044 | `/pvc/data/brats23/train/` |
| Val   | ~148  | `/pvc/data/brats23/val/`   |
| Test  | ~297  | `/pvc/data/brats23/test/`  |

```bash
kubectl apply -f nautilius/jobs/prepare-brats23.yaml
kubectl logs -f pod/prepare-brats23 -n gai-lina-group
```

The job uses Python's `tarfile` streaming (one member at a time) to keep memory usage low and is resumable — it skips cases already extracted.

**Why training data only?** The BraTS 2023 validation set has no segmentation masks, making evaluation impossible. All 1,489 cases used here (GLI + MET) have complete ground truth.

---

## Training

### Docker image

The image is built automatically via GitHub Actions on every push to `main` that changes `Dockerfile`, `requirements.txt`, or `run.sh`:

```
ghcr.io/kagozi/brats23-cwdm:latest
```

To build locally:

```bash
docker build -t ghcr.io/kagozi/brats23-cwdm:latest .
```

### Launch parallel training (all 4 modalities)

```bash
kubectl apply -f nautilius/jobs/train-cwdm.yaml
```

This creates four pods simultaneously:

| Pod | Modality | WandB run name |
|-----|----------|----------------|
| `train-cwdm-t1n` | T1 native | `t1n_sampled_100steps` |
| `train-cwdm-t1c` | T1 contrast-enhanced | `t1c_sampled_100steps` |
| `train-cwdm-t2w` | T2-weighted | `t2w_sampled_100steps` |
| `train-cwdm-t2f` | T2-FLAIR | `t2f_sampled_100steps` |

Monitor logs:

```bash
kubectl logs -f pod/train-cwdm-t1n -n gai-lina-group
kubectl logs -f pod/train-cwdm-t1c -n gai-lina-group
kubectl logs -f pod/train-cwdm-t2w -n gai-lina-group
kubectl logs -f pod/train-cwdm-t2f -n gai-lina-group
```

### Training configuration

| Parameter | Value |
|-----------|-------|
| Architecture | WaveUNet (3D U-Net + wavelet layers) |
| Input channels | 32 (8 target + 24 conditioning) |
| Output channels | 8 (wavelet subbands of target) |
| Channel multipliers | 1, 2, 2, 4, 4 |
| Num residual blocks | 2 |
| Num channels | 64 |
| Diffusion steps | 100 (sampled schedule) |
| Noise schedule | linear |
| Batch size | 1 |
| Learning rate | 1e-5 |
| LR anneal steps | 600,000 |
| Image size | 224 × 224 (axial) |
| Loss | MSE + SSIM (hybrid) |
| GPU | 1× A100/A40/L40/RTX-A6000/RTX 3090/4090/V100 |
| Memory | 24 Gi request / 32 Gi limit |

### Resuming training

The training script automatically resumes from the latest checkpoint for each modality:

```bash
ls /pvc/checkpoints/cwdm/brats_t1n_*.pt   # latest is picked up automatically
```

To resume from a specific checkpoint, set `RESUME_CKPT` in the pod spec.

### Environment variables

All training pods are configured via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `TRAIN_MODALITY` | Modality to train: `t1n`, `t1c`, `t2w`, `t2f`, or `all` | `all` |
| `SAMPLE_STRATEGY` | Diffusion sampling schedule: `sampled` or `direct` | `sampled` |
| `TIMESTEPS` | Number of diffusion steps | `100` |
| `TRAIN_DATA_DIR` | Path to training split | `/pvc/data/brats23/train` |
| `VAL_DATA_DIR` | Path to validation split | `/pvc/data/brats23/val` |
| `CHECKPOINT_DIR` | Where checkpoints are saved | `/pvc/checkpoints/cwdm` |
| `WANDB_PROJECT` | WandB project name | `brats2023-cwdm` |
| `WANDB_ENTITY` | WandB entity (from secret) | — |
| `WANDB_API_KEY` | WandB API key (from secret) | — |
| `RESUME_CKPT` | Explicit checkpoint path to resume from | — |

---

## WandB Logging

### Training (per step)

| Metric | Description |
|--------|-------------|
| `loss/MSE` | Mean squared error in wavelet domain |
| `loss/SSIM_loss` | `1 - SSIM` structural similarity loss |
| `loss/Hybrid` | Combined MSE + SSIM loss |
| `metrics/SSIM` | Structural similarity index (higher = better) |

Every **200 steps**: sample images are logged — midplane axial slice of the reconstructed volume (after IDWT) and all 8 wavelet subband channels.

On every **new best SSIM**: a checkpoint event is logged and the previous best is deleted.

### Evaluation (per case)

| Metric | Description |
|--------|-------------|
| `l1` | L1 distance |
| `mse` | Mean squared error |
| `psnr` | Peak signal-to-noise ratio (dB) |
| `ssim` | Structural similarity index |
| `brain_volume_ratio` | Fraction of non-zero voxels (brain mask sanity check) |
| `sample_time` | Inference time per case (seconds) |

Aggregate mean ± std per modality is also logged, along with 3-slice visual comparisons and difference maps.

---

## Evaluation

After training completes, run the evaluation pod on the validation split:

```bash
kubectl apply -f nautilius/jobs/eval-cwdm.yaml
kubectl logs -f pod/eval-cwdm -n gai-lina-group
```

Outputs (NIfTI files + metrics JSON) are written to `/pvc/outputs/cwdm/val/`.

---

## Checkpoints

Checkpoints are saved to `/pvc/checkpoints/cwdm/` on `brats23-pvc`.

```
/pvc/checkpoints/cwdm/
├── brats_t1n_<step>.pt          # saved every 500 steps
├── brats_t1c_<step>.pt
├── brats_t2w_<step>.pt
├── brats_t2f_<step>.pt
└── <modality>/best_ssim/        # best checkpoint per modality (only one kept)
```

Only the best SSIM checkpoint is retained per modality — when a new best is found, the old one is deleted to save disk space.

---

## Data Format

The dataset loader expects the BraTS 2023 NIfTI naming convention:

```
BraTS-GLI-00000-000/
├── BraTS-GLI-00000-000-t1n.nii.gz
├── BraTS-GLI-00000-000-t1c.nii.gz
├── BraTS-GLI-00000-000-t2w.nii.gz
├── BraTS-GLI-00000-000-t2f.nii.gz
└── BraTS-GLI-00000-000-seg.nii.gz   # segmentation (optional for training)
```

Each volume is:
- Loaded as float32
- Clipped to [0.1th, 99.9th] percentile
- Normalized to [0, 1]
- Zero-padded to 240 × 240 × 160, then center-cropped to 224 × 224 × 155

---

## CI/CD

Pushing to `main` with changes to `Dockerfile`, `requirements.txt`, or `run.sh` triggers a GitHub Actions build:

1. Builds the Docker image
2. Pushes to `ghcr.io/kagozi/brats23-cwdm:latest` and `ghcr.io/kagozi/brats23-cwdm:<git-sha>`
3. Uses GitHub Actions layer cache for fast rebuilds

The training pods use `imagePullPolicy: Always` so they always pull the latest image on start.

---

## Troubleshooting

**Pod stuck in `Pending`**
```bash
kubectl describe pod/train-cwdm-t1n -n gai-lina-group
```
Usually a GPU scheduling issue — check node availability. The affinity allows A100, A40, L40, RTX-A6000, RTX 3090/4090, and V100.

**Pod in `Error` state**
```bash
kubectl logs pod/train-cwdm-t1n -n gai-lina-group
```
Check for OOM (exit code 137), missing PVC mounts, or import errors.

**WandB not logging**
- Verify the secret exists: `kubectl get secret wandb-credentials -n gai-lina-group`
- Check the API key is valid and `WANDB_ENTITY` matches your account

**Restarting a failed pod**
Kubernetes pods with `restartPolicy: Never` do not auto-restart. Delete and reapply:
```bash
kubectl delete pod train-cwdm-t1n -n gai-lina-group
kubectl apply -f nautilius/jobs/train-cwdm.yaml
```
Training resumes automatically from the latest checkpoint.

**Checking disk usage on PVC**
```bash
kubectl exec -it pod/train-cwdm-t1n -n gai-lina-group -- df -h /pvc
kubectl exec -it pod/train-cwdm-t1n -n gai-lina-group -- du -sh /pvc/checkpoints/cwdm/
```
