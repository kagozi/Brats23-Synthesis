# BraTS 2023 — NRP Nautilus Setup

This repo contains Kubernetes manifests for uploading and extracting the MICCAI BraTS 2023 dataset onto a Nautilus NRP PersistentVolumeClaim (PVC).

## Prerequisites

- `kubectl` configured and pointing at the Nautilus cluster
- Access to the `gai-lina-group` namespace
- `data/MICCAI-BraTS2023.zip` present locally (not committed to git)

## Kubernetes Resources

| File | Kind | Purpose |
|------|------|---------|
| `nautilius/pods/brats23pvc.yaml` | PersistentVolumeClaim | 200Gi shared storage (`rook-cephfs`) |
| `nautilius/pods/data-uploader.yaml` | Pod | Long-running pod used to `kubectl cp` data onto the PVC |
| `nautilius/jobs/extract-brats.yaml` | Pod | One-shot pod that installs `unzip` and extracts the dataset |

## Data Layout on PVC

After extraction, the PVC is structured as:

```
/pvc/
└── data/
    ├── MICCAI-BraTS2023.zip   ← uploaded zip
    └── brats23/               ← extracted dataset
```

## Step-by-Step Setup

### 1. Create the PVC

```bash
kubectl apply -f nautilius/pods/brats23pvc.yaml
kubectl get pvc brats23-pvc -n gai-lina-group -w
```

Wait until `STATUS` is `Bound`.

### 2. Start the uploader pod

```bash
kubectl apply -f nautilius/pods/data-uploader.yaml
kubectl get pod brats23-uploader -n gai-lina-group -w
```

Wait until `STATUS` is `Running`.

### 3. Copy the dataset zip to the PVC

Run from the project root:

```bash
kubectl cp ./data/MICCAI-BraTS2023.zip brats23-uploader:/pvc/data/ -n gai-lina-group
```

This may take several minutes depending on your connection speed.

### 4. Extract the dataset

```bash
kubectl apply -f nautilius/jobs/extract-brats.yaml
kubectl logs -f pod/extract-brats -n gai-lina-group
```

The job installs `unzip`, extracts the zip into `/pvc/data/brats23/`, and prints the directory contents on completion.

### 5. Verify

```bash
kubectl exec -n gai-lina-group brats23-uploader -- ls -lah /pvc/data/brats23/
```

### 6. Clean up uploader pods

```bash
kubectl delete pod brats23-uploader -n gai-lina-group
kubectl delete pod extract-brats -n gai-lina-group
```

## Notes

- The zip file is excluded from git via `.gitignore` — download it separately from the MICCAI BraTS 2023 challenge page.
- The PVC uses `ReadWriteMany` access mode so multiple pods can mount it simultaneously during training.
- To delete the PVC and all stored data: `kubectl delete pvc brats23-pvc -n gai-lina-group` (irreversible).
