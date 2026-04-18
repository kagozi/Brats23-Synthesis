# BraTS 2023 — NRP Nautilus Setup

This repo contains Kubernetes manifests for uploading and extracting the MICCAI BraTS 2023 dataset onto a Nautilus NRP PersistentVolumeClaim (PVC).

---

## 0. Get the dataset

Download `ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip`, then move it into the `data/` folder and rename it:

**macOS / Linux**
```bash
mkdir -p data && mv ~/Downloads/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip data/MICCAI-BraTS2023.zip
```

**Windows (PowerShell)**
```powershell
New-Item -ItemType Directory -Force -Path data
Move-Item "$env:USERPROFILE\Downloads\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip" "data\MICCAI-BraTS2023.zip"
```

> The zip is excluded from git via `.gitignore` and must never be committed.

---

## 1. Set up kubectl for NRP Nautilus

### Install kubectl

**macOS**
```bash
brew install kubectl
```

**Linux**
```bash
curl -LO "https://dl.k8s.io/release/$(curl -sL https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl && sudo mv kubectl /usr/local/bin/
```

**Windows** — download the `.exe` from the [official Kubernetes release page](https://kubernetes.io/docs/tasks/tools/install-kubectl-windows/) and add it to your `PATH`.

### Configure access to Nautilus

1. Go to [https://nrp.ai](https://nrp.ai) and log in with your institution credentials.
2. Click your name in the top-right → **Get Config**.
3. Save the downloaded file as `~/.kube/config` (macOS/Linux) or `%USERPROFILE%\.kube\config` (Windows).

```bash
# macOS / Linux — move the downloaded config
mkdir -p ~/.kube
mv ~/Downloads/config ~/.kube/config
chmod 600 ~/.kube/config
```

4. Verify access:
```bash
kubectl get pods -n gai-lina-group
```

You should see an empty list (or existing pods) without any authentication errors.

---

## 2. Prerequisites checklist

- [ ] `kubectl` installed and pointing at the Nautilus cluster (`kubectl config current-context`)
- [ ] Access to the `gai-lina-group` namespace
- [ ] `data/MICCAI-BraTS2023.zip` present locally (step 0 above)

---

## 3. Kubernetes Resources

| File | Kind | Purpose |
|------|------|---------|
| `nautilius/pods/brats23pvc.yaml` | PersistentVolumeClaim | 200Gi shared storage (`rook-cephfs`) |
| `nautilius/pods/data-uploader.yaml` | Pod | Long-running pod used to `kubectl cp` data onto the PVC |
| `nautilius/jobs/extract-brats.yaml` | Pod | One-shot pod that installs `unzip` and extracts the dataset |
| `nautilius/jobs/inspect-brats.yaml` | Pod | One-shot pod that reports case counts and modality completeness |

---

## 4. Data Layout on PVC

After extraction, the PVC is structured as:

```
/pvc/
└── data/
    ├── MICCAI-BraTS2023.zip   ← uploaded zip
    └── brats23/               ← extracted dataset
```

---

## 5. Step-by-Step Setup

### 5.1 Create the PVC

```bash
kubectl apply -f nautilius/pods/brats23pvc.yaml
kubectl get pvc brats23-pvc -n gai-lina-group -w
```

Wait until `STATUS` is `Bound`.

### 5.2 Start the uploader pod

```bash
kubectl apply -f nautilius/pods/data-uploader.yaml
kubectl get pod brats23-uploader -n gai-lina-group -w
```

Wait until `STATUS` is `Running`.

### 5.3 Copy the dataset zip to the PVC

Run from the project root:

```bash
kubectl cp ./data/MICCAI-BraTS2023.zip brats23-uploader:/pvc/data/ -n gai-lina-group
```

This may take several minutes depending on your connection speed.

### 5.4 Extract the dataset

```bash
kubectl apply -f nautilius/jobs/extract-brats.yaml
kubectl logs -f pod/extract-brats -n gai-lina-group
```

The job installs `unzip`, extracts the zip into `/pvc/data/brats23/`, and prints the directory contents on completion.

### 5.5 Inspect the dataset

```bash
kubectl apply -f nautilius/jobs/inspect-brats.yaml
kubectl logs -f pod/inspect-brats -n gai-lina-group
```

Reports top-level structure, total case count, modality completeness (`t1c`, `t1n`, `t2f`, `t2w`, `seg`), and sample volume stats.

### 5.6 Verify

```bash
kubectl exec -n gai-lina-group brats23-uploader -- ls -lah /pvc/data/brats23/
```

### 5.7 Clean up

```bash
kubectl delete pod brats23-uploader -n gai-lina-group
kubectl delete pod extract-brats -n gai-lina-group
kubectl delete pod inspect-brats -n gai-lina-group
```

---

## Notes

- The PVC uses `ReadWriteMany` so multiple pods can mount it simultaneously during training.
- To delete the PVC and **all stored data**: `kubectl delete pvc brats23-pvc -n gai-lina-group` (irreversible).
- See `cheatsheet.md` for quick-reference commands.
