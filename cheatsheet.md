# Nautilus NRP — kubectl Cheatsheet for BraTS23

## One-time setup

```bash
# 1. Create PVC (200Gi shared storage)
kubectl apply -f nautilius/pods/brats23pvc.yaml
kubectl get pvc brats23-pvc -n gai-lina-group -w
```

## Upload data to PVC

```bash
# Start uploader pod
kubectl apply -f nautilius/pods/data-uploader.yaml
kubectl get pod brats23-uploader -n gai-lina-group -w   # wait for Running

# Upload zip file (run from project root)
kubectl cp ./data/MICCAI-BraTS2023.zip brats23-uploader:/pvc/data/ -n gai-lina-group

# Extract dataset via one-shot job
kubectl apply -f nautilius/jobs/extract-brats.yaml
kubectl logs -f pod/extract-brats -n gai-lina-group

# Verify extraction
kubectl exec -n gai-lina-group brats23-uploader -- ls -lah /pvc/data/brats23/

# Delete uploader and extractor pods when done
kubectl delete pod brats23-uploader -n gai-lina-group
kubectl delete pod extract-brats -n gai-lina-group
```

## Inspect dataset on PVC

```bash
# Run inspection pod (installs nibabel, prints case counts + modality completeness)
kubectl apply -f nautilius/jobs/inspect-brats.yaml
kubectl logs -f pod/inspect-brats -n gai-lina-group

# Clean up when done
kubectl delete pod inspect-brats -n gai-lina-group
```

The script (`scripts/inspect_brats.py`) can also be run locally:

```bash
pip install nibabel numpy
python scripts/inspect_brats.py --root /local/path/to/brats23
```

## Build & push Docker image

```bash
# Option A: Manual (local Docker)
docker build -t ghcr.io/{your_name}/brats23-train:latest \
    -f docker/Dockerfile.train .
docker push ghcr.io/{your_name}/brats23-train:latest

# Option B: GitHub Actions (auto on push to main)
git push origin main   # triggers .github/workflows/docker-build.yml
```

## Run training jobs

```bash
# Monitor all jobs
kubectl get jobs -n gai-lina-group -w
```

## Monitor & debug

```bash
# Check pod/job status
kubectl describe pod brats23-uploader -n gai-lina-group


# GPU check inside pod
nvidia-smi

# Check PVC data contents
kubectl exec -n gai-lina-group brats23-uploader -- ls -lah /pvc/data/brats23/
kubectl exec -n gai-lina-group brats23-uploader -- ls -lah /pvc/checkpoints/
```

## Cleanup

```bash
# Delete completed jobs
kubectl delete job brats23-train-fold0 -n gai-lina-group

# Delete all brats23 jobs
kubectl delete jobs -l app=brats23 -n gai-lina-group

# Delete PVC (WARNING: destroys all data)
kubectl delete pvc brats23-pvc -n gai-lina-group
```

## Copy results from PVC to local

```bash
# Restart uploader pod first
kubectl apply -f nautilius/pods/data-uploader.yaml
kubectl get pod brats23-uploader -n gai-lina-group -w