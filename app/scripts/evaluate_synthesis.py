"""
Segmentation-based evaluation of MRI synthesis quality.

Pipeline:
  1. nnUNet predicts segmentation masks on (3 real + 1 synthesized) test inputs
  2. Predicted masks are compared to ground-truth seg.nii.gz from BraTS test split
  3. WT / TC / ET Dice is reported per subject and aggregated

BraTS 2023 GLI labels:
  0 = background
  1 = NCR  (necrotic tumor core)
  2 = SNFH (surrounding non-enhancing FLAIR hyperintensity / edema)
  3 = ET   (enhancing tumor)

BraTS evaluation regions:
  WT (whole tumor)  = labels 1 + 2 + 3
  TC (tumor core)   = labels 1 + 3
  ET (enhancing)    = label  3

Usage:
  python scripts/evaluate_synthesis.py \\
    --pred_dir      /pvc/nnunet/predictions/t1n \\
    --gt_dir        /pvc/data/brats23/test \\
    --synthesized_modality t1n \\
    --output_dir    /pvc/outputs/cwdm/dice
"""

import argparse
import csv
import os

import nibabel as nib
import numpy as np
import wandb


# ─────────────────────────────────────────────────────────────────────────────
# Dice helpers
# ─────────────────────────────────────────────────────────────────────────────

def binary_dice(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    intersection = np.sum(pred_bin & gt_bin)
    denom = np.sum(pred_bin) + np.sum(gt_bin)
    if denom == 0:
        return float('nan')   # both empty — undefined, don't count
    return float(2 * intersection / denom)


def region_dice(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Compute WT/TC/ET Dice for BraTS 2023 labels (1/2/3).

    pred, gt: integer arrays with values in {0,1,2,3}.
    """
    wt_dice = binary_dice(pred > 0,              gt > 0)
    tc_dice = binary_dice(np.isin(pred, [1, 3]), np.isin(gt, [1, 3]))
    et_dice = binary_dice(pred == 3,             gt == 3)
    return {"WT": wt_dice, "TC": tc_dice, "ET": et_dice}


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(pred_dir: str, gt_dir: str, synth_mod: str) -> list[dict]:
    """
    Compare nnUNet predictions in pred_dir against BraTS ground-truth seg masks.
    Returns list of per-subject result dicts.
    """
    results = []

    pred_files = sorted(
        f for f in os.listdir(pred_dir)
        if f.endswith(".nii.gz")
    )
    print(f"Found {len(pred_files)} prediction files in {pred_dir}")

    for fname in pred_files:
        # nnUNet outputs filenames like BraTS-GLI-00001-000.nii.gz
        case = fname.replace(".nii.gz", "")
        gt_path = os.path.join(gt_dir, case, f"{case}-seg.nii.gz")

        if not os.path.exists(gt_path):
            print(f"  [SKIP] {case}: ground-truth seg not found at {gt_path}")
            continue

        try:
            pred_np = nib.load(os.path.join(pred_dir, fname)).get_fdata().astype(np.int32)
            gt_np   = nib.load(gt_path).get_fdata().astype(np.int32)

            if pred_np.shape != gt_np.shape:
                print(f"  [WARN] {case}: shape mismatch pred={pred_np.shape} gt={gt_np.shape}")

            scores = region_dice(pred_np, gt_np)
            row = {"subject": case, **scores}
            results.append(row)

            print(f"  {case}  WT={scores['WT']:.4f}  TC={scores['TC']:.4f}  ET={scores['ET']:.4f}")

        except Exception as e:
            print(f"  [ERROR] {case}: {e}")

    return results


def summarise(results: list[dict], synth_mod: str) -> dict:
    """Print and return aggregate statistics."""
    def valid(key):
        return [r[key] for r in results if not np.isnan(r[key])]

    stats = {}
    print(f"\n=== {synth_mod} Segmentation Dice Summary ({len(results)} subjects) ===")
    for region in ["WT", "TC", "ET"]:
        vals = valid(region)
        if vals:
            mean, std = np.mean(vals), np.std(vals)
            stats[f"{region}_mean"] = mean
            stats[f"{region}_std"]  = std
            print(f"  {region}: {mean:.4f} ± {std:.4f}  (n={len(vals)})")
        else:
            stats[f"{region}_mean"] = float('nan')
            stats[f"{region}_std"]  = float('nan')
            print(f"  {region}: no valid scores")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="WT/TC/ET Dice evaluation of nnUNet predictions vs. ground truth"
    )
    parser.add_argument("--pred_dir", required=True,
                        help="Directory of nnUNet prediction .nii.gz files")
    parser.add_argument("--gt_dir", default="/pvc/data/brats23/test",
                        help="BraTS 2023 test split (contains {case}/{case}-seg.nii.gz)")
    parser.add_argument("--synthesized_modality", required=True,
                        choices=["t1n", "t1c", "t2w", "t2f"],
                        help="Which modality was synthesized (for logging)")
    parser.add_argument("--output_dir", default="/pvc/outputs/cwdm/dice",
                        help="Where to write the per-subject CSV")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    mod = args.synthesized_modality

    # ── WandB init ────────────────────────────────────────────────────────────
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "brats2023-cwdm"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        name=f"dice_{mod}",
        config=vars(args),
        mode=os.environ.get("WANDB_MODE", "online"),
        tags=["segmentation", "dice", "test-set", mod],
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    results = evaluate(args.pred_dir, args.gt_dir, mod)

    if not results:
        print("No results — check pred_dir and gt_dir paths.")
        wandb.finish()
        return

    # ── Per-subject WandB logging ─────────────────────────────────────────────
    for r in results:
        wandb.log({
            f"{mod}/WT_dice": r["WT"],
            f"{mod}/TC_dice": r["TC"],
            f"{mod}/ET_dice": r["ET"],
            "subject": r["subject"],
        })

    # ── Aggregate summary ─────────────────────────────────────────────────────
    stats = summarise(results, mod)

    wandb.summary[f"{mod}/WT_mean"] = stats["WT_mean"]
    wandb.summary[f"{mod}/WT_std"]  = stats["WT_std"]
    wandb.summary[f"{mod}/TC_mean"] = stats["TC_mean"]
    wandb.summary[f"{mod}/TC_std"]  = stats["TC_std"]
    wandb.summary[f"{mod}/ET_mean"] = stats["ET_mean"]
    wandb.summary[f"{mod}/ET_std"]  = stats["ET_std"]
    wandb.summary[f"{mod}/n_subjects"] = len(results)

    # WandB Table
    table = wandb.Table(columns=["subject", "WT", "TC", "ET"])
    for r in results:
        table.add_data(r["subject"], r["WT"], r["TC"], r["ET"])
    wandb.log({f"{mod}/per_subject_dice": table})

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, f"dice_{mod}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "WT", "TC", "ET"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nDice scores saved to: {csv_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
