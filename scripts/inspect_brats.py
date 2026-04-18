"""
inspect_brats.py — Inspect the BraTS 2023 dataset on the PVC.

Expects data at /pvc/data/brats23/ (extracted from MICCAI-BraTS2023.zip).
Reports directory structure, case counts, modality completeness, and sample volume stats.

Usage (local):  python scripts/inspect_brats.py [--root /path/to/brats23]
Usage (in pod): python /scripts/inspect_brats.py
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

try:
    import nibabel as nib  # type: ignore[import-untyped]
    import numpy as np  # type: ignore[import-untyped]
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

# BraTS 2023 expected modality suffixes
MODALITIES = ["t1c", "t1n", "t2f", "t2w"]
SEG_SUFFIX = "seg"


def find_root(base: Path) -> Path:
    """Walk one level down if the zip extracted into a subdirectory."""
    candidates = [base] + [p for p in base.iterdir() if p.is_dir()]
    for c in candidates:
        if any(c.iterdir()):
            entries = list(c.iterdir())
            if entries and entries[0].is_dir():
                return c
    return base


def inspect(root: Path) -> None:
    print(f"\n{'='*60}")
    print(f"BraTS 2023 Dataset Inspector")
    print(f"Root: {root}")
    print(f"{'='*60}\n")

    if not root.exists():
        print(f"ERROR: {root} does not exist. Has the zip been extracted?")
        sys.exit(1)

    # ── Top-level structure ──────────────────────────────────────
    print("Top-level contents:")
    for item in sorted(root.iterdir()):
        size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
        print(f"  {'[DIR] ' if item.is_dir() else '      '}{item.name}  ({size / 1e9:.2f} GB)")

    # ── Discover case directories ────────────────────────────────
    case_dirs = sorted([p for p in root.rglob("*") if p.is_dir() and
                        any(f.suffix in (".gz", ".nii") for f in p.iterdir() if f.is_file())])

    print(f"\nTotal cases found: {len(case_dirs)}")

    if not case_dirs:
        print("No NIfTI files found — check extraction path.")
        return

    # ── Modality completeness check ──────────────────────────────
    missing = defaultdict(list)
    has_seg = 0

    for case in case_dirs:
        files = {f.name for f in case.iterdir() if f.is_file()}
        for mod in MODALITIES:
            if not any(mod in f for f in files):
                missing[mod].append(case.name)
        if any(SEG_SUFFIX in f for f in files):
            has_seg += 1

    print(f"\nModality completeness (out of {len(case_dirs)} cases):")
    for mod in MODALITIES:
        n_missing = len(missing[mod])
        status = "OK" if n_missing == 0 else f"MISSING in {n_missing} cases"
        print(f"  {mod:>5}: {status}")
    print(f"  {'seg':>5}: present in {has_seg} cases")

    if any(missing.values()):
        print("\nCases with missing modalities:")
        for mod, cases in missing.items():
            for c in cases[:5]:
                print(f"  [{mod}] {c}")
            if len(cases) > 5:
                print(f"  ... and {len(cases)-5} more")

    # ── Sample volume stats (first case) ────────────────────────
    if HAS_NIBABEL:
        sample_case = case_dirs[0]
        nii_files = sorted(sample_case.glob("*.nii.gz"))
        print(f"\nSample case: {sample_case.name}")
        for nf in nii_files:
            img = nib.load(str(nf))
            arr = img.get_fdata()
            print(f"  {nf.name}")
            print(f"    shape={img.shape}  dtype={arr.dtype}  "
                  f"min={arr.min():.1f}  max={arr.max():.1f}  "
                  f"affine_det={np.linalg.det(img.affine):.3f}")
    else:
        print("\nnibabel not installed — skipping volume stats.")

    print(f"\n{'='*60}")
    print("Inspection complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/pvc/data/brats23",
                        help="Path to extracted brats23 directory")
    args = parser.parse_args()
    inspect(Path(args.root))
