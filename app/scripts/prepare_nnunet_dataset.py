"""
Convert BraTS 2023 training data to nnUNet v2 format (Dataset001_BraTS2023_GLI).

BraTS 2023 GLI labels:
  0 = background
  1 = NCR  (necrotic tumor core)
  2 = SNFH (surrounding non-enhancing FLAIR hyperintensity / edema)
  3 = ET   (enhancing tumor)

No label remapping is needed — BraTS 2023 already uses 1/2/3 (unlike BraTS 2021
which used 1/2/4 and required 4→3 remapping).

nnUNet region evaluation:
  WT (whole tumor)   = labels 1+2+3
  TC (tumor core)    = labels 1+3
  ET (enhancing)     = label  3

Output layout expected by nnUNet:
  $nnUNet_raw/Dataset001_BraTS2023_GLI/
      imagesTr/{case}_0000.nii.gz   ← t1n
      imagesTr/{case}_0001.nii.gz   ← t1c
      imagesTr/{case}_0002.nii.gz   ← t2w
      imagesTr/{case}_0003.nii.gz   ← t2f
      labelsTr/{case}.nii.gz        ← seg (labels 0/1/2/3, no remapping)
      dataset.json
"""

import argparse
import json
import os
import shutil

import nibabel as nib
import numpy as np

DATASET_ID   = "001"
DATASET_NAME = f"Dataset{DATASET_ID}_BraTS2023_GLI"

MODALITY_SUFFIXES = {
    "t1n": "_0000",
    "t1c": "_0001",
    "t2w": "_0002",
    "t2f": "_0003",
}


def verify_labels(seg_path: str) -> None:
    """Warn if unexpected label values are found."""
    data = nib.load(seg_path).get_fdata().astype(np.int32)
    unexpected = set(np.unique(data)) - {0, 1, 2, 3}
    if unexpected:
        print(f"  [WARN] Unexpected labels {unexpected} in {seg_path}")


def convert_case(case_dir: str, images_out: str, labels_out: str) -> bool:
    """
    Convert one BraTS 2023 case to nnUNet format.
    Returns True on success.

    Atomic: all source files are verified to exist before any copying begins,
    so imagesTr/ and labelsTr/ always contain the same set of cases.
    """
    case_name = os.path.basename(case_dir)

    # --- verify all files exist before touching the output dirs ---
    seg_src = os.path.join(case_dir, f"{case_name}-seg.nii.gz")
    if not os.path.exists(seg_src):
        print(f"  [SKIP] {case_name}: missing seg")
        return False

    mod_srcs = {}
    for mod, suffix in MODALITY_SUFFIXES.items():
        src = os.path.join(case_dir, f"{case_name}-{mod}.nii.gz")
        if not os.path.exists(src):
            print(f"  [SKIP] {case_name}: missing {mod}")
            return False
        mod_srcs[suffix] = src

    # --- all files present: copy atomically ---
    for suffix, src in mod_srcs.items():
        shutil.copy2(src, os.path.join(images_out, f"{case_name}{suffix}.nii.gz"))

    verify_labels(seg_src)
    # BraTS 2023 labels are already 0/1/2/3 — copy directly, no remapping
    shutil.copy2(seg_src, os.path.join(labels_out, f"{case_name}.nii.gz"))

    return True


def generate_dataset_json(output_dir: str, num_training: int) -> None:
    # nnUNet v2 region-based training format.
    # "labels" must map region names to lists of raw label values.
    # "regions_class_order" controls the output channel order for the
    # sigmoid heads and must match the order of regions in "labels".
    #
    # BraTS 2023 GLI raw labels: 0=bg, 1=NCR, 2=SNFH/edema, 3=ET
    # Regions:
    #   whole tumor (WT) = 1+2+3
    #   tumor core  (TC) = 1+3
    #   enhancing   (ET) = 3
    dataset_json = {
        "channel_names": {
            "0": "T1n",
            "1": "T1c",
            "2": "T2w",
            "3": "T2f",
        },
        "labels": {
            "background":      0,
            "whole tumor":     [1, 2, 3],
            "tumor core":      [1, 3],
            "enhancing tumor": [3],
        },
        "regions_class_order": [1, 2, 3],
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "dataset_name": DATASET_NAME,
        "reference": "BraTS 2023 GLI Challenge",
        "release": "1.0",
    }
    path = os.path.join(output_dir, "dataset.json")
    with open(path, "w") as f:
        json.dump(dataset_json, f, indent=2)
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert BraTS 2023 training split to nnUNet Dataset001"
    )
    parser.add_argument(
        "--input_dir",
        default="/pvc/data/brats23/train",
        help="BraTS 2023 training split directory",
    )
    parser.add_argument(
        "--nnunet_raw",
        default=os.environ.get("nnUNet_raw", "/pvc/nnunet/raw"),
        help="nnUNet_raw root (env var nnUNet_raw)",
    )
    args = parser.parse_args()

    output_dir  = os.path.join(args.nnunet_raw, DATASET_NAME)
    images_out  = os.path.join(output_dir, "imagesTr")
    labels_out  = os.path.join(output_dir, "labelsTr")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    case_dirs = sorted(
        d for d in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, d))
    )
    print(f"Found {len(case_dirs)} cases in {args.input_dir}")

    ok = 0
    for name in case_dirs:
        if convert_case(os.path.join(args.input_dir, name), images_out, labels_out):
            ok += 1

    print(f"\nConverted {ok}/{len(case_dirs)} cases → {output_dir}")
    generate_dataset_json(output_dir, ok)

    print(f"""
Next steps (handled by nnunet-train.yaml):
  export nnUNet_raw={args.nnunet_raw}
  nnUNetv2_plan_and_preprocess -d {DATASET_ID} --verify_dataset_integrity
  nnUNetv2_train {DATASET_ID} 3d_fullres FOLD   (for FOLD in 0 1 2 3 4)
  nnUNetv2_find_best_configuration {DATASET_ID} -c 3d_fullres
""")


if __name__ == "__main__":
    main()
