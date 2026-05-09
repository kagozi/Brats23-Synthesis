"""
Reformat synthesis outputs into nnUNet imagesTs/ format for inference.

For each synthesized modality (t1n/t1c/t2w/t2f), this script assembles:
  - 3 real modalities from the BraTS test split
  - 1 synthesized modality from sample.py output (sample.nii.gz)

into the nnUNet 4-channel input format:
  imagesTs/{case}_0000.nii.gz  ← t1n
  imagesTs/{case}_0001.nii.gz  ← t1c
  imagesTs/{case}_0002.nii.gz  ← t2w
  imagesTs/{case}_0003.nii.gz  ← t2f

The synthesized image (224×224×155) is padded back to 240×240×155
to match the original BraTS spatial dimensions before being saved with
the original affine/header so nnUNet geometry checks pass.

Outputs (one directory per synthesized modality):
  $nnunet_input_root/imagesTs_t1n/
  $nnunet_input_root/imagesTs_t1c/
  $nnunet_input_root/imagesTs_t2w/
  $nnunet_input_root/imagesTs_t2f/

Usage:
  python scripts/reformat_for_nnunet.py \\
    --synthesis_dir /pvc/outputs/cwdm/test \\
    --test_data_dir /pvc/data/brats23/test \\
    --output_root   /pvc/nnunet/imagesTs
"""

import argparse
import os

import nibabel as nib
import numpy as np


MODALITY_ORDER = ["t1n", "t1c", "t2w", "t2f"]   # maps to _0000 … _0003


def pad_synthesized(synth_np: np.ndarray) -> np.ndarray:
    """
    Pad a 224×224×155 synthesized volume back to 240×240×155.
    sample.py crops 8px each side in x/y (bratsloader does t[:, 8:-8, 8:-8, :]).
    This reverses that crop.
    """
    assert synth_np.shape == (224, 224, 155), (
        f"Expected synthesized shape (224,224,155), got {synth_np.shape}"
    )
    padded = np.zeros((240, 240, 155), dtype=synth_np.dtype)
    padded[8:232, 8:232, :] = synth_np
    return padded


def reformat_modality(synth_mod: str, synthesis_dir: str, test_data_dir: str,
                      output_dir: str) -> int:
    """
    Build imagesTs/ for one synthesized modality.
    Returns number of cases successfully processed.
    """
    os.makedirs(output_dir, exist_ok=True)
    real_modalities = [m for m in MODALITY_ORDER if m != synth_mod]

    # Subjects for which we have synthesis output
    synth_subjects = sorted(
        d for d in os.listdir(synthesis_dir)
        if os.path.isdir(os.path.join(synthesis_dir, d))
        and os.path.exists(os.path.join(synthesis_dir, d, "sample.nii.gz"))
    )

    ok = 0
    for case in synth_subjects:
        case_real_dir  = os.path.join(test_data_dir,  case)
        synth_sample   = os.path.join(synthesis_dir,  case, "sample.nii.gz")

        if not os.path.isdir(case_real_dir):
            print(f"  [SKIP] {case}: not found in test_data_dir")
            continue

        try:
            # Load synthesized image and pad back to 240×240×155
            synth_img   = nib.load(synth_sample)
            synth_np    = synth_img.get_fdata().astype(np.float32)
            synth_padded = pad_synthesized(synth_np)

            # Use original file header/affine so nnUNet geometry checks pass
            ref_path = os.path.join(case_real_dir, f"{case}-{real_modalities[0]}.nii.gz")
            ref_img  = nib.load(ref_path)

            # Write all 4 channels
            for mod in MODALITY_ORDER:
                suffix = f"_000{MODALITY_ORDER.index(mod)}"
                dst    = os.path.join(output_dir, f"{case}{suffix}.nii.gz")

                if mod == synth_mod:
                    # Use synthesized (padded back to original dims)
                    out_img = nib.Nifti1Image(synth_padded, ref_img.affine, ref_img.header)
                else:
                    # Use original real modality file directly
                    src = os.path.join(case_real_dir, f"{case}-{mod}.nii.gz")
                    if not os.path.exists(src):
                        raise FileNotFoundError(f"Missing real modality: {src}")
                    out_img = nib.load(src)

                nib.save(out_img, dst)

            ok += 1

        except Exception as e:
            print(f"  [ERROR] {case}: {e}")
            continue

    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Reformat synthesis outputs to nnUNet imagesTs/ format"
    )
    parser.add_argument("--synthesis_dir", default="/pvc/outputs/cwdm/test",
                        help="Root dir from sample.py (contains {case}/sample.nii.gz)")
    parser.add_argument("--test_data_dir", default="/pvc/data/brats23/test",
                        help="BraTS 2023 test split (original modalities + seg)")
    parser.add_argument("--output_root", default="/pvc/nnunet/imagesTs",
                        help="Root where imagesTs_{mod}/ dirs will be created")
    parser.add_argument("--modalities", nargs="+", default=MODALITY_ORDER,
                        choices=MODALITY_ORDER,
                        help="Which synthesized modalities to reformat (default: all 4)")
    args = parser.parse_args()

    print(f"Synthesis dir : {args.synthesis_dir}")
    print(f"Test data dir : {args.test_data_dir}")
    print(f"Output root   : {args.output_root}")
    print(f"Modalities    : {args.modalities}")
    print()

    for mod in args.modalities:
        out_dir = os.path.join(args.output_root, f"imagesTs_{mod}")
        print(f"=== Reformatting synthesized {mod} → {out_dir} ===")
        n = reformat_modality(mod, args.synthesis_dir, args.test_data_dir, out_dir)
        print(f"    Done: {n} cases\n")

    print("Reformat complete.")


if __name__ == "__main__":
    main()
