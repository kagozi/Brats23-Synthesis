"""
A script for sampling from a diffusion model for paired image-to-image translation.
"""

import argparse
import csv
import nibabel as nib
import numpy as np
import os
import pathlib
import random
import sys
import torch as th
import wandb

sys.path.append(".")

from guided_diffusion import (dist_util, logger)
from guided_diffusion.bratsloader import BRATSVolumes
from guided_diffusion.script_util import (model_and_diffusion_defaults, create_model_and_diffusion,
                                          add_dict_to_argparser, args_to_dict)
from DWT_IDWT.DWT_IDWT_layer import IDWT_3D, DWT_3D
from monai.metrics import SSIMMetric


def compute_metrics(pred_np, target_np):
    """Compute MSE, PSNR, and SSIM (brain-masked) using monai metrics.

    pred_np, target_np: numpy arrays [H, W, D] in [0, 1]
    Returns (mse, psnr, ssim) as floats.
    """
    # Tensors in [B, C, H, W, D] format for monai
    pred_t = th.from_numpy(pred_np).float().unsqueeze(0).unsqueeze(0)
    tgt_t = th.from_numpy(target_np).float().unsqueeze(0).unsqueeze(0)

    # Brain mask from ground truth
    mask = (tgt_t > 0.01).float()
    pred_m = pred_t * mask
    tgt_m = tgt_t * mask

    mse = float(th.mean((pred_m - tgt_m) ** 2).item())
    psnr = float(10 * np.log10(1.0 / mse)) if mse > 0 else float('inf')

    try:
        ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0, win_size=7)
        ssim_val = float(ssim_metric(y_pred=pred_m, y=tgt_m).mean().item())
    except Exception as e:
        print(f"  Warning: SSIM failed: {e}")
        ssim_val = float('nan')

    return mse, psnr, ssim_val


def main():
    args = create_argparser().parse_args()
    seed = args.seed
    dist_util.setup_dist(devices=args.devices)
    logger.configure()

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "brats2023-cwdm"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        name=f"eval_test_{args.contr}",
        config=vars(args),
        mode=os.environ.get("WANDB_MODE", "online"),
        tags=["evaluation", "test-set", args.contr],
    )

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    diffusion.mode = 'i2i'
    logger.log("Load model from: {}".format(args.model_path))
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev())

    if args.dataset == 'brats':
        ds = BRATSVolumes(args.data_dir, mode='eval')

    datal = th.utils.data.DataLoader(ds,
                                     batch_size=args.batch_size,
                                     num_workers=8,
                                     shuffle=False)

    model.eval()
    idwt = IDWT_3D("haar")
    dwt = DWT_3D("haar")

    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    metrics_rows = []

    for batch in iter(datal):
        batch['t1n'] = batch['t1n'].to(dist_util.dev())
        batch['t1c'] = batch['t1c'].to(dist_util.dev())
        batch['t2w'] = batch['t2w'].to(dist_util.dev())
        batch['t2f'] = batch['t2f'].to(dist_util.dev())

        # Robust subject name extraction — works regardless of parent path
        subj_path = batch['subj'][0]
        subj = os.path.basename(os.path.dirname(subj_path))
        print(f"Processing: {subj}")

        if args.contr == 't1n':
            target = batch['t1n']
            cond_1 = batch['t1c']
            cond_2 = batch['t2w']
            cond_3 = batch['t2f']
        elif args.contr == 't1c':
            target = batch['t1c']
            cond_1 = batch['t1n']
            cond_2 = batch['t2w']
            cond_3 = batch['t2f']
        elif args.contr == 't2w':
            target = batch['t2w']
            cond_1 = batch['t1n']
            cond_2 = batch['t1c']
            cond_3 = batch['t2f']
        elif args.contr == 't2f':
            target = batch['t2f']
            cond_1 = batch['t1n']
            cond_2 = batch['t1c']
            cond_3 = batch['t2w']
        else:
            print(f"Unknown contrast: {args.contr}")
            continue

        # Conditioning vector (3 × 8 wavelet subbands = 24 channels)
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(cond_1)
        cond = th.cat([LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(cond_2)
        cond = th.cat([cond, LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(cond_3)
        cond = th.cat([cond, LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)

        noise = th.randn(args.batch_size, 8, 112, 112, 80).to(dist_util.dev())

        sample = diffusion.p_sample_loop(
            model=model,
            shape=noise.shape,
            noise=noise,
            cond=cond,
            clip_denoised=args.clip_denoised,
            model_kwargs={},
        )

        B, _, D, H, W = sample.size()
        sample = idwt(sample[:, 0, :, :, :].view(B, 1, D, H, W) * 3.,
                      sample[:, 1, :, :, :].view(B, 1, D, H, W),
                      sample[:, 2, :, :, :].view(B, 1, D, H, W),
                      sample[:, 3, :, :, :].view(B, 1, D, H, W),
                      sample[:, 4, :, :, :].view(B, 1, D, H, W),
                      sample[:, 5, :, :, :].view(B, 1, D, H, W),
                      sample[:, 6, :, :, :].view(B, 1, D, H, W),
                      sample[:, 7, :, :, :].view(B, 1, D, H, W))

        sample = sample.clamp(0, 1)
        sample[cond_1 == 0] = 0  # zero out non-brain voxels

        if len(sample.shape) == 5:
            sample = sample.squeeze(dim=1)
        sample = sample[:, :, :, :155]

        if len(target.shape) == 5:
            target = target.squeeze(dim=1)
        target = target[:, :, :, :155]

        subj_dir = os.path.join(args.output_dir, subj)
        pathlib.Path(subj_dir).mkdir(parents=True, exist_ok=True)

        for i in range(sample.shape[0]):
            pred_np = sample.detach().cpu().numpy()[i]
            tgt_np = target.detach().cpu().numpy()[i]

            sample_path = os.path.join(subj_dir, 'sample.nii.gz')
            nib.save(nib.Nifti1Image(pred_np, np.eye(4)), sample_path)
            print(f'Saved: {sample_path}')

            target_path = os.path.join(subj_dir, 'target.nii.gz')
            nib.save(nib.Nifti1Image(tgt_np, np.eye(4)), target_path)

            # Compute metrics (brain masking applied inside compute_metrics)
            if tgt_np.max() > 0:
                mse, psnr, ssim_val = compute_metrics(pred_np, tgt_np)
            else:
                mse, psnr, ssim_val = float('nan'), float('nan'), float('nan')
            print(f'  MSE={mse:.6f}  PSNR={psnr:.2f}  SSIM={ssim_val:.4f}')
            metrics_rows.append({'subject': subj, 'MSE': mse, 'PSNR': psnr, 'SSIM': ssim_val})
            wandb.log({
                f"{args.contr}/MSE": mse,
                f"{args.contr}/PSNR": psnr,
                f"{args.contr}/SSIM": ssim_val,
                "subject": subj,
            })

    # Save per-subject metrics CSV
    csv_path = os.path.join(args.output_dir, f'metrics_{args.contr}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['subject', 'MSE', 'PSNR', 'SSIM'])
        writer.writeheader()
        writer.writerows(metrics_rows)

    if metrics_rows:
        mses  = [r['MSE']  for r in metrics_rows if np.isfinite(r['MSE'])]
        psnrs = [r['PSNR'] for r in metrics_rows if np.isfinite(r['PSNR'])]
        ssims = [r['SSIM'] for r in metrics_rows if np.isfinite(r['SSIM'])]
        n = len(mses)

        print(f"\n=== {args.contr} Summary ({n} subjects) ===")
        print(f"  MSE  : {np.mean(mses):.6f} ± {np.std(mses):.6f}")
        print(f"  PSNR : {np.mean(psnrs):.2f} ± {np.std(psnrs):.2f} dB")
        print(f"  SSIM : {np.mean(ssims):.4f} ± {np.std(ssims):.4f}")

        # Log aggregate summary to WandB
        wandb.summary[f"{args.contr}/MSE_mean"] = np.mean(mses)
        wandb.summary[f"{args.contr}/MSE_std"] = np.std(mses)
        wandb.summary[f"{args.contr}/PSNR_mean"] = np.mean(psnrs)
        wandb.summary[f"{args.contr}/PSNR_std"] = np.std(psnrs)
        wandb.summary[f"{args.contr}/SSIM_mean"] = np.mean(ssims)
        wandb.summary[f"{args.contr}/SSIM_std"] = np.std(ssims)
        wandb.summary[f"{args.contr}/n_subjects"] = n

        # Log per-subject results as a WandB Table
        table = wandb.Table(columns=["subject", "MSE", "PSNR", "SSIM"])
        for r in metrics_rows:
            table.add_data(r['subject'], r['MSE'], r['PSNR'], r['SSIM'])
        wandb.log({f"{args.contr}/per_subject_metrics": table})

    print(f"\nMetrics saved to: {csv_path}")
    wandb.finish()


def create_argparser():
    defaults = dict(
        seed=0,
        dataset='brats',
        data_dir="",
        clip_denoised=True,
        batch_size=1,
        model_path="",
        devices=[0],
        output_dir='./results',
        contr='t1n',
        # Architecture — must match training config exactly
        image_size=224,
        num_channels=64,
        in_channels=32,
        out_channels=8,
        num_res_blocks=2,
        num_heads=1,
        num_groups=32,
        channel_mult="1,2,2,4,4",
        dims=3,
        learn_sigma=False,
        use_scale_shift_norm=False,
        attention_resolutions="",
        use_fp16=True,
        # Diffusion
        diffusion_steps=100,
        noise_schedule='linear',
        sample_schedule='sampled',
    )
    defaults.update({k: v for k, v in model_and_diffusion_defaults().items() if k not in defaults})
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
