#!/bin/bash
# =============================================================================
# Fast-CWDM BraTS 2023 training script
# Replicates https://github.com/tsereda/brats-synthesis
#
# Env vars (set in k8s pod / docker run):
#   TRAIN_MODALITY   : t1n | t1c | t2w | t2f | all  (default: all)
#   SAMPLE_STRATEGY  : sampled | direct              (default: sampled)
#   TIMESTEPS        : diffusion steps               (default: 100)
#   TRAIN_DATA_DIR   : path to train split           (default: /pvc/data/brats23/train)
#   VAL_DATA_DIR     : path to val split             (default: /pvc/data/brats23/val)
#   RESUME_CKPT      : path to checkpoint to resume  (default: none)
#   WANDB_PROJECT    : WandB project name            (default: brats2023-cwdm)
#   WANDB_ENTITY     : WandB entity/username
#   WANDB_API_KEY    : WandB API key
# =============================================================================
set -e

MODALITY="${TRAIN_MODALITY:-all}"
STRATEGY="${SAMPLE_STRATEGY:-sampled}"
TIMESTEPS="${TIMESTEPS:-100}"
TRAIN_DIR="${TRAIN_DATA_DIR:-/pvc/data/brats23/train}"
VAL_DIR="${VAL_DATA_DIR:-/pvc/data/brats23/val}"
CKPT_DIR="${CHECKPOINT_DIR:-/pvc/checkpoints/cwdm}"
WANDB_PROJECT="${WANDB_PROJECT:-brats2023-cwdm}"

mkdir -p "$CKPT_DIR"

# Log config
echo "========================================"
echo " Fast-CWDM BraTS 2023 Training"
echo "========================================"
echo " Modality      : $MODALITY"
echo " Sample strat  : $STRATEGY"
echo " Timesteps     : $TIMESTEPS"
echo " Train dir     : $TRAIN_DIR"
echo " Val dir       : $VAL_DIR"
echo " Checkpoint dir: $CKPT_DIR"
echo " WandB project : $WANDB_PROJECT"
echo " WandB entity  : ${WANDB_ENTITY:-not set}"
echo "========================================"

train_modality() {
    local mod=$1
    echo ""
    echo ">>> Training modality: $mod"

    RESUME_ARG=""
    if [ -n "$RESUME_CKPT" ] && [ -f "$RESUME_CKPT" ]; then
        RESUME_ARG="--resume_checkpoint $RESUME_CKPT"
        echo "    Resuming from: $RESUME_CKPT"
    else
        # Auto-detect latest checkpoint for this modality
        local latest=$(ls -t "$CKPT_DIR"/brats_${mod}_*.pt 2>/dev/null | head -1)
        if [ -n "$latest" ]; then
            RESUME_ARG="--resume_checkpoint $latest"
            echo "    Auto-resuming from: $latest"
        fi
    fi

    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_ENTITY="${WANDB_ENTITY:-}" \
    CHECKPOINT_DIR="$CKPT_DIR" \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
    python scripts/train.py \
        --dataset=brats \
        --num_channels=32 \
        --class_cond=False \
        --num_res_blocks=2 \
        --num_heads=1 \
        --learn_sigma=False \
        --use_scale_shift_norm=False \
        --attention_resolutions= \
        --channel_mult=1,2,2,4,4 \
        --diffusion_steps=$TIMESTEPS \
        --sample_schedule=$STRATEGY \
        --noise_schedule=linear \
        --dims=3 \
        --batch_size=1 \
        --num_groups=32 \
        --in_channels=32 \
        --out_channels=8 \
        --lr=1e-5 \
        --lr_anneal_steps=600000 \
        --data_dir="$TRAIN_DIR" \
        --image_size=224 \
        --use_fp16=True \
        --save_interval=500 \
        --log_interval=100 \
        --num_workers=8 \
        --contr=$mod \
        $RESUME_ARG
}

if [ "$MODALITY" = "all" ]; then
    for mod in t1n t1c t2w t2f; do
        train_modality "$mod"
    done
else
    train_modality "$MODALITY"
fi

echo ""
echo "Training complete."
