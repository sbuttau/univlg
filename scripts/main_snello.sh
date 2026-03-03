#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Licensed under the LICENSE file in the root directory of this source tree.

#SBATCH --job-name=univlg-train-scanrefer-multi
#SBATCH --nodes=2
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10-00:00:00
#SBATCH --cpus-per-task=5
#SBATCH --output=logs/%x-%j-node%N.out
#SBATCH --error=logs/%x-%j-node%N.err
#SBATCH --mem=50G

get_cuda_device_count() {
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l
    else
        nvidia-smi --query-gpu=name --format=csv,noheader | wc -l
    fi
}

# ----------------------------
# Singularity container setup
# ----------------------------
SIF="$HOME/univlg/univlg.sif"
HOST_REPO="$HOME/univlg"

# Make sure SLURM binaries are visible inside container
export PATH=$PATH:/usr/bin

DIR="$(dirname "$PWD")"
export PYTHONPATH="$DIR:$DIR/pretrain"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

export CKPTS_PATH="ckpts"
export PRECOMPUTED_SCANNET_PATH="data/"
export DETECTRON2_DATASETS_2D="data/datasets_2d"
export DETECTRON2_DATASETS="data/SEMSEG_100k"
export REF_DATASET="data/refer_it_3d"

SCANNET_DATA_DIR="data/mask3d_processed/scannet/train_validation_database.yaml"
SCANNET200_DATA_DIR="data/mask3d_processed/scannet200/train_validation_database.yaml"
MATTERPORT_DATA_DIR="data/mask3d_processed/matterport/train_validation_database.yaml"
S3DIS_DATA_DIR="data/SEMSEG_100k/s3dis/train_validation_database.yaml"
OUTPUT_DIR_PREFIX="outputs/train"

NUM_GPUS=${NUM_GPUS:-$(get_cuda_device_count)}
BS=${BS:-2}
BS2D=${BS2D:-1}  # Default BS2D if not set
SAMPLING_FRAME_NUM=${SAMPLING_FRAME_NUM:-15}
SIDE_FRAMES=$(( (SAMPLING_FRAME_NUM - 1) / 2 ))
CHECKPOINT_PERIOD=${CHECKPOINT_PERIOD:-8000}
EVAL_PERIOD=${EVAL_PERIOD:-8000}
IGNORERUN=${IGNORERUN:-0}
NAME=${NAME:-"univlg"}
NUM_DATALOADERS=${NUM_DATALOADERS:-16}
NUM_VAL_DATALOADERS=${NUM_VAL_DATALOADERS:-4}
NUM_MACHINES=${NUM_MACHINES:-1}
BREAKPOINT_ON_ERROR=${BREAKPOINT_ON_ERROR:-False}
USE_STANDALONE=${USE_STANDALONE:-0}
export RETURN_SCENE_BATCH_SIZE=${RETURN_SCENE_BATCH_SIZE:-8}

USE_SLURM=${USE_SLURM:-1}
EVAL_ONLY=${EVAL_ONLY:-0}
RESUME=${RESUME:-1}
USE_SWIN=${USE_SWIN:-0}
USE_DINO=${USE_DINO:-1}
CKPT_PATH=${CKPT_PATH:-"${CKPTS_PATH}/misc/m2f_coco_swin.pth"}

TOTAL_BATCH=$((NUM_GPUS * NUM_MACHINES * BS))
TOTAL_BATCH_2D=$((NUM_GPUS * NUM_MACHINES * BS2D))
TOTAL_BATCH_3D=1

if [[ -f .env ]]; then
    echo "Initializing environment variables"
    source .env
fi

if [[ "$NAME" = "univlg" && "$EVAL_ONLY" -eq 1 ]]; then
    NAME="univlg_eval"
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
if [[ ! -d "$OUTPUT_DIR" ]]; then
    if [[ "$IGNORERUN" -eq 1 ]]; then
        OUTPUT_DIR="${OUTPUT_DIR_PREFIX}/debug/ignore_${TIMESTAMP}_${NAME}"
        BREAKPOINT_ON_ERROR=True
    elif [[ "$NAME" = "univlg" || "$NAME" = "univlg_eval" ]]; then
        OUTPUT_DIR="${OUTPUT_DIR_PREFIX}/${TIMESTAMP}_${NAME}"
    else
        OUTPUT_DIR="${OUTPUT_DIR_PREFIX}/${NAME}"
    fi
    echo "Using output dir: ${OUTPUT_DIR}"
else
    echo "Using existing output dir: ${OUTPUT_DIR}"
fi

if [[ "$USE_DINO" -eq 1 ]]; then
    CONFIG_FILE="univlg/configs/dinov2_3d.yaml"
elif [[ "$USE_SWIN" -eq 1 ]]; then
    CONFIG_FILE="univlg/configs/swin_3d.yaml"
else
    CONFIG_FILE="univlg/configs/3d.yaml"
fi

EVAL_ARG=""
if [[ "$EVAL_ONLY" -eq 1 ]]; then
    EVAL_ARG="--eval-only"
fi

RESUME_ARG=""
if [[ "$RESUME" -eq 1 ]]; then
    RESUME_ARG="--resume"
fi

if [[ "$USE_STANDALONE" -eq 1 ]]; then
    export PYTHONPATH="$PYTHONPATH:$PWD"
    PYTHON_FILE="scripts/standalone_eval.py"
else
    PYTHON_FILE="train.py"
fi

# ----------------------------
# Execute inside container
# ----------------------------
singularity exec --nv \
    --bind "$HOST_REPO:/workspaces" \
    "$SIF" bash -c "
        cd /workspaces
        source scripts/setup.sh
        python $PYTHON_FILE \
            --dist-url='tcp://127.0.0.1:$RANDOM' \
            --num-gpus $NUM_GPUS \
            --num-machines $NUM_MACHINES \
            --config-file $CONFIG_FILE $EVAL_ARG $RESUME_ARG \
            OUTPUT_DIR $OUTPUT_DIR \
            SOLVER.IMS_PER_BATCH $TOTAL_BATCH \
            SOLVER.CHECKPOINT_PERIOD $CHECKPOINT_PERIOD \
            TEST.EVAL_PERIOD $EVAL_PERIOD \
            INPUT.FRAME_LEFT $SIDE_FRAMES \
            INPUT.FRAME_RIGHT $SIDE_FRAMES \
            INPUT.SAMPLING_FRAME_NUM $SAMPLING_FRAME_NUM \
            INPUT.FRAME_LEFT_2D 0 \
            INPUT.FRAME_RIGHT_2D 0 \
            INPUT.SAMPLING_FRAME_NUM_2D 1 \
            MODEL.WEIGHTS $CKPT_PATH \
            SOLVER.BASE_LR 1e-4 \
            MODEL.CROSS_VIEW_CONTEXTUALIZE True \
            INPUT.CAMERA_DROP False \
            INPUT.STRONG_AUGS True \
            INPUT.COLOR_AUG False \
            MODEL.KNN 8 \
            INPUT.AUGMENT_3D True \
            INPUT.SAMPLE_CHUNK_AUG True \
            MODEL.MASK_FORMER.TRAIN_NUM_POINTS 50000 \
            INPUT.VOXELIZE True \
            MODEL.CROSS_VIEW_BACKBONE True \
            MODEL.CROSS_VIEW_NUM_LAYERS '[2,2,6,2]' \
            DO_TRILINEAR_INTERPOLATION True \
            INTERP_NEIGHBORS 8 \
            MODEL.SEM_SEG_HEAD.NUM_CLASSES 20 \
            MODEL.MASK_FORMER.TEST.SEMANTIC_ON True \
            SKIP_CLASSES None \
            MODEL.FREEZE_BACKBONE False \
            SOLVER.TEST_IMS_PER_BATCH $TOTAL_BATCH \
            DATALOADER.NUM_WORKERS $NUM_DATALOADERS \
            DATALOADER.TEST_NUM_WORKERS $NUM_VAL_DATALOADERS \
            SCANNET_DATA_DIR $SCANNET_DATA_DIR \
            SCANNET200_DATA_DIR $SCANNET200_DATA_DIR \
            MATTERPORT_DATA_DIR $MATTERPORT_DATA_DIR \
            S3DIS_DATA_DIR $S3DIS_DATA_DIR \
            SOLVER.IMS_PER_BATCH_2D $TOTAL_BATCH_2D \
            SOLVER.IMS_PER_BATCH_3D $TOTAL_BATCH_3D \
            BREAKPOINT_ON_ERROR $BREAKPOINT_ON_ERROR
"

# Requeue on timeout
if [[ $? == 124 ]]; then 
    scontrol requeue $SLURM_JOB_ID
fi