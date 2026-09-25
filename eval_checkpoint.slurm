#!/usr/bin/env bash
#SBATCH --job-name=mambavo-eval-checkpoint
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out

# General checkpoint eval for EuRoC and KITTI ATE
#from what i tested, it usually runs for about 16 hours for the entire evaluation (both datasets), so 24 hours should be enough for most checkpoints.

# usage: 'sbatch eval_checkpoint.slurm <checkpoint_path> <label>'
#   checkpoint_path: relative to PROJECT_DIR
#   label: just for output log filenames (eval_<label>_euroc.log / _kitti.log)

set -uo pipefail

CKPT="${1:?usage: sbatch eval_checkpoint.slurm <checkpoint_path> <label>}"
LABEL="${2:?usage: sbatch eval_checkpoint.slurm <checkpoint_path> <label>}"

PROJECT_DIR="/home/woody/iwi9/iwi9153h/MambaVO-Reproduction"
ENV_PREFIX="${PROJECT_DIR}/envs/mambavo"
export TORCH_HOME="${PROJECT_DIR}/torch_hub_cache"
export PYTHONUNBUFFERED=1

module purge
module load python/3.12-conda
module load cuda/11.8.0

cd "${PROJECT_DIR}"

conda run --no-capture-output -p "${ENV_PREFIX}" python -u evaluate.py \
    --dataset euroc --datapath EuRoC \
    --checkpoint "${CKPT}" \
    > "eval_${LABEL}_euroc.log" 2>&1 &
EUROC_PID=$!

conda run --no-capture-output -p "${ENV_PREFIX}" python -u evaluate.py \
    --dataset kitti --datapath KITTI \
    --checkpoint "${CKPT}" \
    > "eval_${LABEL}_kitti.log" 2>&1 &
KITTI_PID=$!

wait $EUROC_PID
wait $KITTI_PID
