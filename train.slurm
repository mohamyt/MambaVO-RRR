#!/usr/bin/env bash
#SBATCH --job-name=mambavo-train
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --time=1-00:00:00
#SBATCH --output=%x_%j.out

#1 day is the max wall time on this cluster, so this
# will need resubmitting multiple times to finish
# --resume picks up from the latest checkpoint automatically, so just rerun the same:
#   sbatch train.slurm
#
# no need to pass --ckpt_dir yourself. train.py defaults it to the relative
# path "checkpoints", and we cd into PROJECT_DIR below before running train.py, so it always resolves to PROJECT_DIR/checkpoints, same place checkpoints get
# saved to in the first place. just make sure nobody runs train.py by hand from some other directory, or this breaks.
#
# to run an ablation instead of the full model, add --ablate <name> to the python call below.
# options: none, context_feature, geometric_feature, pnp, history_fusion, mamba, gru, grad_weight, history_balance
#
# needs v100 (or better) bec the work partition's rtx3080/rtx2080ti cards only have 10-11GB.

set -euo pipefail

PROJECT_DIR="/home/woody/iwi9/iwi9153h/MambaVO-Reproduction"
ENV_PREFIX="${PROJECT_DIR}/envs/mambavo"
export TORCH_HOME="${PROJECT_DIR}/torch_hub_cache"

module purge
module load python/3.12-conda
module load cuda/11.8.0

cd "${PROJECT_DIR}"
conda run --no-capture-output -p "${ENV_PREFIX}" python train.py \
    --datapath data/TartanAir \
    --resume \
    --log_dir runs_mambavo_xi \
    --ckpt_dir checkpoints_full_model \
    --ablate none \
