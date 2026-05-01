#!/usr/bin/env bash
# Run REAP pruning on Strix Halo with sensible defaults.
#
# Defaults to Nemotron-3 Nano 30B-A3B with CPU device map (fits the 128 GB UMA
# but the iGPU compiler is unreliable on 60 GB allocations). Override anything
# via env vars or positional args; this script just forwards to the upstream
# experiments/pruning-cli.sh after sanity-checking the bootstrap state.
#
# Usage:
#     bash run.sh                     # defaults below
#     MODEL=Qwen/Qwen3-30B-A3B bash run.sh
#     CUDA_DEVICES=0 bash run.sh      # for the smaller MoEs that fit on the iGPU

set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REAP_DIR="${REAP_DIR:-${ROOT}/reap}"

if [ ! -d "${REAP_DIR}" ]; then
    echo "error: ${REAP_DIR} not found. run bootstrap.sh first."
    exit 1
fi

CUDA_DEVICES="${CUDA_DEVICES:-cpu}"   # "cpu" => --device-map cpu inside the script (override the launcher to skip CUDA visibility)
MODEL="${MODEL:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"
METHOD="${METHOD:-reap}"
SEED="${SEED:-17}"
COMPRESSION="${COMPRESSION:-0.25}"
DATASET="${DATASET:-theblackcat102/evol-codealpaca-v1}"

# Eval flags. Default to false for the long ones; flip to true to actually
# benchmark a pruned checkpoint.
RUN_LM_EVAL="${RUN_LM_EVAL:-false}"
RUN_EVALPLUS="${RUN_EVALPLUS:-false}"
RUN_LCB="${RUN_LCB:-false}"
RUN_MATH="${RUN_MATH:-false}"
RUN_WILDBENCH="${RUN_WILDBENCH:-false}"

echo "[run] model=${MODEL}"
echo "[run] method=${METHOD} compression=${COMPRESSION} seed=${SEED}"
echo "[run] dataset=${DATASET}"
echo "[run] cuda_devices=${CUDA_DEVICES}"

cd "${REAP_DIR}"
exec bash experiments/pruning-cli.sh \
    "${CUDA_DEVICES}" "${MODEL}" "${METHOD}" "${SEED}" "${COMPRESSION}" "${DATASET}" \
    "${RUN_LM_EVAL}" "${RUN_EVALPLUS}" "${RUN_LCB}" "${RUN_MATH}" "${RUN_WILDBENCH}"
