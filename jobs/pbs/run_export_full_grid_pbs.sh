#!/bin/bash -l
# Cartesian export for a retained HASH/DUAL_HASH checkpoint.

#PBS -N neuf_export
#PBS -q gpu
#PBS -l walltime=08:00:00
#PBS -l nodes=1:ppn=8:gpus=1:gpu48
#PBS -l mem=64gb
#PBS -j oe

set -euo pipefail

REPO_DIR="${REPO_DIR:-/misc/raid/zchen/Code/NeUF}"
PYTHON_BIN="${PYTHON_BIN:-/home/zchen/.conda/envs/neuf/bin/python}"
CHECKPOINT="${CHECKPOINT:-${REPO_DIR}/runs/basic_hash_seed3407/latest/ckpt.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/exports/current_hash}"
ALPHA="${ALPHA:-1.0}"
COMPONENT="${COMPONENT:-intensity}"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

"${PYTHON_BIN}" -m neuf.export_full_grid_from_ckpt \
  --ckpt "${CHECKPOINT}" \
  --output "${OUTPUT_DIR}" \
  --alpha "${ALPHA}" \
  --component "${COMPONENT}" \
  --save-float-output
