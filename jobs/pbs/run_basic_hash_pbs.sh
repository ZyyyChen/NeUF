#!/bin/bash -l
# Basic fixed-geometry, single-output HashGrid baseline.

#PBS -N neuf_basic_hash
#PBS -q gpu
#PBS -l walltime=23:59:00
#PBS -l nodes=1:ppn=16:gpus=1:gpu48
#PBS -l mem=128gb
#PBS -j oe

set -euo pipefail

REPO_DIR="${REPO_DIR:-/misc/raid/zchen/Code/NeUF}"
PYTHON_BIN="${PYTHON_BIN:-/home/zchen/.conda/envs/neuf/bin/python}"
DATASET_PATH="${DATASET_PATH:-${REPO_DIR}/data/cerebral_data/Pre_traitement_echo_v2/Recalage/Patient0/us_recal_original/baked_dataset_physical.pkl}"
RUN_ROOT="${RUN_ROOT:-${REPO_DIR}/runs/basic_hash_seed3407}"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

"${PYTHON_BIN}" -m neuf.main \
  --dataset "${DATASET_PATH}" \
  --encoding HASH \
  --field-head legacy_fixed_geometry \
  --intensity-activation identity \
  --training-mode Random \
  --points-per-iter 50000 \
  --nb-iters-max 20000 \
  --plot-freq 1000 \
  --save-freq 5000 \
  --seed 3407 \
  --root "${RUN_ROOT}" \
  --hash-n-levels 16 \
  --hash-n-features-per-level 2 \
  --hash-log2-hashmap-size 19 \
  --hash-base-resolution 16 \
  --hash-finest-resolution 256
