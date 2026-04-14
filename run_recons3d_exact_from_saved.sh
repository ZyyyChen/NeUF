#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/zchen/Code/NeUF}"
PYTHON_BIN="${PYTHON_BIN:-/home/zchen/.conda/envs/neuf/bin/python}"
SCRIPT_PATH="${REPO_DIR}/recons3d_exact_from_saved.py"

RECONS_POINTS_MAT="${RECONS_POINTS_MAT:-${REPO_DIR}/data/cerebral_data/Pre_traitement_echo_v2/Reconstruction_3D/Patient0/data_3D_Patient0_J35_2_reconstruction_points.mat}"
RECONS_SAGPLAN_MAT="${RECONS_SAGPLAN_MAT:-${REPO_DIR}/data/cerebral_data/Pre_traitement_echo_v2/Reconstruction_3D/Patient0/data_3D_Patient0_J35_2_sagplan_params.mat}"
DATA_RECAL_MAT="${DATA_RECAL_MAT:-${REPO_DIR}/data/cerebral_data/Pre_traitement_echo_v2/Recalage/Patient0/data_recal_Patient0_J35_2_d_0.5.mat}"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/exports/recons3d_exact_from_saved}"
DELTA_X_CC="${DELTA_X_CC:-1}"
DELTA_X_SEQDYN="${DELTA_X_SEQDYN:-1}"
DATASET_PKL="${DATASET_PKL:-}"
SAVE_DEBUG_NPY="${SAVE_DEBUG_NPY:-1}"

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_DIR}"

cmd=(
  "${PYTHON_BIN}" "${SCRIPT_PATH}"
  --recons-points-mat "${RECONS_POINTS_MAT}"
  --recons-sagplan-mat "${RECONS_SAGPLAN_MAT}"
  --data-recal-mat "${DATA_RECAL_MAT}"
  --delta-x-cc "${DELTA_X_CC}"
  --delta-x-seqdyn "${DELTA_X_SEQDYN}"
  --output "${OUTPUT_DIR}"
)

if [[ -n "${DATASET_PKL}" ]]; then
  cmd+=(--dataset-pkl "${DATASET_PKL}")
fi

if [[ "${SAVE_DEBUG_NPY}" == "1" ]]; then
  cmd+=(--save-debug-npy)
fi

echo "Host: $(hostname)"
echo "Date: $(date --iso-8601=seconds)"
echo "Working dir: ${REPO_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "recons_points_mat: ${RECONS_POINTS_MAT}"
echo "recons_sagplan_mat: ${RECONS_SAGPLAN_MAT}"
echo "data_recal_mat: ${DATA_RECAL_MAT}"
echo "Output base dir: ${OUTPUT_DIR}"
echo "delta_x_cc: ${DELTA_X_CC}"
echo "delta_x_seqdyn: ${DELTA_X_SEQDYN}"
echo "save_debug_npy: ${SAVE_DEBUG_NPY}"
if [[ -n "${DATASET_PKL}" ]]; then
  echo "dataset_pkl: ${DATASET_PKL}"
fi

printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

time "${cmd[@]}"
