#!/bin/bash
set -euo pipefail

CONFIG_FILE="$1"
source "${CONFIG_FILE}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${THREADS_PER_RANK}"
export OPENBLAS_NUM_THREADS="${THREADS_PER_RANK}"
export MKL_NUM_THREADS="${THREADS_PER_RANK}"
export NUMEXPR_NUM_THREADS="${THREADS_PER_RANK}"
export VECLIB_MAXIMUM_THREADS="${THREADS_PER_RANK}"
export BLIS_NUM_THREADS="${THREADS_PER_RANK}"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export KMP_AFFINITY=granularity=fine,compact,1,0

cd "${REPO_DIR}"

RANK="${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-0}}"
WORLD="${OMPI_COMM_WORLD_SIZE:-${PMI_SIZE:-1}}"
RANK_DIR="${SHARD_ROOT}/rank_${RANK}"

cmd=(
  "${PYTHON_BIN}" "${SCRIPT_PATH}"
  --ckpt "${CKPT_PATH}"
  --output "${OUTPUT_DIR}"
  --exact-output-dir "${RANK_DIR}"
  --shard-rank "${RANK}"
  --shard-world-size "${WORLD}"
  --query-workers "${THREADS_PER_RANK}"
  --chunk-size "${KNN_CHUNK_SIZE}"
  --k "${KNN_K}"
  --resolution-scale "${KNN_RESOLUTION_SCALE}"
)

if [[ -n "${KNN_MAX_DIST}" ]]; then
  cmd+=(--max-dist "${KNN_MAX_DIST}")
fi
if [[ -n "${KNN_SPACING}" ]]; then
  read -r -a spacing_vals <<< "${KNN_SPACING}"
  cmd+=(--spacing "${spacing_vals[@]}")
fi
if [[ -n "${KNN_BOUNDS_DATASET}" ]]; then
  cmd+=(--bounds-dataset "${KNN_BOUNDS_DATASET}")
fi
if [[ -n "${KNN_POINT_MIN}" ]]; then
  read -r -a point_min_vals <<< "${KNN_POINT_MIN}"
  cmd+=(--point-min "${point_min_vals[@]}")
fi
if [[ -n "${KNN_POINT_MAX}" ]]; then
  read -r -a point_max_vals <<< "${KNN_POINT_MAX}"
  cmd+=(--point-max "${point_max_vals[@]}")
fi
if [[ "${KNN_USE_BBOX_MASK}" == "1" ]]; then
  cmd+=(--use-bbox-mask)
fi
if [[ "${KNN_DISABLE_SEQUENCE_PLANE_MASK}" == "1" ]]; then
  cmd+=(--disable-sequence-plane-mask)
fi

echo "Rank ${RANK}/${WORLD}: output -> ${RANK_DIR}"
printf 'Rank %s command:' "${RANK}"
printf ' %q' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
