#!/bin/bash -l
#PBS -N neuf_render_3d
#PBS -q gpu
#PBS -l walltime=23:59:00
#PBS -l nodes=1:ppn=16:gpus=1:gpu48
#PBS -l mem=128gb
#PBS -j oe
#PBS -o /home/zchen/history/neuf_render_3d_volume.pbs.log
#PBS -M ziyi.chen@creatis.insa-lyon.fr
#PBS -m ae
#PBS -t 0-3

set -euo pipefail

REPO_DIR="${REPO_DIR:-/misc/raid/zchen/Code/NeUF}"
PYTHON_BIN="${PYTHON_BIN:-/home/zchen/.conda/envs/neuf/bin/python}"
SCRIPT_PATH="${SCRIPT_PATH:-${REPO_DIR}/render_3d_volume_from_ckpt.py}"
CKPT_DIR="${CKPT_DIR:-/home/zchen/Code/NeUF/logs/23-04-2026/DUAL_HASH_Patient0_9/checkpoints}"
CKPT_STEPS="${CKPT_STEPS:-2000 5000 8000 10000}"
CKPT_PATH="${CKPT_PATH:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/exports/render_3d_volume}"
RENDER_DATE="${RENDER_DATE:-$(date +%d-%m-%Y)}"
CKPT_GROUP="${CKPT_GROUP:-}"
CKPT_GROUP_PREFIX="${CKPT_GROUP_PREFIX:-ckpt}"
LOG_DIR="${LOG_DIR:-/home/zchen/history/neuf_render_3d_volume}"

RENDER_CHUNK_SIZE="${RENDER_CHUNK_SIZE:-131072}"
RENDER_RESOLUTION_SCALE="${RENDER_RESOLUTION_SCALE:-1.0}"
RENDER_SPACING="${RENDER_SPACING:-}"
RENDER_POINT_MIN="${RENDER_POINT_MIN:-}"
RENDER_POINT_MAX="${RENDER_POINT_MAX:-}"
RENDER_USE_BBOX_MASK="${RENDER_USE_BBOX_MASK:-0}"
RENDER_TRAINING_PROGRESS="${RENDER_TRAINING_PROGRESS:-}"
RENDER_NB_ITERS_MAX="${RENDER_NB_ITERS_MAX:-10000}"
RENDER_OUTPUT_TYPE="${RENDER_OUTPUT_TYPE:-uint8}"
RENDER_SAVE_VOLUME_NPY="${RENDER_SAVE_VOLUME_NPY:-0}"

ARRAY_ID="${PBS_ARRAYID:-${PBS_ARRAY_INDEX:-${CKPT_INDEX:-0}}}"

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"
JOB_LOG="${LOG_DIR}/${PBS_JOBID:-manual}_${ARRAY_ID}.log"
exec > >(tee -a "${JOB_LOG}") 2>&1

cd "${REPO_DIR}"

read -r -a ckpt_steps <<< "${CKPT_STEPS}"
if ! [[ "${ARRAY_ID}" =~ ^[0-9]+$ ]]; then
  echo "Invalid ARRAY_ID=${ARRAY_ID}; expected 0-$(( ${#ckpt_steps[@]} - 1 ))"
  exit 1
fi

if [[ -z "${CKPT_PATH}" ]]; then
  if [[ "${ARRAY_ID}" -ge "${#ckpt_steps[@]}" ]]; then
    echo "Invalid ARRAY_ID=${ARRAY_ID}; expected 0-$(( ${#ckpt_steps[@]} - 1 ))"
    exit 1
  fi
  CKPT_STEP="${ckpt_steps[${ARRAY_ID}]}"
  CKPT_PATH="${CKPT_DIR}/${CKPT_STEP}.pkl"
else
  CKPT_STEP="$(basename "${CKPT_PATH}")"
  CKPT_STEP="${CKPT_STEP%.pkl}"
fi

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "Checkpoint not found: ${CKPT_PATH}"
  exit 1
fi

DATE_OUTPUT_ROOT="${OUTPUT_ROOT}/${RENDER_DATE}"
mkdir -p "${DATE_OUTPUT_ROOT}"

PBS_JOB_KEY="${PBS_JOBID:-manual}"
PBS_JOB_KEY="${PBS_JOB_KEY%%.*}"
PBS_JOB_KEY="${PBS_JOB_KEY%%[*}"
if [[ -z "${CKPT_GROUP}" ]]; then
  group_marker="${DATE_OUTPUT_ROOT}/.ckpt_group_${PBS_JOB_KEY}"
  lock_dir="${DATE_OUTPUT_ROOT}/.ckpt_group_lock"
  lock_wait_seconds=0
  until mkdir "${lock_dir}" 2>/dev/null; do
    sleep 1
    lock_wait_seconds=$((lock_wait_seconds + 1))
    if [[ "${lock_wait_seconds}" -ge 300 ]]; then
      echo "Timed out waiting for checkpoint group lock: ${lock_dir}"
      exit 1
    fi
  done

  if [[ -f "${group_marker}" ]]; then
    CKPT_GROUP="$(<"${group_marker}")"
  else
    ckpt_group_idx=0
    while [[ -e "${DATE_OUTPUT_ROOT}/${CKPT_GROUP_PREFIX}_${ckpt_group_idx}" ]]; do
      ckpt_group_idx=$((ckpt_group_idx + 1))
    done
    CKPT_GROUP="${CKPT_GROUP_PREFIX}_${ckpt_group_idx}"
    mkdir -p "${DATE_OUTPUT_ROOT}/${CKPT_GROUP}"
    printf "%s\n" "${CKPT_GROUP}" > "${group_marker}"
  fi

  rmdir "${lock_dir}"
fi

CHECKPOINT_OUTPUT_ROOT="${DATE_OUTPUT_ROOT}/${CKPT_GROUP}/${CKPT_STEP}"
mkdir -p "${CHECKPOINT_OUTPUT_ROOT}"

THREADS="${PBS_NP:-16}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"
export VECLIB_MAXIMUM_THREADS="${THREADS}"
export BLIS_NUM_THREADS="${THREADS}"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export KMP_AFFINITY=granularity=fine,compact,1,0

cmd=(
  "${PYTHON_BIN}" "${SCRIPT_PATH}"
  --ckpt "${CKPT_PATH}"
  --output "${CHECKPOINT_OUTPUT_ROOT}"
  --output-exact
  --chunk-size "${RENDER_CHUNK_SIZE}"
  --resolution-scale "${RENDER_RESOLUTION_SCALE}"
  --nb-iters-max "${RENDER_NB_ITERS_MAX}"
  --output-type "${RENDER_OUTPUT_TYPE}"
)

if [[ -n "${RENDER_SPACING}" ]]; then
  read -r -a spacing_vals <<< "${RENDER_SPACING}"
  if [[ "${#spacing_vals[@]}" -ne 1 && "${#spacing_vals[@]}" -ne 3 ]]; then
    echo "RENDER_SPACING expects 1 value or 3 values, got: ${RENDER_SPACING}"
    exit 1
  fi
  cmd+=(--spacing "${spacing_vals[@]}")
fi
if [[ -n "${RENDER_POINT_MIN}" ]]; then
  read -r -a point_min_vals <<< "${RENDER_POINT_MIN}"
  if [[ "${#point_min_vals[@]}" -ne 3 ]]; then
    echo "RENDER_POINT_MIN expects 3 values, got: ${RENDER_POINT_MIN}"
    exit 1
  fi
  cmd+=(--point-min "${point_min_vals[@]}")
fi
if [[ -n "${RENDER_POINT_MAX}" ]]; then
  read -r -a point_max_vals <<< "${RENDER_POINT_MAX}"
  if [[ "${#point_max_vals[@]}" -ne 3 ]]; then
    echo "RENDER_POINT_MAX expects 3 values, got: ${RENDER_POINT_MAX}"
    exit 1
  fi
  cmd+=(--point-max "${point_max_vals[@]}")
fi
if [[ "${RENDER_USE_BBOX_MASK}" == "1" ]]; then
  cmd+=(--use-bbox-mask)
fi
if [[ -n "${RENDER_TRAINING_PROGRESS}" ]]; then
  cmd+=(--training-progress "${RENDER_TRAINING_PROGRESS}")
fi
if [[ "${RENDER_SAVE_VOLUME_NPY}" == "1" ]]; then
  cmd+=(--save-volume-npy)
fi

start_time="$(date --iso-8601=seconds)"
manifest_path="${CHECKPOINT_OUTPUT_ROOT}/render_manifest_${PBS_JOBID:-manual}_${ARRAY_ID}.csv"
config_path="${CHECKPOINT_OUTPUT_ROOT}/render_config_${PBS_JOBID:-manual}_${ARRAY_ID}.txt"
printf "ckpt,output_root,latest_output,start_time,end_time\n" > "${manifest_path}"
{
  echo "script=${SCRIPT_PATH}"
  echo "array_id=${ARRAY_ID}"
  echo "ckpt_dir=${CKPT_DIR}"
  echo "ckpt_steps=${CKPT_STEPS}"
  echo "ckpt_step=${CKPT_STEP}"
  echo "ckpt=${CKPT_PATH}"
  echo "render_date=${RENDER_DATE}"
  echo "ckpt_group=${CKPT_GROUP}"
  echo "output_root=${CHECKPOINT_OUTPUT_ROOT}"
  echo "chunk_size=${RENDER_CHUNK_SIZE}"
  echo "resolution_scale=${RENDER_RESOLUTION_SCALE}"
  echo "spacing=${RENDER_SPACING:-default}"
  echo "point_min=${RENDER_POINT_MIN:-default}"
  echo "point_max=${RENDER_POINT_MAX:-default}"
  echo "use_bbox_mask=${RENDER_USE_BBOX_MASK}"
  echo "training_progress=${RENDER_TRAINING_PROGRESS:-infer_from_checkpoint}"
  echo "nb_iters_max=${RENDER_NB_ITERS_MAX}"
  echo "output_type=${RENDER_OUTPUT_TYPE}"
  echo "save_volume_npy=${RENDER_SAVE_VOLUME_NPY}"
  echo "start_time=${start_time}"
  printf "command="
  printf " %q" "${cmd[@]}"
  printf "\n"
} > "${config_path}"

echo "Host: $(hostname)"
echo "Date: ${start_time}"
echo "PBS job id: ${PBS_JOBID:-n/a}"
echo "PBS array id: ${ARRAY_ID}"
echo "PBS queue: ${PBS_QUEUE:-n/a}"
echo "Working dir: ${REPO_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "Script: ${SCRIPT_PATH}"
echo "Checkpoint dir: ${CKPT_DIR}"
echo "Checkpoint step: ${CKPT_STEP}"
echo "Checkpoint: ${CKPT_PATH}"
echo "Base output root: ${OUTPUT_ROOT}"
echo "Render date: ${RENDER_DATE}"
echo "Checkpoint group: ${CKPT_GROUP}"
echo "Output root: ${CHECKPOINT_OUTPUT_ROOT}"
echo "Threads: ${THREADS}"
echo "Config: ${config_path}"
echo "Manifest: ${manifest_path}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-n/a}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found on this node."
fi

printf "Command:"
printf " %q" "${cmd[@]}"
printf "\n"

time "${cmd[@]}"

end_time="$(date --iso-8601=seconds)"
latest_output="${CHECKPOINT_OUTPUT_ROOT}"
echo "end_time=${end_time}" >> "${config_path}"
echo "latest_output=${latest_output}" >> "${config_path}"

printf "%s,%s,%s,%s,%s\n" \
  "${CKPT_PATH}" \
  "${CHECKPOINT_OUTPUT_ROOT}" \
  "${latest_output}" \
  "${start_time}" \
  "${end_time}" >> "${manifest_path}"

echo "Finished render_3d_volume_from_ckpt.py at: ${end_time}"
echo "Latest output: ${latest_output}"
