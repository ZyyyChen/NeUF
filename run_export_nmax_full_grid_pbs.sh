#!/bin/bash -l
#PBS -N nmax_full_grid
#PBS -q gpu
#PBS -l walltime=23:59:00
#PBS -l nodes=1:ppn=16:gpus=1:gpu48
#PBS -l mem=128gb
#PBS -j oe
#PBS -o /home/zchen/history/nmax_full_grid_exports.pbs.log
#PBS -M ziyi.chen@creatis.insa-lyon.fr
#PBS -m ae
#PBS -t 0-6

set -euo pipefail

REPO_DIR="${REPO_DIR:-/misc/raid/zchen/Code/NeUF}"
PYTHON_BIN="${PYTHON_BIN:-/home/zchen/.conda/envs/neuf/bin/python}"
SCRIPT_PATH="${REPO_DIR}/export_full_grid_from_ckpt.py"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${REPO_DIR}/experiments/hash_grid_nmax}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/exports/hash_grid_nmax_full_grid}"
LOG_DIR="${LOG_DIR:-/home/zchen/history/nmax_full_grid_exports}"

EXPORT_CHUNK_SIZE="${EXPORT_CHUNK_SIZE:-131072}"
EXPORT_RESOLUTION_SCALE="${EXPORT_RESOLUTION_SCALE:-1.0}"
EXPORT_SPACING="${EXPORT_SPACING:-}"
EXPORT_RECONS_COMMON_GRID_H5="${EXPORT_RECONS_COMMON_GRID_H5:-}"
EXPORT_USE_BBOX_MASK="${EXPORT_USE_BBOX_MASK:-0}"
EXPORT_DISABLE_SEQUENCE_PLANE_MASK="${EXPORT_DISABLE_SEQUENCE_PLANE_MASK:-0}"
EXPORT_SAVE_LARGE_NPY="${EXPORT_SAVE_LARGE_NPY:-0}"
EXPORT_SAVE_GT_EXPORTS="${EXPORT_SAVE_GT_EXPORTS:-0}"

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"
JOB_LOG="${LOG_DIR}/${PBS_JOBID:-manual}.log"
exec > >(tee -a "${JOB_LOG}") 2>&1

cd "${REPO_DIR}"

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

ARRAY_ID="${PBS_ARRAYID:-${PBS_ARRAY_INDEX:-${EXPERIMENT_INDEX:-0}}}"
MANIFEST_PATH="${OUTPUT_ROOT}/export_manifest_${PBS_JOBID:-manual}_${ARRAY_ID}.csv"
printf "experiment,ckpt,output_root,latest_output,start_time,end_time\n" > "${MANIFEST_PATH}"

EXPERIMENTS=(
  "exp1_A_L16_Nmax512"
  "exp1_B_L16_Nmax256"
  "exp1_C_L16_Nmax128"
  "exp1_D_L16_Nmax64"
  "exp1_E_L16_Nmax32"
  "exp2_F_L12_Nmax512"
  "exp2_G_L8_Nmax512"
)

if [[ "${ARRAY_ID}" -lt 0 || "${ARRAY_ID}" -ge "${#EXPERIMENTS[@]}" ]]; then
  echo "Invalid ARRAY_ID=${ARRAY_ID}; expected 0-$(( ${#EXPERIMENTS[@]} - 1 ))"
  exit 1
fi

experiment="${EXPERIMENTS[${ARRAY_ID}]}"
experiment_dir="${EXPERIMENT_ROOT}/${experiment}"
ckpt_path="${experiment_dir}/${experiment}_latest.pkl"
if [[ ! -f "${ckpt_path}" ]]; then
  ckpt_path="${experiment_dir}/latest/ckpt.pkl"
fi
if [[ ! -f "${ckpt_path}" ]]; then
  echo "Checkpoint not found for ${experiment}"
  echo "Tried:"
  echo "  ${experiment_dir}/${experiment}_latest.pkl"
  echo "  ${experiment_dir}/latest/ckpt.pkl"
  exit 1
fi

experiment_output_root="${OUTPUT_ROOT}/${experiment}"
mkdir -p "${experiment_output_root}"

start_time="$(date --iso-8601=seconds)"

echo "Host: $(hostname)"
echo "Date: ${start_time}"
echo "PBS job id: ${PBS_JOBID:-n/a}"
echo "PBS array id: ${ARRAY_ID}"
echo "PBS queue: ${PBS_QUEUE:-n/a}"
echo "Working dir: ${REPO_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "Export script: ${SCRIPT_PATH}"
echo "Experiment: ${experiment}"
echo "Checkpoint: ${ckpt_path}"
echo "Output root: ${experiment_output_root}"
echo "Threads: ${THREADS}"
echo "Chunk size: ${EXPORT_CHUNK_SIZE}"
echo "Resolution scale: ${EXPORT_RESOLUTION_SCALE}"
echo "Spacing override: ${EXPORT_SPACING:-default}"
echo "Recons common grid h5: ${EXPORT_RECONS_COMMON_GRID_H5:-none}"
echo "Use bbox mask: ${EXPORT_USE_BBOX_MASK}"
echo "Disable sequence plane mask: ${EXPORT_DISABLE_SEQUENCE_PLANE_MASK}"
echo "Save large npy: ${EXPORT_SAVE_LARGE_NPY}"
echo "Save GT exports: ${EXPORT_SAVE_GT_EXPORTS}"
echo "Manifest: ${MANIFEST_PATH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-n/a}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found on this node."
fi

cmd=(
  "${PYTHON_BIN}" "${SCRIPT_PATH}"
  --ckpt "${ckpt_path}"
  --output "${experiment_output_root}"
  --chunk-size "${EXPORT_CHUNK_SIZE}"
  --resolution-scale "${EXPORT_RESOLUTION_SCALE}"
)

if [[ -n "${EXPORT_SPACING}" ]]; then
  read -r -a spacing_vals <<< "${EXPORT_SPACING}"
  cmd+=(--spacing "${spacing_vals[@]}")
fi
if [[ -n "${EXPORT_RECONS_COMMON_GRID_H5}" ]]; then
  cmd+=(--recons-common-grid-h5 "${EXPORT_RECONS_COMMON_GRID_H5}")
fi
if [[ "${EXPORT_USE_BBOX_MASK}" == "1" ]]; then
  cmd+=(--use-bbox-mask)
fi
if [[ "${EXPORT_DISABLE_SEQUENCE_PLANE_MASK}" == "1" ]]; then
  cmd+=(--disable-sequence-plane-mask)
fi
if [[ "${EXPORT_SAVE_LARGE_NPY}" == "1" ]]; then
  cmd+=(--save-large-npy)
fi
if [[ "${EXPORT_SAVE_GT_EXPORTS}" == "1" ]]; then
  cmd+=(--save-gt-exports)
fi

config_path="${experiment_output_root}/export_config_${PBS_JOBID:-manual}_${ARRAY_ID}.txt"
{
  echo "experiment=${experiment}"
  echo "ckpt=${ckpt_path}"
  echo "output_root=${experiment_output_root}"
  echo "chunk_size=${EXPORT_CHUNK_SIZE}"
  echo "resolution_scale=${EXPORT_RESOLUTION_SCALE}"
  echo "spacing=${EXPORT_SPACING:-default}"
  echo "recons_common_grid_h5=${EXPORT_RECONS_COMMON_GRID_H5:-none}"
  echo "use_bbox_mask=${EXPORT_USE_BBOX_MASK}"
  echo "disable_sequence_plane_mask=${EXPORT_DISABLE_SEQUENCE_PLANE_MASK}"
  echo "save_large_npy=${EXPORT_SAVE_LARGE_NPY}"
  echo "save_gt_exports=${EXPORT_SAVE_GT_EXPORTS}"
  echo "start_time=${start_time}"
  printf "command="
  printf " %q" "${cmd[@]}"
  printf "\n"
} > "${config_path}"

printf "Command:"
printf " %q" "${cmd[@]}"
printf "\n"

time "${cmd[@]}"

end_time="$(date --iso-8601=seconds)"
latest_output="$(
  find "${experiment_output_root}" -mindepth 2 -maxdepth 2 -type d -printf "%T@ %p\n" 2>/dev/null \
    | sort -n \
    | tail -1 \
    | cut -d " " -f 2-
)"
echo "end_time=${end_time}" >> "${config_path}"
echo "latest_output=${latest_output}" >> "${config_path}"

printf "%s,%s,%s,%s,%s,%s\n" \
  "${experiment}" \
  "${ckpt_path}" \
  "${experiment_output_root}" \
  "${latest_output}" \
  "${start_time}" \
  "${end_time}" >> "${MANIFEST_PATH}"

echo "Finished ${experiment} export at: ${end_time}"
echo "Latest output: ${latest_output}"
