#!/bin/bash -l
#PBS -N neuf_train_export
#PBS -q gpu
#PBS -l walltime=23:59:00
#PBS -l nodes=1:ppn=16:gpus=1:gpu48
#PBS -l mem=128gb
#PBS -j oe
#PBS -o /home/zchen/history/neuf_train_export.pbs.log
#PBS -M ziyi.chen@creatis.insa-lyon.fr
#PBS -m ae
#PBS -t 0-6

set -euo pipefail

REPO_DIR="${REPO_DIR:-/misc/raid/zchen/Code/NeUF}"
PYTHON_BIN="${PYTHON_BIN:-/home/zchen/.conda/envs/neuf/bin/python}"
TRAIN_SCRIPT="${REPO_DIR}/main.py"
EXPORT_SCRIPT="${REPO_DIR}/export_full_grid_from_ckpt.py"
DATASET_PATH="${DATASET_PATH:-${REPO_DIR}/data/cerebral_data/Pre_traitement_echo_v2/Recalage/Patient0/us_recal_original/baked_dataset.pkl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/experiments/hash_grid_nmax}"
EXPORT_OUTPUT_ROOT="${EXPORT_OUTPUT_ROOT:-${REPO_DIR}/exports/hash_grid_nmax_full_grid}"
LOG_DIR="${LOG_DIR:-/home/zchen/history/neuf_train_export}"

# --- Training hyperparameters ---
NB_ITERS_MAX="${NB_ITERS_MAX:-8500}"
PLOT_FREQ="${PLOT_FREQ:-100}"
SAVE_FREQ="${SAVE_FREQ:-100}"
POINTS_PER_ITER="${POINTS_PER_ITER:-50000}"
SEED="${SEED:-19981708}"
TRAINING_MODE="${TRAINING_MODE:-Random}"

HASH_N_MIN="${HASH_N_MIN:-16}"
HASH_FEATURES_PER_LEVEL="${HASH_FEATURES_PER_LEVEL:-2}"
HASH_LOG2_HASHMAP_SIZE="${HASH_LOG2_HASHMAP_SIZE:-19}"

RAW_DATASET="${RAW_DATASET:-0}"
JITTER_TRAINING="${JITTER_TRAINING:-0}"

# --- Export parameters ---
EXPORT_CHUNK_SIZE="${EXPORT_CHUNK_SIZE:-131072}"
EXPORT_RESOLUTION_SCALE="${EXPORT_RESOLUTION_SCALE:-1.0}"
EXPORT_SPACING="${EXPORT_SPACING:-}"
EXPORT_RECONS_COMMON_GRID_H5="${EXPORT_RECONS_COMMON_GRID_H5:-}"
EXPORT_USE_BBOX_MASK="${EXPORT_USE_BBOX_MASK:-0}"
EXPORT_DISABLE_SEQUENCE_PLANE_MASK="${EXPORT_DISABLE_SEQUENCE_PLANE_MASK:-0}"
EXPORT_SAVE_LARGE_NPY="${EXPORT_SAVE_LARGE_NPY:-0}"
EXPORT_SAVE_GT_EXPORTS="${EXPORT_SAVE_GT_EXPORTS:-0}"

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}" "${EXPORT_OUTPUT_ROOT}"
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
TRAIN_MANIFEST="${OUTPUT_ROOT}/experiment_manifest_${PBS_JOBID:-manual}_${ARRAY_ID}.csv"
EXPORT_MANIFEST="${EXPORT_OUTPUT_ROOT}/export_manifest_${PBS_JOBID:-manual}_${ARRAY_ID}.csv"
printf "experiment,description,hash_n_levels,hash_n_min,hash_n_max,per_level_scale,root,latest_checkpoint,named_checkpoint,start_time,end_time\n" > "${TRAIN_MANIFEST}"
printf "experiment,ckpt,output_root,latest_output,start_time,end_time\n" > "${EXPORT_MANIFEST}"

echo "Host: $(hostname)"
echo "Date: $(date --iso-8601=seconds)"
echo "PBS job id: ${PBS_JOBID:-n/a}"
echo "PBS array id: ${ARRAY_ID}"
echo "PBS queue: ${PBS_QUEUE:-n/a}"
echo "Working dir: ${REPO_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "Dataset: ${DATASET_PATH}"
echo "Train output root: ${OUTPUT_ROOT}"
echo "Export output root: ${EXPORT_OUTPUT_ROOT}"
echo "Threads: ${THREADS}"
echo "Iterations: ${NB_ITERS_MAX}"
echo "Points per iter: ${POINTS_PER_ITER}"
echo "Train manifest: ${TRAIN_MANIFEST}"
echo "Export manifest: ${EXPORT_MANIFEST}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-n/a}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found on this node."
fi

main_flags=()
if [[ "${RAW_DATASET}" == "1" ]]; then
  main_flags+=(--raw-dataset)
fi
if [[ "${JITTER_TRAINING}" == "1" ]]; then
  main_flags+=(--jitter-training)
fi

run_training() {
  local experiment="$1"
  local description="$2"
  local hash_n_levels="$3"
  local hash_n_max="$4"
  local experiment_root="${OUTPUT_ROOT}/${experiment}"
  local start_time
  local end_time
  local per_level_scale
  local latest_checkpoint="${experiment_root}/latest/ckpt.pkl"
  local named_checkpoint="${experiment_root}/${experiment}_latest.pkl"
  local config_path="${experiment_root}/experiment_config.txt"

  mkdir -p "${experiment_root}"
  start_time="$(date --iso-8601=seconds)"
  per_level_scale="$(
    awk \
      -v n_min="${HASH_N_MIN}" \
      -v n_max="${hash_n_max}" \
      -v levels="${hash_n_levels}" \
      'BEGIN { printf "%.3f", exp(log(n_max / n_min) / (levels - 1)) }'
  )"

  echo ""
  echo "============================================================"
  echo "[TRAIN] Experiment: ${experiment}"
  echo "Description: ${description}"
  echo "HashGrid: L=${hash_n_levels}, N_min=${HASH_N_MIN}, N_max=${hash_n_max}, per_level_scale=${per_level_scale}"
  echo "Root: ${experiment_root}"
  echo "Start: ${start_time}"
  echo "============================================================"

  {
    echo "experiment=${experiment}"
    echo "description=${description}"
    echo "hash_n_levels=${hash_n_levels}"
    echo "hash_n_min=${HASH_N_MIN}"
    echo "hash_n_max=${hash_n_max}"
    echo "per_level_scale=${per_level_scale}"
    echo "hash_features_per_level=${HASH_FEATURES_PER_LEVEL}"
    echo "hash_log2_hashmap_size=${HASH_LOG2_HASHMAP_SIZE}"
    echo "training_mode=${TRAINING_MODE}"
    echo "nb_iters_max=${NB_ITERS_MAX}"
    echo "points_per_iter=${POINTS_PER_ITER}"
    echo "plot_freq=${PLOT_FREQ}"
    echo "save_freq=${SAVE_FREQ}"
    echo "seed=${SEED}"
    echo "dataset=${DATASET_PATH}"
    echo "root=${experiment_root}"
    echo "latest_checkpoint=${latest_checkpoint}"
    echo "named_checkpoint=${named_checkpoint}"
    echo "start_time=${start_time}"
  } > "${config_path}"

  time "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    --dataset "${DATASET_PATH}" \
    --encoding Hash \
    --training-mode "${TRAINING_MODE}" \
    --points-per-iter "${POINTS_PER_ITER}" \
    --nb-iters-max "${NB_ITERS_MAX}" \
    --plot-freq "${PLOT_FREQ}" \
    --save-freq "${SAVE_FREQ}" \
    --seed "${SEED}" \
    --root "${experiment_root}" \
    --hash-n-levels "${hash_n_levels}" \
    --hash-n-min "${HASH_N_MIN}" \
    --hash-n-max "${hash_n_max}" \
    --hash-n-features-per-level "${HASH_FEATURES_PER_LEVEL}" \
    --hash-log2-hashmap-size "${HASH_LOG2_HASHMAP_SIZE}" \
    "${main_flags[@]}"

  end_time="$(date --iso-8601=seconds)"
  if [[ -f "${latest_checkpoint}" ]]; then
    cp -f "${latest_checkpoint}" "${named_checkpoint}"
  else
    echo "Warning: latest checkpoint was not found: ${latest_checkpoint}"
  fi
  echo "end_time=${end_time}" >> "${config_path}"

  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "${experiment}" \
    "${description}" \
    "${hash_n_levels}" \
    "${HASH_N_MIN}" \
    "${hash_n_max}" \
    "${per_level_scale}" \
    "${experiment_root}" \
    "${latest_checkpoint}" \
    "${named_checkpoint}" \
    "${start_time}" \
    "${end_time}" >> "${TRAIN_MANIFEST}"

  echo "[TRAIN] Finished ${experiment} at: ${end_time}"
}

run_export() {
  local experiment="$1"
  local experiment_dir="${OUTPUT_ROOT}/${experiment}"
  local ckpt_path="${experiment_dir}/${experiment}_latest.pkl"
  if [[ ! -f "${ckpt_path}" ]]; then
    ckpt_path="${experiment_dir}/latest/ckpt.pkl"
  fi
  if [[ ! -f "${ckpt_path}" ]]; then
    echo "[EXPORT] Checkpoint not found for ${experiment}"
    echo "Tried:"
    echo "  ${experiment_dir}/${experiment}_latest.pkl"
    echo "  ${experiment_dir}/latest/ckpt.pkl"
    return 1
  fi

  local experiment_output_root="${EXPORT_OUTPUT_ROOT}/${experiment}"
  mkdir -p "${experiment_output_root}"
  local start_time
  start_time="$(date --iso-8601=seconds)"

  echo ""
  echo "============================================================"
  echo "[EXPORT] Experiment: ${experiment}"
  echo "Checkpoint: ${ckpt_path}"
  echo "Output root: ${experiment_output_root}"
  echo "Start: ${start_time}"
  echo "============================================================"

  local cmd=(
    "${PYTHON_BIN}" "${EXPORT_SCRIPT}"
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

  local config_path="${experiment_output_root}/export_config_${PBS_JOBID:-manual}_${ARRAY_ID}.txt"
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

  printf "[EXPORT] Command:"
  printf " %q" "${cmd[@]}"
  printf "\n"

  time "${cmd[@]}"

  local end_time
  end_time="$(date --iso-8601=seconds)"
  local latest_output
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
    "${end_time}" >> "${EXPORT_MANIFEST}"

  echo "[EXPORT] Finished ${experiment} at: ${end_time}"
  echo "[EXPORT] Latest output: ${latest_output}"
}

EXPERIMENTS=(
  "exp1_A_L16_Nmax512 experiment_1_baseline 16 512"
  "exp1_B_L16_Nmax256 experiment_1_lower_Nmax 16 256"
  "exp1_C_L16_Nmax128 experiment_1_lower_Nmax 16 128"
  "exp1_D_L16_Nmax64 experiment_1_lower_Nmax 16 64"
  "exp1_E_L16_Nmax32 experiment_1_lower_Nmax 16 32"
  "exp2_F_L12_Nmax512 experiment_2_fewer_levels 12 512"
  "exp2_G_L8_Nmax512 experiment_2_fewer_levels 8 512"
)

if [[ "${ARRAY_ID}" -lt 0 || "${ARRAY_ID}" -ge "${#EXPERIMENTS[@]}" ]]; then
  echo "Invalid ARRAY_ID=${ARRAY_ID}; expected 0-$(( ${#EXPERIMENTS[@]} - 1 ))"
  exit 1
fi

read -r experiment description hash_n_levels hash_n_max <<< "${EXPERIMENTS[${ARRAY_ID}]}"

run_training "${experiment}" "${description}" "${hash_n_levels}" "${hash_n_max}"
run_export "${experiment}"

echo ""
echo "Train+export task ${ARRAY_ID} finished at: $(date --iso-8601=seconds)"
