#!/bin/bash -l
#PBS -N neuf_nmax_exp
#PBS -q gpu
#PBS -l walltime=23:59:00
#PBS -l nodes=1:ppn=16:gpus=1:gpu48
#PBS -l mem=128gb
#PBS -j oe
#PBS -o /home/zchen/history/neuf_nmax_experiments.pbs.log
#PBS -M ziyi.chen@creatis.insa-lyon.fr
#PBS -m ae
#PBS -t 0-6

set -euo pipefail

REPO_DIR="${REPO_DIR:-/misc/raid/zchen/Code/NeUF}"
PYTHON_BIN="${PYTHON_BIN:-/home/zchen/.conda/envs/neuf/bin/python}"
SCRIPT_PATH="${REPO_DIR}/main.py"
DATASET_PATH="${DATASET_PATH:-${REPO_DIR}/data/cerebral_data/Pre_traitement_echo_v2/Recalage/Patient0/us_recal_original/baked_dataset.pkl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/experiments/hash_grid_nmax}"
LOG_DIR="${LOG_DIR:-/home/zchen/history/neuf_nmax_experiments}"

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
MANIFEST_PATH="${OUTPUT_ROOT}/experiment_manifest_${PBS_JOBID:-manual}_${ARRAY_ID}.csv"
printf "experiment,description,hash_n_levels,hash_n_min,hash_n_max,per_level_scale,root,latest_checkpoint,named_checkpoint,start_time,end_time\n" > "${MANIFEST_PATH}"

echo "Host: $(hostname)"
echo "Date: $(date --iso-8601=seconds)"
echo "PBS job id: ${PBS_JOBID:-n/a}"
echo "PBS array id: ${ARRAY_ID}"
echo "PBS queue: ${PBS_QUEUE:-n/a}"
echo "Working dir: ${REPO_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "Dataset: ${DATASET_PATH}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Threads: ${THREADS}"
echo "Iterations: ${NB_ITERS_MAX}"
echo "Points per iter: ${POINTS_PER_ITER}"
echo "Manifest: ${MANIFEST_PATH}"
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

run_experiment() {
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
  echo "Experiment: ${experiment}"
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

  time "${PYTHON_BIN}" "${SCRIPT_PATH}" \
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
    "${end_time}" >> "${MANIFEST_PATH}"

  echo "Finished ${experiment} at: ${end_time}"
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
run_experiment "${experiment}" "${description}" "${hash_n_levels}" "${hash_n_max}"

echo ""
echo "N_max experiment task ${ARRAY_ID} finished at: $(date --iso-8601=seconds)"
