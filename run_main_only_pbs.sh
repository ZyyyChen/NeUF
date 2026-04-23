#!/bin/bash -l
#PBS -N neuf_main
#PBS -q gpu
#PBS -l walltime=23:59:00
#PBS -l nodes=1:ppn=16:gpus=1:gpu48
#PBS -l mem=128gb
#PBS -j oe
#PBS -o /home/zchen/history/neuf_main.pbs.log
#PBS -M ziyi.chen@creatis.insa-lyon.fr
#PBS -m ae

set -euo pipefail

REPO_DIR="${REPO_DIR:-/misc/raid/zchen/Code/NeUF}"
PYTHON_BIN="${PYTHON_BIN:-/home/zchen/.conda/envs/neuf/bin/python}"
SCRIPT_PATH="${REPO_DIR}/main.py"
DATASET_PATH="${DATASET_PATH:-${REPO_DIR}/data/cerebral_data/Pre_traitement_echo_v2/Recalage/Patient0/us_recal_original/baked_dataset.pkl}"
RUN_ROOT="${RUN_ROOT:-${REPO_DIR}}"
LOG_DIR="${LOG_DIR:-/home/zchen/history/neuf_main}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
ENCODING="${ENCODING:-Hash}"
TRAINING_MODE="${TRAINING_MODE:-Random}"
POINTS_PER_ITER="${POINTS_PER_ITER:-50000}"
PATCH_SIZE="${PATCH_SIZE:-32}"
NB_ITERS_MAX="${NB_ITERS_MAX:-10000}"
PLOT_FREQ="${PLOT_FREQ:-100}"
SAVE_FREQ="${SAVE_FREQ:-100}"
SEED="${SEED:-19981708}"
GRAD_WEIGHT="${GRAD_WEIGHT:-0.1}"
GRAD_BLUR_KERNEL_SIZE="${GRAD_BLUR_KERNEL_SIZE:-6}"
GRAD_BLUR_SIGMA="${GRAD_BLUR_SIGMA:-1.5}"
TV_WEIGHT="${TV_WEIGHT:-100}"
SLICE_MIX_INTERVAL="${SLICE_MIX_INTERVAL:-10}"
SMOOTHNESS_DELTA="${SMOOTHNESS_DELTA:-0.1}"

HASH_N_LEVELS="${HASH_N_LEVELS:-16}"
HASH_N_MIN="${HASH_N_MIN:-16}"
HASH_N_MAX="${HASH_N_MAX:-256}"
HASH_FEATURES_PER_LEVEL="${HASH_FEATURES_PER_LEVEL:-2}"
HASH_LOG2_HASHMAP_SIZE="${HASH_LOG2_HASHMAP_SIZE:-19}"

RAW_DATASET="${RAW_DATASET:-0}"
JITTER_TRAINING="${JITTER_TRAINING:-0}"

mkdir -p "${LOG_DIR}" "${RUN_ROOT}"
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

cmd=(
  "${PYTHON_BIN}" "${SCRIPT_PATH}"
  --dataset "${DATASET_PATH}"
  --encoding "${ENCODING}"
  --training-mode "${TRAINING_MODE}"
  --points-per-iter "${POINTS_PER_ITER}"
  --patch-size "${PATCH_SIZE}"
  --nb-iters-max "${NB_ITERS_MAX}"
  --plot-freq "${PLOT_FREQ}"
  --save-freq "${SAVE_FREQ}"
  --seed "${SEED}"
  --grad-weight "${GRAD_WEIGHT}"
  --grad-blur-kernel-size "${GRAD_BLUR_KERNEL_SIZE}"
  --grad-blur-sigma "${GRAD_BLUR_SIGMA}"
  --root "${RUN_ROOT}"
  --tv-weight "${TV_WEIGHT}"
  --slice-mix-interval "${SLICE_MIX_INTERVAL}"
  --smoothness-delta "${SMOOTHNESS_DELTA}"
  --hash-n-levels "${HASH_N_LEVELS}"
  --hash-n-min "${HASH_N_MIN}"
  --hash-n-max "${HASH_N_MAX}"
  --hash-n-features-per-level "${HASH_FEATURES_PER_LEVEL}"
  --hash-log2-hashmap-size "${HASH_LOG2_HASHMAP_SIZE}"
)

if [[ -n "${CHECKPOINT_PATH}" ]]; then
  cmd+=(--checkpoint "${CHECKPOINT_PATH}")
fi
if [[ "${RAW_DATASET}" == "1" ]]; then
  cmd+=(--raw-dataset)
fi
if [[ "${JITTER_TRAINING}" == "1" ]]; then
  cmd+=(--jitter-training)
fi

start_time="$(date --iso-8601=seconds)"
config_path="${RUN_ROOT}/main_config_${PBS_JOBID:-manual}.txt"
{
  echo "script=${SCRIPT_PATH}"
  echo "dataset=${DATASET_PATH}"
  echo "checkpoint=${CHECKPOINT_PATH:-none}"
  echo "run_root=${RUN_ROOT}"
  echo "encoding=${ENCODING}"
  echo "training_mode=${TRAINING_MODE}"
  echo "points_per_iter=${POINTS_PER_ITER}"
  echo "patch_size=${PATCH_SIZE}"
  echo "nb_iters_max=${NB_ITERS_MAX}"
  echo "plot_freq=${PLOT_FREQ}"
  echo "save_freq=${SAVE_FREQ}"
  echo "seed=${SEED}"
  echo "grad_weight=${GRAD_WEIGHT}"
  echo "grad_blur_kernel_size=${GRAD_BLUR_KERNEL_SIZE}"
  echo "grad_blur_sigma=${GRAD_BLUR_SIGMA}"
  echo "tv_weight=${TV_WEIGHT}"
  echo "slice_mix_interval=${SLICE_MIX_INTERVAL}"
  echo "smoothness_delta=${SMOOTHNESS_DELTA}"
  echo "hash_n_levels=${HASH_N_LEVELS}"
  echo "hash_n_min=${HASH_N_MIN}"
  echo "hash_n_max=${HASH_N_MAX}"
  echo "hash_features_per_level=${HASH_FEATURES_PER_LEVEL}"
  echo "hash_log2_hashmap_size=${HASH_LOG2_HASHMAP_SIZE}"
  echo "raw_dataset=${RAW_DATASET}"
  echo "jitter_training=${JITTER_TRAINING}"
  echo "start_time=${start_time}"
  printf "command="
  printf " %q" "${cmd[@]}"
  printf "\n"
} > "${config_path}"

echo "Host: $(hostname)"
echo "Date: ${start_time}"
echo "PBS job id: ${PBS_JOBID:-n/a}"
echo "PBS queue: ${PBS_QUEUE:-n/a}"
echo "Working dir: ${REPO_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "Script: ${SCRIPT_PATH}"
echo "Dataset: ${DATASET_PATH}"
echo "Run root: ${RUN_ROOT}"
echo "Latest checkpoint: ${RUN_ROOT}/latest/ckpt.pkl"
echo "Threads: ${THREADS}"
echo "Config: ${config_path}"
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
echo "end_time=${end_time}" >> "${config_path}"

echo "Finished main.py at: ${end_time}"
echo "Latest checkpoint: ${RUN_ROOT}/latest/ckpt.pkl"
