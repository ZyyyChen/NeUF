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
MAIN_MODULE="neuf.main"
DATASET_PATH="${DATASET_PATH:-${REPO_DIR}/data/cerebral_data/Pre_traitement_echo_v2/Recalage/Patient0/us_recal_original/baked_dataset_physical.pkl}"
RUN_ROOT="${RUN_ROOT:-${REPO_DIR}}"
LOG_DIR="${LOG_DIR:-/home/zchen/history/neuf_main}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
ENCODING="${ENCODING:-DUAL_HASH}"
TRAINING_MODE="${TRAINING_MODE:-CurriculumRS}"
POINTS_PER_ITER="${POINTS_PER_ITER:-50000}"
PATCH_SIZE="${PATCH_SIZE:-32}"
NB_ITERS_MAX="${NB_ITERS_MAX:-10000}"
PLOT_FREQ="${PLOT_FREQ:-100}"
SAVE_FREQ="${SAVE_FREQ:-100}"
SEED="${SEED:-19981708}"
GRAD_WEIGHT="${GRAD_WEIGHT:-5}"
GRAD_BLUR_KERNEL_SIZE="${GRAD_BLUR_KERNEL_SIZE:-6}"
GRAD_BLUR_SIGMA="${GRAD_BLUR_SIGMA:-1.5}"
TV_WEIGHT="${TV_WEIGHT:-1}"
SLICE_MIX_INTERVAL="${SLICE_MIX_INTERVAL:-10}"
SMOOTHNESS_DELTA="${SMOOTHNESS_DELTA:-0.1}"
USE_LATERAL_PERTURBATION="${USE_LATERAL_PERTURBATION:-0}"

HASH_N_LEVELS="${HASH_N_LEVELS:-16}"
HASH_N_MIN="${HASH_N_MIN:-16}"
HASH_N_MAX="${HASH_N_MAX:-256}"
HASH_FEATURES_PER_LEVEL="${HASH_FEATURES_PER_LEVEL:-2}"
HASH_LOG2_HASHMAP_SIZE="${HASH_LOG2_HASHMAP_SIZE:-19}"

KRONECKER_N_LEVELS_LATERAL="${KRONECKER_N_LEVELS_LATERAL:-8}"
KRONECKER_N_LEVELS_AXIAL="${KRONECKER_N_LEVELS_AXIAL:-8}"
KRONECKER_N_MAX_LATERAL="${KRONECKER_N_MAX_LATERAL:-128}"
KRONECKER_N_MAX_AXIAL="${KRONECKER_N_MAX_AXIAL:-512}"
KRONECKER_N_MIN="${KRONECKER_N_MIN:-16}"
KRONECKER_FEATURES_PER_LEVEL="${KRONECKER_FEATURES_PER_LEVEL:-${HASH_FEATURES_PER_LEVEL}}"
KRONECKER_LOG2_HASHMAP_SIZE="${KRONECKER_LOG2_HASHMAP_SIZE:-${HASH_LOG2_HASHMAP_SIZE}}"
KRONECKER_COMBINE="${KRONECKER_COMBINE:-cat}"

DUAL_PE_TYPE="${DUAL_PE_TYPE:-}"
if [[ -z "${DUAL_PE_TYPE}" ]]; then
  if [[ "${ENCODING}" == "DUAL_FREQ" || "${ENCODING}" == "dual_freq" ]]; then
    DUAL_PE_TYPE="fourier"
  else
    DUAL_PE_TYPE="hash"
  fi
fi
DUAL_N_LEVELS_LOW="${DUAL_N_LEVELS_LOW:-8}"
DUAL_N_LEVELS_HIGH="${DUAL_N_LEVELS_HIGH:-8}"
DUAL_N_MIN_LOW="${DUAL_N_MIN_LOW:-16}"
DUAL_N_MAX_LOW="${DUAL_N_MAX_LOW:-64}"
DUAL_N_MIN_HIGH="${DUAL_N_MIN_HIGH:-64}"
DUAL_N_MAX_HIGH="${DUAL_N_MAX_HIGH:-512}"
DUAL_SIGMA_LOW="${DUAL_SIGMA_LOW:-1.0}"
DUAL_SIGMA_HIGH="${DUAL_SIGMA_HIGH:-20.0}"
DUAL_N_FREQ="${DUAL_N_FREQ:-64}"
DUAL_USE_GATE="${DUAL_USE_GATE:-1}"
DUAL_HF_ACTIVATE_RATIO="${DUAL_HF_ACTIVATE_RATIO:-0.4}"
PHASE_SWITCH_RATIO="${PHASE_SWITCH_RATIO:-${DUAL_HF_ACTIVATE_RATIO}}"
DUAL_HF_MAX_WEIGHT="${DUAL_HF_MAX_WEIGHT:-1.0}"
DUAL_SPARSITY_WEIGHT="${DUAL_SPARSITY_WEIGHT:-10}"
DUAL_GATE_WEIGHT="${DUAL_GATE_WEIGHT:-2}"

USE_LOUPAS="${USE_LOUPAS:-0}"
NOISE_SIGMA_MIN="${NOISE_SIGMA_MIN:-1e-3}"
NOISE_SIGMA_MAX="${NOISE_SIGMA_MAX:-1.0}"
LOUPAS_GAMMA="${LOUPAS_GAMMA:-0.5}"
LOUPAS_WEIGHT="${LOUPAS_WEIGHT:-0.1}"

RAW_DATASET="${RAW_DATASET:-0}"
JITTER_TRAINING="${JITTER_TRAINING:-0}"

mkdir -p "${LOG_DIR}" "${RUN_ROOT}"
JOB_LOG="${LOG_DIR}/${PBS_JOBID:-manual}.log"
exec > >(tee -a "${JOB_LOG}") 2>&1

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

THREADS="${PBS_NP:-16}"
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"
export VECLIB_MAXIMUM_THREADS="${THREADS}"
export BLIS_NUM_THREADS="${THREADS}"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export KMP_AFFINITY=granularity=fine,compact,1,0

cmd=(
  "${PYTHON_BIN}" -m "${MAIN_MODULE}"
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
  --phase-switch-ratio "${PHASE_SWITCH_RATIO}"
  --hash-n-levels "${HASH_N_LEVELS}"
  --hash-n-min "${HASH_N_MIN}"
  --hash-n-max "${HASH_N_MAX}"
  --hash-n-features-per-level "${HASH_FEATURES_PER_LEVEL}"
  --hash-log2-hashmap-size "${HASH_LOG2_HASHMAP_SIZE}"
  --kronecker-n-levels-lateral "${KRONECKER_N_LEVELS_LATERAL}"
  --kronecker-n-levels-axial "${KRONECKER_N_LEVELS_AXIAL}"
  --kronecker-finest-lateral "${KRONECKER_N_MAX_LATERAL}"
  --kronecker-finest-axial "${KRONECKER_N_MAX_AXIAL}"
  --kronecker-n-features-per-level "${KRONECKER_FEATURES_PER_LEVEL}"
  --kronecker-log2-hashmap-size "${KRONECKER_LOG2_HASHMAP_SIZE}"
  --kronecker-base-resolution "${KRONECKER_N_MIN}"
  --kronecker-combine "${KRONECKER_COMBINE}"
  --dual-pe-type "${DUAL_PE_TYPE}"
  --dual-n-levels-low "${DUAL_N_LEVELS_LOW}"
  --dual-n-levels-high "${DUAL_N_LEVELS_HIGH}"
  --dual-base-resolution-low "${DUAL_N_MIN_LOW}"
  --dual-finest-resolution-low "${DUAL_N_MAX_LOW}"
  --dual-base-resolution-high "${DUAL_N_MIN_HIGH}"
  --dual-finest-resolution-high "${DUAL_N_MAX_HIGH}"
  --dual-sigma-low "${DUAL_SIGMA_LOW}"
  --dual-sigma-high "${DUAL_SIGMA_HIGH}"
  --dual-n-freq "${DUAL_N_FREQ}"
  --dual-hf-activate-ratio "${DUAL_HF_ACTIVATE_RATIO}"
  --dual-hf-max-weight "${DUAL_HF_MAX_WEIGHT}"
  --dual-sparsity-weight "${DUAL_SPARSITY_WEIGHT}"
  --dual-gate-weight "${DUAL_GATE_WEIGHT}"
  --noise-sigma-min "${NOISE_SIGMA_MIN}"
  --noise-sigma-max "${NOISE_SIGMA_MAX}"
  --loupas-gamma "${LOUPAS_GAMMA}"
  --loupas-weight "${LOUPAS_WEIGHT}"
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
if [[ "${USE_LATERAL_PERTURBATION}" == "1" ]]; then
  cmd+=(--use-lateral-perturbation)
fi
if [[ "${DUAL_USE_GATE}" == "0" ]]; then
  cmd+=(--dual-no-gate)
fi
if [[ "${USE_LOUPAS}" == "0" ]]; then
  cmd+=(--no-loupas)
else
  cmd+=(--use-loupas)
fi

start_time="$(date --iso-8601=seconds)"
config_path="${RUN_ROOT}/main_config_${PBS_JOBID:-manual}.txt"
{
  echo "module=${MAIN_MODULE}"
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
  echo "use_lateral_perturbation=${USE_LATERAL_PERTURBATION}"
  echo "phase_switch_ratio=${PHASE_SWITCH_RATIO}"
  echo "hash_n_levels=${HASH_N_LEVELS}"
  echo "hash_n_min=${HASH_N_MIN}"
  echo "hash_n_max=${HASH_N_MAX}"
  echo "hash_features_per_level=${HASH_FEATURES_PER_LEVEL}"
  echo "hash_log2_hashmap_size=${HASH_LOG2_HASHMAP_SIZE}"
  echo "kronecker_n_levels_lateral=${KRONECKER_N_LEVELS_LATERAL}"
  echo "kronecker_n_levels_axial=${KRONECKER_N_LEVELS_AXIAL}"
  echo "kronecker_n_min=${KRONECKER_N_MIN}"
  echo "kronecker_n_max_lateral=${KRONECKER_N_MAX_LATERAL}"
  echo "kronecker_n_max_axial=${KRONECKER_N_MAX_AXIAL}"
  echo "kronecker_features_per_level=${KRONECKER_FEATURES_PER_LEVEL}"
  echo "kronecker_log2_hashmap_size=${KRONECKER_LOG2_HASHMAP_SIZE}"
  echo "kronecker_combine=${KRONECKER_COMBINE}"
  echo "dual_pe_type=${DUAL_PE_TYPE}"
  echo "dual_n_levels_low=${DUAL_N_LEVELS_LOW}"
  echo "dual_n_levels_high=${DUAL_N_LEVELS_HIGH}"
  echo "dual_n_min_low=${DUAL_N_MIN_LOW}"
  echo "dual_n_max_low=${DUAL_N_MAX_LOW}"
  echo "dual_n_min_high=${DUAL_N_MIN_HIGH}"
  echo "dual_n_max_high=${DUAL_N_MAX_HIGH}"
  echo "dual_sigma_low=${DUAL_SIGMA_LOW}"
  echo "dual_sigma_high=${DUAL_SIGMA_HIGH}"
  echo "dual_n_freq=${DUAL_N_FREQ}"
  echo "dual_use_gate=${DUAL_USE_GATE}"
  echo "dual_hf_activate_ratio=${DUAL_HF_ACTIVATE_RATIO}"
  echo "dual_hf_max_weight=${DUAL_HF_MAX_WEIGHT}"
  echo "dual_sparsity_weight=${DUAL_SPARSITY_WEIGHT}"
  echo "dual_gate_weight=${DUAL_GATE_WEIGHT}"
  echo "use_loupas=${USE_LOUPAS}"
  echo "noise_sigma_min=${NOISE_SIGMA_MIN}"
  echo "noise_sigma_max=${NOISE_SIGMA_MAX}"
  echo "loupas_gamma=${LOUPAS_GAMMA}"
  echo "loupas_weight=${LOUPAS_WEIGHT}"
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
echo "Module: ${MAIN_MODULE}"
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
