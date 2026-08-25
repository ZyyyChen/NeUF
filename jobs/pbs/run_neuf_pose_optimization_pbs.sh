#!/bin/bash -l
# Submit with:
#   qsub jobs/pbs/run_neuf_pose_optimization_pbs.sh
# Override settings with, for example:
#   qsub -v DATASET_PATH=/path/data.pkl,NB_ITERS_MAX=20000,POSE_LR=5e-5 \
#     jobs/pbs/run_neuf_pose_optimization_pbs.sh

#PBS -N neuf_pose_opt
#PBS -q gpu
#PBS -l walltime=23:59:00
#PBS -l nodes=1:ppn=16:gpus=1:gpu48
#PBS -l mem=128gb
#PBS -j oe
#PBS -o /home/zchen/history/neuf_pose_optimization.pbs.log
#PBS -M ziyi.chen@creatis.insa-lyon.fr
#PBS -m ae

set -euo pipefail

REPO_DIR="${REPO_DIR:-/misc/raid/zchen/Code/NeUF}"
PYTHON_BIN="${PYTHON_BIN:-/home/zchen/.conda/envs/neuf/bin/python}"
DATASET_PATH="${DATASET_PATH:-${REPO_DIR}/data/cerebral_data/Pre_traitement_echo_v2/Recalage/Patient0/us_recal_original/baked_dataset_physical.pkl}"
RUN_ROOT="${RUN_ROOT:-${REPO_DIR}/experiments/neuf_pose_optimization}"
LOG_DIR="${LOG_DIR:-/home/zchen/history/neuf_pose_optimization}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
DRY_RUN="${DRY_RUN:-0}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"

# NeUF training configuration.
ENCODING="${ENCODING:-DUAL_HASH}"
INTENSITY_ACTIVATION="${INTENSITY_ACTIVATION:-}"
TRAINING_MODE="${TRAINING_MODE:-CurriculumRPS}"
POINTS_PER_ITER="${POINTS_PER_ITER:-50000}"
PATCH_SIZE="${PATCH_SIZE:-32}"
NB_ITERS_MAX="${NB_ITERS_MAX:-10000}"
PLOT_FREQ="${PLOT_FREQ:-100}"
SAVE_FREQ="${SAVE_FREQ:-100}"
SEED="${SEED:-19981708}"
GRAD_WEIGHT="${GRAD_WEIGHT:-0.1}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
TV_WEIGHT="${TV_WEIGHT:-1e-4}"
SSIM_WEIGHT="${SSIM_WEIGHT:-0.1}"
SSIM_WINDOW_SIZE="${SSIM_WINDOW_SIZE:-11}"
PHASE_SWITCH_RATIO="${PHASE_SWITCH_RATIO:-0.4}"
CURRICULUM_RANDOM_RATIO="${CURRICULUM_RANDOM_RATIO:-0.2}"
CURRICULUM_PATCH_RATIO="${CURRICULUM_PATCH_RATIO:-0.5}"

# BARF-style SE(3) pose optimization. Translation is in mm; rotation is in rad.
OPTIMIZE_POSES="${OPTIMIZE_POSES:-1}"
POSE_ANCHOR_FIRST="${POSE_ANCHOR_FIRST:-1}"
POSE_LR="${POSE_LR:-1e-4}"
POSE_LR_END="${POSE_LR_END:-1e-5}"
POSE_WARMUP_ITERS="${POSE_WARMUP_ITERS:-500}"
POSE_START_ITER="${POSE_START_ITER:-2000}"
POSE_ROTATION_REG_WEIGHT="${POSE_ROTATION_REG_WEIGHT:-1e-4}"
POSE_TRANSLATION_REG_WEIGHT="${POSE_TRANSLATION_REG_WEIGHT:-1e-5}"
POSE_VELOCITY_REG_WEIGHT="${POSE_VELOCITY_REG_WEIGHT:-1e-4}"
POSE_ACCELERATION_REG_WEIGHT="${POSE_ACCELERATION_REG_WEIGHT:-1e-4}"
POSE_GRAD_CLIP_NORM="${POSE_GRAD_CLIP_NORM:-1.0}"

# Auxiliary sagittal image.  It starts at the tracked centre pose and learns
# its own 6-DoF SE(3) correction jointly with the NeUF field.
USE_SAGITTAL="${USE_SAGITTAL:-1}"
SAGITTAL_MAT="${SAGITTAL_MAT:-${REPO_DIR}/data/cerebral_data/Pre_traitement_echo_v2/Repositionnement/Patient0/data_repos_Patient0_J35_2_sag.mat}"
SAGITTAL_VARIABLE="${SAGITTAL_VARIABLE:-data_sag}"
SAGITTAL_WEIGHT="${SAGITTAL_WEIGHT:-0.1}"
SAGITTAL_POINTS_PER_ITER="${SAGITTAL_POINTS_PER_ITER:-8192}"
SAGITTAL_START_ITER="${SAGITTAL_START_ITER:-4000}"
SAGITTAL_RAMP_ITERS="${SAGITTAL_RAMP_ITERS:-2000}"
OPTIMIZE_SAGITTAL_POSE="${OPTIMIZE_SAGITTAL_POSE:-1}"
SAGITTAL_POSE_LR="${SAGITTAL_POSE_LR:-1e-4}"
SAGITTAL_POSE_LR_END="${SAGITTAL_POSE_LR_END:-1e-5}"
SAGITTAL_POSE_WARMUP_ITERS="${SAGITTAL_POSE_WARMUP_ITERS:-500}"
SAGITTAL_POSE_START_ITER="${SAGITTAL_POSE_START_ITER:-5000}"
SAGITTAL_POSE_ROTATION_REG_WEIGHT="${SAGITTAL_POSE_ROTATION_REG_WEIGHT:-0.0}"
SAGITTAL_POSE_TRANSLATION_REG_WEIGHT="${SAGITTAL_POSE_TRANSLATION_REG_WEIGHT:-0.0}"
SAGITTAL_POSE_GRAD_CLIP_NORM="${SAGITTAL_POSE_GRAD_CLIP_NORM:-1.0}"

# Hash/dual-frequency configuration.
HASH_N_LEVELS="${HASH_N_LEVELS:-16}"
HASH_N_MIN="${HASH_N_MIN:-16}"
HASH_N_MAX="${HASH_N_MAX:-256}"
HASH_FEATURES_PER_LEVEL="${HASH_FEATURES_PER_LEVEL:-2}"
HASH_LOG2_HASHMAP_SIZE="${HASH_LOG2_HASHMAP_SIZE:-19}"
DUAL_PE_TYPE="${DUAL_PE_TYPE:-hash}"
DUAL_N_LEVELS_LOW="${DUAL_N_LEVELS_LOW:-8}"
DUAL_N_LEVELS_HIGH="${DUAL_N_LEVELS_HIGH:-8}"
DUAL_N_MIN_LOW="${DUAL_N_MIN_LOW:-16}"
DUAL_N_MAX_LOW="${DUAL_N_MAX_LOW:-64}"
DUAL_N_MIN_HIGH="${DUAL_N_MIN_HIGH:-64}"
DUAL_N_MAX_HIGH="${DUAL_N_MAX_HIGH:-512}"
DUAL_HF_ACTIVATE_RATIO="${DUAL_HF_ACTIVATE_RATIO:-0.4}"
DUAL_HF_MAX_WEIGHT="${DUAL_HF_MAX_WEIGHT:-1.0}"
DUAL_SPARSITY_WEIGHT="${DUAL_SPARSITY_WEIGHT:-0.01}"
DUAL_GATE_WEIGHT="${DUAL_GATE_WEIGHT:-0.1}"
DUAL_USE_GATE="${DUAL_USE_GATE:-1}"

# Optional data/noise switches.
RAW_DATASET="${RAW_DATASET:-0}"
JITTER_TRAINING="${JITTER_TRAINING:-0}"
USE_LOUPAS="${USE_LOUPAS:-0}"
NOISE_SIGMA_MIN="${NOISE_SIGMA_MIN:-1e-3}"
NOISE_SIGMA_MAX="${NOISE_SIGMA_MAX:-1.0}"
LOUPAS_GAMMA="${LOUPAS_GAMMA:-0.5}"
LOUPAS_WEIGHT="${LOUPAS_WEIGHT:-0.1}"

require_boolean() {
  local name="$1"
  local value="$2"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "${name} must be 0 or 1, got: ${value}" >&2
    exit 2
  fi
}

require_boolean "OPTIMIZE_POSES" "${OPTIMIZE_POSES}"
require_boolean "POSE_ANCHOR_FIRST" "${POSE_ANCHOR_FIRST}"
require_boolean "USE_SAGITTAL" "${USE_SAGITTAL}"
require_boolean "OPTIMIZE_SAGITTAL_POSE" "${OPTIMIZE_SAGITTAL_POSE}"
require_boolean "DUAL_USE_GATE" "${DUAL_USE_GATE}"
require_boolean "RAW_DATASET" "${RAW_DATASET}"
require_boolean "JITTER_TRAINING" "${JITTER_TRAINING}"
require_boolean "USE_LOUPAS" "${USE_LOUPAS}"
require_boolean "DRY_RUN" "${DRY_RUN}"
require_boolean "REQUIRE_CUDA" "${REQUIRE_CUDA}"

if [[ ! -d "${REPO_DIR}" ]]; then
  echo "Repository directory not found: ${REPO_DIR}" >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -e "${DATASET_PATH}" ]]; then
  echo "Dataset not found: ${DATASET_PATH}" >&2
  exit 2
fi
if [[ "${USE_SAGITTAL}" == "1" && ! -f "${SAGITTAL_MAT}" ]]; then
  echo "Sagittal MATLAB file not found: ${SAGITTAL_MAT}" >&2
  exit 2
fi
if [[ -n "${CHECKPOINT_PATH}" && ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}" "${RUN_ROOT}" "${RUN_ROOT}/.matplotlib"
JOB_LOG="${LOG_DIR}/${PBS_JOBID:-manual}.log"
exec > >(tee -a "${JOB_LOG}") 2>&1

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${RUN_ROOT}/.matplotlib"

THREADS="${PBS_NP:-16}"
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
  "${PYTHON_BIN}" -m neuf.main
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
  --grad-clip-norm "${GRAD_CLIP_NORM}"
  --root "${RUN_ROOT}"
  --tv-weight "${TV_WEIGHT}"
  --ssim-weight "${SSIM_WEIGHT}"
  --ssim-window-size "${SSIM_WINDOW_SIZE}"
  --phase-switch-ratio "${PHASE_SWITCH_RATIO}"
  --curriculum-random-ratio "${CURRICULUM_RANDOM_RATIO}"
  --curriculum-patch-ratio "${CURRICULUM_PATCH_RATIO}"
  --pose-lr "${POSE_LR}"
  --pose-lr-end "${POSE_LR_END}"
  --pose-warmup-iters "${POSE_WARMUP_ITERS}"
  --pose-start-iter "${POSE_START_ITER}"
  --pose-rotation-reg-weight "${POSE_ROTATION_REG_WEIGHT}"
  --pose-translation-reg-weight "${POSE_TRANSLATION_REG_WEIGHT}"
  --pose-velocity-reg-weight "${POSE_VELOCITY_REG_WEIGHT}"
  --pose-acceleration-reg-weight "${POSE_ACCELERATION_REG_WEIGHT}"
  --pose-grad-clip-norm "${POSE_GRAD_CLIP_NORM}"
  --hash-n-levels "${HASH_N_LEVELS}"
  --hash-n-min "${HASH_N_MIN}"
  --hash-n-max "${HASH_N_MAX}"
  --hash-n-features-per-level "${HASH_FEATURES_PER_LEVEL}"
  --hash-log2-hashmap-size "${HASH_LOG2_HASHMAP_SIZE}"
  --dual-pe-type "${DUAL_PE_TYPE}"
  --dual-n-levels-low "${DUAL_N_LEVELS_LOW}"
  --dual-n-levels-high "${DUAL_N_LEVELS_HIGH}"
  --dual-base-resolution-low "${DUAL_N_MIN_LOW}"
  --dual-finest-resolution-low "${DUAL_N_MAX_LOW}"
  --dual-base-resolution-high "${DUAL_N_MIN_HIGH}"
  --dual-finest-resolution-high "${DUAL_N_MAX_HIGH}"
  --dual-hf-activate-ratio "${DUAL_HF_ACTIVATE_RATIO}"
  --dual-hf-max-weight "${DUAL_HF_MAX_WEIGHT}"
  --dual-sparsity-weight "${DUAL_SPARSITY_WEIGHT}"
  --dual-gate-weight "${DUAL_GATE_WEIGHT}"
  --noise-sigma-min "${NOISE_SIGMA_MIN}"
  --noise-sigma-max "${NOISE_SIGMA_MAX}"
  --loupas-gamma "${LOUPAS_GAMMA}"
  --loupas-weight "${LOUPAS_WEIGHT}"
)

if [[ -n "${INTENSITY_ACTIVATION}" ]]; then
  cmd+=(--intensity-activation "${INTENSITY_ACTIVATION}")
fi

if [[ "${OPTIMIZE_POSES}" == "1" ]]; then
  cmd+=(--optimize-poses)
else
  cmd+=(--no-optimize-poses)
fi
if [[ "${POSE_ANCHOR_FIRST}" == "1" ]]; then
  cmd+=(--pose-anchor-first)
else
  cmd+=(--no-pose-anchor-first)
fi
if [[ "${USE_SAGITTAL}" == "1" ]]; then
  cmd+=(
    --sagittal-mat "${SAGITTAL_MAT}"
    --sagittal-variable "${SAGITTAL_VARIABLE}"
    --sagittal-weight "${SAGITTAL_WEIGHT}"
    --sagittal-points-per-iter "${SAGITTAL_POINTS_PER_ITER}"
    --sagittal-start-iter "${SAGITTAL_START_ITER}"
    --sagittal-ramp-iters "${SAGITTAL_RAMP_ITERS}"
    --sagittal-pose-lr "${SAGITTAL_POSE_LR}"
    --sagittal-pose-lr-end "${SAGITTAL_POSE_LR_END}"
    --sagittal-pose-warmup-iters "${SAGITTAL_POSE_WARMUP_ITERS}"
    --sagittal-pose-start-iter "${SAGITTAL_POSE_START_ITER}"
    --sagittal-pose-rotation-reg-weight "${SAGITTAL_POSE_ROTATION_REG_WEIGHT}"
    --sagittal-pose-translation-reg-weight "${SAGITTAL_POSE_TRANSLATION_REG_WEIGHT}"
    --sagittal-pose-grad-clip-norm "${SAGITTAL_POSE_GRAD_CLIP_NORM}"
  )
  if [[ "${OPTIMIZE_SAGITTAL_POSE}" == "1" ]]; then
    cmd+=(--optimize-sagittal-pose)
  else
    cmd+=(--no-optimize-sagittal-pose)
  fi
else
  cmd+=(--no-sagittal)
fi
if [[ "${RAW_DATASET}" == "1" ]]; then
  cmd+=(--raw-dataset)
fi
if [[ "${JITTER_TRAINING}" == "1" ]]; then
  cmd+=(--jitter-training)
fi
if [[ "${DUAL_USE_GATE}" == "0" ]]; then
  cmd+=(--dual-no-gate)
fi
if [[ "${USE_LOUPAS}" == "1" ]]; then
  cmd+=(--use-loupas)
else
  cmd+=(--no-loupas)
fi
if [[ -n "${CHECKPOINT_PATH}" ]]; then
  cmd+=(--checkpoint "${CHECKPOINT_PATH}")
fi

start_time="$(date --iso-8601=seconds)"
CONFIG_PATH="${RUN_ROOT}/pose_optimization_config_${PBS_JOBID:-manual}.txt"
{
  echo "pbs_job_id=${PBS_JOBID:-manual}"
  echo "pbs_queue=${PBS_QUEUE:-unknown}"
  echo "host=$(hostname)"
  echo "repo=${REPO_DIR}"
  echo "python=${PYTHON_BIN}"
  echo "dataset=${DATASET_PATH}"
  echo "checkpoint=${CHECKPOINT_PATH:-none}"
  echo "run_root=${RUN_ROOT}"
  echo "encoding=${ENCODING}"
  echo "intensity_activation=${INTENSITY_ACTIVATION:-auto}"
  echo "training_mode=${TRAINING_MODE}"
  echo "curriculum_random_ratio=${CURRICULUM_RANDOM_RATIO}"
  echo "curriculum_patch_ratio=${CURRICULUM_PATCH_RATIO}"
  echo "points_per_iter=${POINTS_PER_ITER}"
  echo "nb_iters_max=${NB_ITERS_MAX}"
  echo "optimize_poses=${OPTIMIZE_POSES}"
  echo "pose_anchor_first=${POSE_ANCHOR_FIRST}"
  echo "pose_lr=${POSE_LR}"
  echo "pose_lr_end=${POSE_LR_END}"
  echo "pose_warmup_iters=${POSE_WARMUP_ITERS}"
  echo "pose_start_iter=${POSE_START_ITER}"
  echo "pose_rotation_reg_weight=${POSE_ROTATION_REG_WEIGHT}"
  echo "pose_translation_reg_weight=${POSE_TRANSLATION_REG_WEIGHT}"
  echo "pose_velocity_reg_weight=${POSE_VELOCITY_REG_WEIGHT}"
  echo "pose_acceleration_reg_weight=${POSE_ACCELERATION_REG_WEIGHT}"
  echo "pose_grad_clip_norm=${POSE_GRAD_CLIP_NORM}"
  echo "use_sagittal=${USE_SAGITTAL}"
  echo "sagittal_mat=${SAGITTAL_MAT}"
  echo "sagittal_variable=${SAGITTAL_VARIABLE}"
  echo "sagittal_weight=${SAGITTAL_WEIGHT}"
  echo "sagittal_points_per_iter=${SAGITTAL_POINTS_PER_ITER}"
  echo "sagittal_start_iter=${SAGITTAL_START_ITER}"
  echo "sagittal_ramp_iters=${SAGITTAL_RAMP_ITERS}"
  echo "optimize_sagittal_pose=${OPTIMIZE_SAGITTAL_POSE}"
  echo "sagittal_pose_lr=${SAGITTAL_POSE_LR}"
  echo "sagittal_pose_lr_end=${SAGITTAL_POSE_LR_END}"
  echo "sagittal_pose_warmup_iters=${SAGITTAL_POSE_WARMUP_ITERS}"
  echo "sagittal_pose_start_iter=${SAGITTAL_POSE_START_ITER}"
  echo "sagittal_pose_rotation_reg_weight=${SAGITTAL_POSE_ROTATION_REG_WEIGHT}"
  echo "sagittal_pose_translation_reg_weight=${SAGITTAL_POSE_TRANSLATION_REG_WEIGHT}"
  echo "sagittal_pose_grad_clip_norm=${SAGITTAL_POSE_GRAD_CLIP_NORM}"
  echo "use_loupas=${USE_LOUPAS}"
  echo "dry_run=${DRY_RUN}"
  echo "require_cuda=${REQUIRE_CUDA}"
  echo "threads=${THREADS}"
  echo "start_time=${start_time}"
  printf "command="
  printf " %q" "${cmd[@]}"
  printf "\n"
} > "${CONFIG_PATH}"

echo "Host: $(hostname)"
echo "Date: ${start_time}"
echo "PBS job id: ${PBS_JOBID:-manual}"
echo "PBS queue: ${PBS_QUEUE:-unknown}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not-set}"
echo "Repository: ${REPO_DIR}"
echo "Dataset: ${DATASET_PATH}"
echo "Run root: ${RUN_ROOT}"
echo "Config: ${CONFIG_PATH}"
echo "Job log: ${JOB_LOG}"
echo "Requested resources: 1 gpu48 GPU, 16 CPU cores, 128 GB RAM, 23:59:00"

if command -v nvidia-smi >/dev/null 2>&1; then
  if ! nvidia-smi; then
    echo "Warning: nvidia-smi exists but could not access a GPU."
  fi
else
  echo "Warning: nvidia-smi was not found on the allocated node."
fi

"${PYTHON_BIN}" -c 'import torch; print(f"PyTorch={torch.__version__}, CUDA={torch.cuda.is_available()}, GPUs={torch.cuda.device_count()}")'
if [[ "${REQUIRE_CUDA}" == "1" && "${DRY_RUN}" == "0" ]]; then
  "${PYTHON_BIN}" -c 'import sys, torch; sys.exit("CUDA is unavailable on the allocated node") if not torch.cuda.is_available() else None'
fi

printf "Command:"
printf " %q" "${cmd[@]}"
printf "\n"

if [[ "${DRY_RUN}" == "1" ]]; then
  "${cmd[@]}" --help >/dev/null
  echo "Dry run passed: NeUF accepted all command-line arguments; training was not started."
  exit 0
fi

time "${cmd[@]}"

end_time="$(date --iso-8601=seconds)"
echo "end_time=${end_time}" >> "${CONFIG_PATH}"
echo "Finished NeUF pose-optimization training at: ${end_time}"
echo "Latest checkpoint: ${RUN_ROOT}/latest/ckpt.pkl"
