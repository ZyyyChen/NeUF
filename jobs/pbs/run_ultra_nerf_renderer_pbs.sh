#!/bin/bash -l
# Strict NeUF + Ultra-NeRF full-frame renderer.
# Submit with:
#   qsub jobs/pbs/run_ultra_nerf_renderer_pbs.sh

#PBS -N neuf_ultra_cerebral_stable
#PBS -q gpu
#PBS -l walltime=23:59:00
#PBS -l nodes=1:ppn=16:gpus=1:gpu48
#PBS -l mem=128gb
#PBS -j oe
#PBS -o /home/zchen/history/neuf_ultra_nerf_cerebral_stable.pbs.log
#PBS -M ziyi.chen@creatis.insa-lyon.fr
#PBS -m ae

set -euo pipefail

REPO_DIR="${REPO_DIR:-/misc/raid/zchen/Code/NeUF}"
PYTHON_BIN="${PYTHON_BIN:-/home/zchen/.conda/envs/neuf/bin/python}"
DATASET_PATH="${DATASET_PATH:-${REPO_DIR}/data/cerebral_data/Pre_traitement_echo_v2/Recalage/Patient0/us_recal_original/baked_dataset_physical.pkl}"
RUN_ROOT="${RUN_ROOT:-${REPO_DIR}/experiments/ultra_nerf_renderer_cerebral_stable}"
LOG_DIR="${LOG_DIR:-/home/zchen/history/neuf_ultra_nerf_cerebral_stable}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"

USE_SAGITTAL="${USE_SAGITTAL:-1}"
SAGITTAL_MAT="${SAGITTAL_MAT:-${REPO_DIR}/data/cerebral_data/Pre_traitement_echo_v2/Repositionnement/Patient0/data_repos_Patient0_J35_2_sag.mat}"
SAGITTAL_VARIABLE="${SAGITTAL_VARIABLE:-data_sag}"
SAGITTAL_WEIGHT="${SAGITTAL_WEIGHT:-0.1}"
SAGITTAL_START_ITER="${SAGITTAL_START_ITER:-4000}"
SAGITTAL_RAMP_ITERS="${SAGITTAL_RAMP_ITERS:-2000}"
OPTIMIZE_SAGITTAL_POSE="${OPTIMIZE_SAGITTAL_POSE:-1}"
SAGITTAL_POSE_LR="${SAGITTAL_POSE_LR:-1e-4}"
SAGITTAL_POSE_LR_END="${SAGITTAL_POSE_LR_END:-1e-5}"
SAGITTAL_POSE_WARMUP_ITERS="${SAGITTAL_POSE_WARMUP_ITERS:-500}"
SAGITTAL_POSE_START_ITER="${SAGITTAL_POSE_START_ITER:-4000}"

ENCODING="${ENCODING:-Hash}"
NB_ITERS_MAX="${NB_ITERS_MAX:-10000}"
PLOT_FREQ="${PLOT_FREQ:-50}"
SAVE_FREQ="${SAVE_FREQ:-100}"
SEED="${SEED:-19981708}"
LR="${LR:-5e-4}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.1}"
QUERY_CHUNK="${QUERY_CHUNK:-65536}"

ULTRA_INIT_ATTENUATION="${ULTRA_INIT_ATTENUATION:-1.0}"
ULTRA_INIT_REFLECTION="${ULTRA_INIT_REFLECTION:-0.02}"
ULTRA_INIT_BORDER_PROBABILITY="${ULTRA_INIT_BORDER_PROBABILITY:-0.005}"
ULTRA_INIT_SCATTER_DENSITY="${ULTRA_INIT_SCATTER_DENSITY:-0.2}"
ULTRA_INIT_SCATTER_AMPLITUDE="${ULTRA_INIT_SCATTER_AMPLITUDE:-0.5}"
ULTRA_INIT_WEIGHT_STD="${ULTRA_INIT_WEIGHT_STD:-1e-4}"
ULTRA_MSE_WARMUP_ITERS="${ULTRA_MSE_WARMUP_ITERS:-500}"
ULTRA_LOSS_RAMP_ITERS="${ULTRA_LOSS_RAMP_ITERS:-1500}"
ULTRA_FINAL_MS_SSIM_WEIGHT="${ULTRA_FINAL_MS_SSIM_WEIGHT:-0.0}"
ULTRA_COLLAPSE_THRESHOLD="${ULTRA_COLLAPSE_THRESHOLD:-1e-6}"
ULTRA_COLLAPSE_PATIENCE="${ULTRA_COLLAPSE_PATIENCE:-20}"

PSF_HALF_SIZE="${PSF_HALF_SIZE:-3}"
PSF_LATERAL_STD="${PSF_LATERAL_STD:-2.0}"
PSF_AXIAL_STD="${PSF_AXIAL_STD:-1.0}"
DISTANCE_UNIT="${DISTANCE_UNIT:-m}"
BERNOULLI_SEED="${BERNOULLI_SEED:-0}"
EVAL_MC_SAMPLES="${EVAL_MC_SAMPLES:-1}"

HASH_N_LEVELS="${HASH_N_LEVELS:-16}"
HASH_N_MIN="${HASH_N_MIN:-16}"
HASH_N_MAX="${HASH_N_MAX:-256}"
HASH_FEATURES_PER_LEVEL="${HASH_FEATURES_PER_LEVEL:-2}"
HASH_LOG2_HASHMAP_SIZE="${HASH_LOG2_HASHMAP_SIZE:-19}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -e "${DATASET_PATH}" ]]; then
  echo "Dataset not found: ${DATASET_PATH}" >&2
  exit 2
fi
if [[ -n "${CHECKPOINT_PATH}" && ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 2
fi
if [[ "${USE_SAGITTAL}" != "0" && "${USE_SAGITTAL}" != "1" ]]; then
  echo "USE_SAGITTAL must be 0 or 1, got: ${USE_SAGITTAL}" >&2
  exit 2
fi
if [[ "${OPTIMIZE_SAGITTAL_POSE}" != "0" && "${OPTIMIZE_SAGITTAL_POSE}" != "1" ]]; then
  echo "OPTIMIZE_SAGITTAL_POSE must be 0 or 1, got: ${OPTIMIZE_SAGITTAL_POSE}" >&2
  exit 2
fi
if [[ "${USE_SAGITTAL}" == "1" && ! -f "${SAGITTAL_MAT}" ]]; then
  echo "Sagittal MATLAB file not found: ${SAGITTAL_MAT}" >&2
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
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

cmd=(
  "${PYTHON_BIN}" -m neuf.main
  --dataset "${DATASET_PATH}"
  --root "${RUN_ROOT}"
  --renderer ultra_nerf
  --encoding "${ENCODING}"
  --training-mode Slice
  --nb-iters-max "${NB_ITERS_MAX}"
  --plot-freq "${PLOT_FREQ}"
  --save-freq "${SAVE_FREQ}"
  --seed "${SEED}"
  --lr "${LR}"
  --grad-clip-norm "${GRAD_CLIP_NORM}"
  --no-loupas
  --grad-weight 0
  --tv-weight 0
  --ssim-weight 0
  --ultra-psf-half-size "${PSF_HALF_SIZE}"
  --ultra-psf-lateral-std "${PSF_LATERAL_STD}"
  --ultra-psf-axial-std "${PSF_AXIAL_STD}"
  --ultra-distance-unit "${DISTANCE_UNIT}"
  --ultra-bernoulli-seed "${BERNOULLI_SEED}"
  --ultra-eval-mc-samples "${EVAL_MC_SAMPLES}"
  --ultra-query-chunk "${QUERY_CHUNK}"
  --ultra-init-attenuation "${ULTRA_INIT_ATTENUATION}"
  --ultra-init-reflection "${ULTRA_INIT_REFLECTION}"
  --ultra-init-border-probability "${ULTRA_INIT_BORDER_PROBABILITY}"
  --ultra-init-scatter-density "${ULTRA_INIT_SCATTER_DENSITY}"
  --ultra-init-scatter-amplitude "${ULTRA_INIT_SCATTER_AMPLITUDE}"
  --ultra-init-weight-std "${ULTRA_INIT_WEIGHT_STD}"
  --ultra-mse-warmup-iters "${ULTRA_MSE_WARMUP_ITERS}"
  --ultra-loss-ramp-iters "${ULTRA_LOSS_RAMP_ITERS}"
  --ultra-final-ms-ssim-weight "${ULTRA_FINAL_MS_SSIM_WEIGHT}"
  --ultra-collapse-threshold "${ULTRA_COLLAPSE_THRESHOLD}"
  --ultra-collapse-patience "${ULTRA_COLLAPSE_PATIENCE}"
  --ultra-save-parameter-maps
  --hash-n-levels "${HASH_N_LEVELS}"
  --hash-n-min "${HASH_N_MIN}"
  --hash-n-max "${HASH_N_MAX}"
  --hash-n-features-per-level "${HASH_FEATURES_PER_LEVEL}"
  --hash-log2-hashmap-size "${HASH_LOG2_HASHMAP_SIZE}"
)

if [[ "${USE_SAGITTAL}" == "1" ]]; then
  cmd+=(
    --sagittal-mat "${SAGITTAL_MAT}"
    --sagittal-variable "${SAGITTAL_VARIABLE}"
    --sagittal-weight "${SAGITTAL_WEIGHT}"
    --sagittal-start-iter "${SAGITTAL_START_ITER}"
    --sagittal-ramp-iters "${SAGITTAL_RAMP_ITERS}"
    --sagittal-pose-lr "${SAGITTAL_POSE_LR}"
    --sagittal-pose-lr-end "${SAGITTAL_POSE_LR_END}"
    --sagittal-pose-warmup-iters "${SAGITTAL_POSE_WARMUP_ITERS}"
    --sagittal-pose-start-iter "${SAGITTAL_POSE_START_ITER}"
    --sagittal-pose-rotation-reg-weight 0
    --sagittal-pose-translation-reg-weight 0
  )
  if [[ "${OPTIMIZE_SAGITTAL_POSE}" == "1" ]]; then
    cmd+=(--optimize-sagittal-pose)
  else
    cmd+=(--no-optimize-sagittal-pose)
  fi
else
  cmd+=(--no-sagittal)
fi

if [[ -n "${CHECKPOINT_PATH}" ]]; then
  cmd+=(--checkpoint "${CHECKPOINT_PATH}")
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
