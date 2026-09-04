#!/bin/bash -l
# Run Phase 1 E0 independently: plain HASH fixed-geometry baseline.
# Example: qsub jobs/phase_1/run_e0.sh

#PBS -N neuf_phase1_e0
#PBS -q gpu
#PBS -l walltime=23:59:00
#PBS -l nodes=1:ppn=16:gpus=1:gpu48
#PBS -l mem=128gb
#PBS -o /misc/raid/zchen/Code/NeUF/qsub/logs/neuf/phase1_e0_stdout.log
#PBS -e /misc/raid/zchen/Code/NeUF/qsub/logs/neuf/phase1_e0_stderr.log
#PBS -M ziyi.chen@creatis.insa-lyon.fr
#PBS -m ae

set -euo pipefail

REPO_DIR="${REPO_DIR:-/misc/raid/zchen/Code/NeUF}"
PYTHON_BIN="${PYTHON_BIN:-/home/zchen/.conda/envs/neuf/bin/python}"
SEED="${SEED:-3407}"
EXPERIMENT_ID=E0
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
TRIAL_ID="${TRIAL_ID:-}"
RUN_CONTENT="${RUN_CONTENT:-phase1_e1_plain_dual_vs_e2_hf02}"
NB_ITERS_MAX="${NB_ITERS_MAX:-20000}"
POINTS_PER_ITER="${POINTS_PER_ITER:-49152}"
PATCH_SIZE="${PATCH_SIZE:-64}"
PLOT_FREQ="${PLOT_FREQ:-1000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"

if ! [[ "${RUN_DATE}" =~ ^[0-9]{8}$ ]]; then
  echo "Invalid RUN_DATE '${RUN_DATE}'; expected YYYYMMDD." >&2
  exit 2
fi
if [[ -n "${TRIAL_ID}" ]] && ! [[ "${TRIAL_ID}" =~ ^0*[1-9][0-9]*$ ]]; then
  echo "Invalid TRIAL_ID '${TRIAL_ID}'; expected a positive integer." >&2
  exit 2
fi
if ! [[ "${RUN_CONTENT}" =~ ^[a-z0-9][a-z0-9._-]*$ ]]; then
  echo "Invalid RUN_CONTENT '${RUN_CONTENT}'." >&2
  exit 2
fi

if [[ -z "${TRIAL_ID}" ]]; then
  NEXT_TRIAL=1
  while :; do
    printf -v TRIAL_CANDIDATE "%02d" "${NEXT_TRIAL}"
    CANDIDATE_ROOT="${REPO_DIR}/experiments/${RUN_DATE}_trial${TRIAL_CANDIDATE}/${RUN_CONTENT}/${EXPERIMENT_ID}/seed${SEED}"
    if [[ ! -e "${CANDIDATE_ROOT}" ]]; then
      TRIAL_ID="${NEXT_TRIAL}"
      break
    fi
    NEXT_TRIAL="$((NEXT_TRIAL + 1))"
  done
fi

printf -v TRIAL_LABEL "%02d" "$((10#${TRIAL_ID}))"
RUN_GROUP="${RUN_DATE}_trial${TRIAL_LABEL}"
PHASE1_DIR="${PHASE1_DIR:-${REPO_DIR}/experiments/${RUN_GROUP}/${RUN_CONTENT}}"
DATASET_PATH="${DATASET_PATH:-${REPO_DIR}/data/cerebral_data/Pre_traitement_echo_v2/Recalage/Patient0/us_recal_original/baked_dataset_physical.pkl}"
RUN_ROOT="${PHASE1_DIR}/${EXPERIMENT_ID}/seed${SEED}"
LOG_DIR="${REPO_DIR}/qsub/logs/neuf/${RUN_GROUP}"

if [[ ! -d "${REPO_DIR}/neuf" || ! -x "${PYTHON_BIN}" || ! -f "${DATASET_PATH}" ]]; then
  echo "Missing repository, Python executable, or dataset." >&2
  exit 1
fi
if [[ -e "${RUN_ROOT}" ]]; then
  echo "Run directory already exists: ${RUN_ROOT}" >&2
  echo "Increment TRIAL_ID or choose another RUN_CONTENT; runs are never overwritten." >&2
  exit 2
fi

mkdir -p "${LOG_DIR}" "${RUN_ROOT}"
JOB_ID="${PBS_JOBID:-manual}"
JOB_LOG="${LOG_DIR}/E0_${JOB_ID}_seed${SEED}.log"
echo "${JOB_ID}" > "${LOG_DIR}/E0_job_id.txt"
exec > >(tee -a "${JOB_LOG}") 2>&1

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
THREADS="${PBS_NP:-16}"
export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"

cmd=(
  "${PYTHON_BIN}" -m neuf.main
  --dataset "${DATASET_PATH}"
  --phase1-image-quality
  --phase1-output-dir "${PHASE1_DIR}"
  --field-head legacy_fixed_geometry
  --encoding HASH
  --intensity-activation identity
  --training-mode Patch
  --patch-size "${PATCH_SIZE}"
  --points-per-iter "${POINTS_PER_ITER}"
  --nb-iters-max "${NB_ITERS_MAX}"
  --plot-freq "${PLOT_FREQ}"
  --save-freq "${SAVE_FREQ}"
  --seed "${SEED}"
  --lr 5e-4
  --lr-decay-factor 0.1
  --hash-n-levels 16
  --hash-n-features-per-level 2
  --hash-log2-hashmap-size 19
  --hash-base-resolution 16
  --hash-finest-resolution 256
  --root "${RUN_ROOT}"
)

START_TIME="$(date --iso-8601=seconds)"
CONFIG_PATH="${RUN_ROOT}/${RUN_GROUP}_${RUN_CONTENT}_E0_config_${JOB_ID}.txt"
{
  echo "run_date=${RUN_DATE}"
  echo "trial_id=${TRIAL_LABEL}"
  echo "run_content=${RUN_CONTENT}"
  echo "pbs_job_id=${JOB_ID}"
  echo "experiment_id=${EXPERIMENT_ID}"
  echo "field_head=legacy_fixed_geometry"
  echo "encoding=HASH"
  echo "seed=${SEED}"
  echo "nb_iters_max=${NB_ITERS_MAX}"
  echo "dataset=${DATASET_PATH}"
  echo "run_root=${RUN_ROOT}"
  echo "start_time=${START_TIME}"
  printf "command="
  printf " %q" "${cmd[@]}"
  printf "\n"
} > "${CONFIG_PATH}"

echo "Host: $(hostname)"
echo "Start: ${START_TIME}"
echo "PBS job id: ${JOB_ID}"
echo "Experiment: ${EXPERIMENT_ID}"
echo "Dataset: ${DATASET_PATH}"
echo "Result directory: ${RUN_ROOT}"
echo "Experiment log: ${JOB_LOG}"
echo "qsub stdout: ${REPO_DIR}/qsub/logs/neuf/phase1_e0_stdout.log"
echo "qsub stderr: ${REPO_DIR}/qsub/logs/neuf/phase1_e0_stderr.log"
printf "Command:"
printf " %q" "${cmd[@]}"
printf "\n"

time "${cmd[@]}"

END_TIME="$(date --iso-8601=seconds)"
echo "end_time=${END_TIME}" >> "${CONFIG_PATH}"
echo "Finished: ${END_TIME}"
echo "Final checkpoint: ${RUN_ROOT}/latest/ckpt.pkl"
