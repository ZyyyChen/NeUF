#!/bin/bash -l
# Evaluate the three seed-3407 checkpoints at alpha=0,0.25,0.5,0.75,1.
# Pass the same RUN_DATE, TRIAL_ID and RUN_CONTENT used for training:
#   qsub -v RUN_DATE=20260901,TRIAL_ID=01,RUN_CONTENT=phase1_e1_plain_dual_vs_e2_hf02 \
#     jobs/pbs/run_phase1_image_quality_eval_pbs.sh

#PBS -N neuf_phase1_eval
#PBS -q gpu
#PBS -l walltime=23:59:00
#PBS -l nodes=1:ppn=16:gpus=1:gpu48
#PBS -l mem=128gb
#PBS -j oe
#PBS -o /home/zchen/history/neuf_phase1_image_quality_eval.pbs.log
#PBS -M ziyi.chen@creatis.insa-lyon.fr
#PBS -m ae

set -euo pipefail

REPO_DIR="${REPO_DIR:-/misc/raid/zchen/Code/NeUF}"
PYTHON_BIN="${PYTHON_BIN:-/home/zchen/.conda/envs/neuf/bin/python}"
SEED=3407
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
TRIAL_ID="${TRIAL_ID:-01}"
RUN_CONTENT="${RUN_CONTENT:-phase1_e1_plain_dual_vs_e2_hf02}"
if ! [[ "${RUN_DATE}" =~ ^[0-9]{8}$ ]]; then
  echo "Invalid RUN_DATE '${RUN_DATE}'; expected YYYYMMDD." >&2
  exit 2
fi
if ! [[ "${TRIAL_ID}" =~ ^0*[1-9][0-9]*$ ]]; then
  echo "Invalid TRIAL_ID '${TRIAL_ID}'; expected a positive integer." >&2
  exit 2
fi
if ! [[ "${RUN_CONTENT}" =~ ^[a-z0-9][a-z0-9._-]*$ ]]; then
  echo "Invalid RUN_CONTENT '${RUN_CONTENT}'; use lowercase letters, digits, '.', '_' or '-'." >&2
  exit 2
fi
printf -v TRIAL_LABEL "%02d" "$((10#${TRIAL_ID}))"
RUN_GROUP="${RUN_DATE}_trial${TRIAL_LABEL}"
PHASE1_DIR="${PHASE1_DIR:-${REPO_DIR}/experiments/${RUN_GROUP}/${RUN_CONTENT}}"
DATASET_PATH="${DATASET_PATH:-${REPO_DIR}/data/cerebral_data/Pre_traitement_echo_v2/Recalage/Patient0/us_recal_original/baked_dataset_physical.pkl}"
LOG_DIR="${LOG_DIR:-/home/zchen/history/neuf_phase1_image_quality_eval}"
ALPHA_VALUES="0.0,0.25,0.50,0.75,1.0"

E0_CHECKPOINT="${PHASE1_DIR}/E0/seed${SEED}/latest/ckpt.pkl"
E1_CHECKPOINT="${PHASE1_DIR}/E1/seed${SEED}/latest/ckpt.pkl"
E2_CHECKPOINT="${PHASE1_DIR}/E2/seed${SEED}/latest/ckpt.pkl"

if [[ ! -d "${REPO_DIR}/neuf" ]]; then
  echo "NeUF package directory not found: ${REPO_DIR}/neuf" >&2
  exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Dataset not found: ${DATASET_PATH}" >&2
  exit 1
fi
for checkpoint in "${E0_CHECKPOINT}" "${E1_CHECKPOINT}" "${E2_CHECKPOINT}"; do
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Required checkpoint not found: ${checkpoint}" >&2
    echo "Wait for all three independent training jobs to finish before evaluation." >&2
    exit 1
  fi
done

mkdir -p "${LOG_DIR}" "${PHASE1_DIR}"
JOB_LOG="${LOG_DIR}/${RUN_GROUP}_${RUN_CONTENT}_${PBS_JOBID:-manual}_seed${SEED}.log"
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
  "${PYTHON_BIN}" -m neuf.phase1_experiment
  --dataset "${DATASET_PATH}"
  --output-dir "${PHASE1_DIR}"
  --e0 "${SEED}=${E0_CHECKPOINT}"
  --e1 "${SEED}=${E1_CHECKPOINT}"
  --e2 "${SEED}=${E2_CHECKPOINT}"
  --allow-incomplete-seeds
)

echo "Host: $(hostname)"
echo "PBS job id: ${PBS_JOBID:-n/a}"
echo "Run group: ${RUN_GROUP}"
echo "Run content: ${RUN_CONTENT}"
echo "Seed: ${SEED}"
echo "Evaluation alpha values: ${ALPHA_VALUES}"
echo "Dataset: ${DATASET_PATH}"
echo "Output directory: ${PHASE1_DIR}"
echo "Job log: ${JOB_LOG}"
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

echo "Evaluation finished at: $(date --iso-8601=seconds)"
echo "Metrics: ${PHASE1_DIR}/metrics"
echo "Figures: ${PHASE1_DIR}/figures"
