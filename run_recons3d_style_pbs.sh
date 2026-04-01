#!/bin/bash -l
#PBS -N recons3d_style
#PBS -q route
#PBS -l walltime=30:00:00
#PBS -l nodes=1:ppn=16
#PBS -l mem=128gb
#PBS -j oe
#PBS -o /home/zchen/history/recons3d_style.pbs.log
#PBS -M ziyi.chen@creatis.insa-lyon.fr
#PBS -m ae

set -euo pipefail

REPO_DIR="/misc/raid/zchen/Code/NeUF"
PYTHON_BIN="/home/zchen/.conda/envs/neuf/bin/python"
SCRIPT_PATH="${REPO_DIR}/export_recons3d_style_baseline.py"
CKPT_PATH="${REPO_DIR}/latest/ckpt.pkl"
OUTPUT_DIR="${REPO_DIR}/exports/recons3d_style_cluster"
LOG_DIR="/home/zchen/history/recons3d_style"

mkdir -p "${LOG_DIR}"
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

echo "Host: $(hostname)"
echo "Date: $(date --iso-8601=seconds)"
echo "PBS job id: ${PBS_JOBID:-n/a}"
echo "PBS queue: ${PBS_QUEUE:-n/a}"
echo "Working dir: ${REPO_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "Checkpoint: ${CKPT_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Threads: ${THREADS}"

time "${PYTHON_BIN}" "${SCRIPT_PATH}" \
  --ckpt "${CKPT_PATH}" \
  --output "${OUTPUT_DIR}" \
  --chunk-size 250000

echo "Finished at: $(date --iso-8601=seconds)"
