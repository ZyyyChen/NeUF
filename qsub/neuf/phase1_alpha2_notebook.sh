#!/bin/bash -l
# Execute the alpha=2 diagnostic notebook without changing the model alpha contract.

#PBS -N neuf_alpha2_diag
#PBS -q gpu
#PBS -l walltime=23:59:00
#PBS -l nodes=1:ppn=16:gpus=1:gpu48
#PBS -l mem=128gb
#PBS -o /misc/raid/zchen/Code/NeUF/qsub/logs/neuf/20260901_train02/stdout.log
#PBS -e /misc/raid/zchen/Code/NeUF/qsub/logs/neuf/20260901_train02/stderr.log
#PBS -M ziyi.chen@creatis.insa-lyon.fr
#PBS -m ae

set -euo pipefail

REPO_DIR="/misc/raid/zchen/Code/NeUF"
PYTHON_ENV="/home/zchen/.conda/envs/neuf"
NOTEBOOK="${REPO_DIR}/analysis_workbench.ipynb"
RESULT_DIR="${REPO_DIR}/logs/20260901_train02/cerebral_patient0/index_all/E2_alpha2_diagnostic"
QSUB_LOG_DIR="${REPO_DIR}/qsub/logs/neuf/20260901_train02"
KERNEL_PREFIX="${TMPDIR:-/tmp}/neuf-alpha2-kernel-${PBS_JOBID:-manual}"

cd "${REPO_DIR}"
export PATH="${PYTHON_ENV}/bin:${PATH}"
mkdir -p "${KERNEL_PREFIX}"
"${PYTHON_ENV}/bin/python" -m ipykernel install \
  --prefix "${KERNEL_PREFIX}" \
  --name neuf-alpha2 \
  --display-name "Python 3 (NeUF alpha2)"
export JUPYTER_PATH="${KERNEL_PREFIX}/share/jupyter"
echo "${PBS_JOBID:-manual}" > "${QSUB_LOG_DIR}/job_id.txt"
echo "Notebook: ${NOTEBOOK}"
echo "Result directory: ${RESULT_DIR}"
echo "qsub log directory: ${QSUB_LOG_DIR}"
echo "Command: ${PYTHON_ENV}/bin/jupyter nbconvert --to notebook --execute --inplace ${NOTEBOOK}"

"${PYTHON_ENV}/bin/jupyter" nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.kernel_name=neuf-alpha2 \
  --ExecutePreprocessor.timeout=600 \
  "${NOTEBOOK}"

echo "Status: complete"
echo "Figure: ${RESULT_DIR}/plots/step_020000/alpha_0_1_2_comparison.png"
echo "Metrics: ${RESULT_DIR}/metrics/alpha2_range_stats.csv"
