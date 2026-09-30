#!/bin/bash -l
#PBS -N neuf_single_edge
#PBS -q gpu
#PBS -l walltime=02:00:00
#PBS -l nodes=1:ppn=16:gpus=1:gpu48
#PBS -l mem=128gb

set -euo pipefail

REPO_DIR=/misc/raid/zchen/Code/NeUF
PYTHON_ENV=/home/zchen/.conda/envs/neuf
RUN_ID="${RUN_ID:?通过 qsub -v RUN_ID=YYYYMMDD_trainNN 传入运行编号}"
RESULT_DIR="${REPO_DIR}/logs/${RUN_ID}/cerebral/index_0/HashObservedEdgeGatedSharpProfileSingle119"
LOG_DIR="${REPO_DIR}/qsub/logs/neuf/${RUN_ID}"
KERNEL_PREFIX="${TMPDIR:-/tmp}/neuf-single-edge-${PBS_JOBID}"

[[ "${RUN_ID}" =~ ^[0-9]{8}_train[0-9]{2,}$ ]]
[[ -x "${PYTHON_ENV}/bin/python" && ! -e "${RESULT_DIR}" ]]
mkdir -p "${LOG_DIR}" "${KERNEL_PREFIX}"
printf '%s\n' "${PBS_JOBID:-unknown}" > "${LOG_DIR}/job_id.txt"
finish() {
    local status=$?
    printf 'exit_code=%s\n' "${status}" > "${LOG_DIR}/status.txt"
    printf '结果目录: %s\n实验日志: %s/metrics\nqsub日志: %s\n' \
        "${RESULT_DIR}" "${RESULT_DIR%/*}/comparison" "${LOG_DIR}"
    printf 'qsub脚本: %s/qsub/neuf/single_frame_edge_ablation.sh\nJob ID/状态: %s / exit_code=%s\n' \
        "${REPO_DIR}" "${PBS_JOBID:-unknown}" "${status}"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

cd "${REPO_DIR}"
export PATH="${PYTHON_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLBACKEND=Agg
export MPLCONFIGDIR="${KERNEL_PREFIX}/matplotlib"
export IPYTHONDIR="${KERNEL_PREFIX}/ipython"
export JUPYTER_RUNTIME_DIR="${KERNEL_PREFIX}/runtime"
export SINGLE_FRAME_RESULT_DIR="${RESULT_DIR}"
export SINGLE_FRAME_PREVIEW_STEPS="2500 5000 7500 10000 15000 20000 30000 40000"

printf 'Host: %s\nFrame: 119\nSteps: 40000\nSeed: 3407\nBaseline: 20260916_train07\n' "$(hostname)"
printf 'Notebook: %s/analysis_workbench.ipynb\nResult: %s\n' "${REPO_DIR}" "${RESULT_DIR}"
"${PYTHON_ENV}/bin/python" -m ipykernel install --prefix "${KERNEL_PREFIX}" \
    --name neuf-single-edge --display-name 'Python 3 (NeUF single-frame edge)'
export JUPYTER_PATH="${KERNEL_PREFIX}/share/jupyter"

/usr/bin/python3 -u - "${REPO_DIR}/analysis_workbench.ipynb" <<'PY'
from copy import deepcopy
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

path = Path(sys.argv[1])
notebook = nbformat.read(path, as_version=4)
selected = [deepcopy(cell) for cell in notebook.cells
            if 'single-frame-edge-ablation' in cell.get('metadata', {}).get('tags', [])]
assert len(selected) == 3, len(selected)
execution = nbformat.v4.new_notebook(cells=selected, metadata=deepcopy(notebook.metadata))
execution.metadata.setdefault('kernelspec', {})['name'] = 'neuf-single-edge'
try:
    NotebookClient(execution, kernel_name='neuf-single-edge', timeout=7200,
                   resources={'metadata': {'path': str(path.parent.resolve())}}).execute()
finally:
    notebook = nbformat.read(path, as_version=4)
    executed = {cell.id: cell for cell in execution.cells}
    for cell in notebook.cells:
        if cell.id in executed and cell.cell_type == 'code':
            cell.outputs = executed[cell.id].get('outputs', [])
            cell.execution_count = executed[cell.id].get('execution_count')
    nbformat.write(notebook, path)
PY

mkdir -p "${RESULT_DIR}/run_config/source"
cp "${REPO_DIR}/qsub/neuf/single_frame_edge_ablation.sh" "${RESULT_DIR}/run_config/source/"
