#!/bin/bash -l
#PBS -N neuf_e0_e1_edge
#PBS -q gpu
#PBS -l walltime=00:30:00
#PBS -l nodes=1:ppn=4:gpus=1:gpu48
#PBS -l mem=32gb

set -euo pipefail

REPO_DIR=/misc/raid/zchen/Code/NeUF
PYTHON_BIN=/home/zchen/.conda/envs/neuf/bin/python
RUN_ID="${RUN_ID:?通过 qsub -v RUN_ID=YYYYMMDD_trainNN 传入运行编号}"
RESULT_DIR="${REPO_DIR}/logs/${RUN_ID}/cerebral/index_0/EdgeWidthE0E1Audit"
LOG_DIR="${REPO_DIR}/qsub/logs/neuf/${RUN_ID}"
KERNEL_PREFIX="${TMPDIR:-/tmp}/neuf-e0-e1-edge-${PBS_JOBID}"

[[ "${RUN_ID}" =~ ^[0-9]{8}_train[0-9]{2,}$ ]]
[[ -x "${PYTHON_BIN}" && ! -e "${RESULT_DIR}" ]]
mkdir -p "${LOG_DIR}" "${KERNEL_PREFIX}"
printf '%s\n' "${PBS_JOBID:-unknown}" > "${LOG_DIR}/job_id.txt"
finish() {
    local status=$?
    printf 'exit_code=%s\n' "${status}" > "${LOG_DIR}/status.txt"
    printf '结果目录: %s\n实验日志: %s/metrics\nqsub日志: %s\n' \
        "${RESULT_DIR}" "${RESULT_DIR}" "${LOG_DIR}"
    printf 'qsub脚本: %s/qsub/neuf/compare_edge_e0_e1.sh\nJob ID/状态: %s / exit_code=%s\n' \
        "${REPO_DIR}" "${PBS_JOBID:-unknown}" "${status}"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLBACKEND=Agg
export MPLCONFIGDIR="${KERNEL_PREFIX}/matplotlib"
export EDGE_WIDTH_AUDIT_RESULT_DIR="${RESULT_DIR}"

printf 'Host: %s\nFrames: 236 226 216 206\nModels: Observed E0 E1 Current\n' "$(hostname)"
"${PYTHON_BIN}" -m ipykernel install --prefix "${KERNEL_PREFIX}" \
    --name neuf-e0-e1-edge --display-name 'Python 3 (NeUF E0 E1 edge audit)'
export JUPYTER_PATH="${KERNEL_PREFIX}/share/jupyter"
/usr/bin/python3 -u - <<'PY'
from copy import deepcopy
from pathlib import Path

import nbformat
from nbclient import NotebookClient

path = Path("analysis_workbench.ipynb")
notebook = nbformat.read(path, as_version=4)
selected = [deepcopy(cell) for cell in notebook.cells
            if "e0-e1-edge-audit" in cell.metadata.get("tags", [])]
assert len(selected) == 2, len(selected)
execution = nbformat.v4.new_notebook(cells=selected, metadata=deepcopy(notebook.metadata))
execution.metadata.setdefault("kernelspec", {})["name"] = "neuf-e0-e1-edge"
NotebookClient(execution, kernel_name="neuf-e0-e1-edge", timeout=1800,
               resources={"metadata": {"path": str(path.parent.resolve())}}).execute()
notebook = nbformat.read(path, as_version=4)
executed = {cell.id: cell for cell in execution.cells}
for cell in notebook.cells:
    if cell.id in executed and cell.cell_type == "code":
        cell.outputs = executed[cell.id].get("outputs", [])
        cell.execution_count = executed[cell.id].get("execution_count")
nbformat.write(notebook, path)
PY

mkdir -p "${RESULT_DIR}/run_config"
cp "${REPO_DIR}/qsub/neuf/compare_edge_e0_e1.sh" "${RESULT_DIR}/run_config/"
