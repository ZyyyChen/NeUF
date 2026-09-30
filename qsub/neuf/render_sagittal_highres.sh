#!/bin/bash -l
#PBS -N neuf_sag_highres
#PBS -q gpu
#PBS -l walltime=01:00:00
#PBS -l nodes=1:ppn=8:gpus=1:gpu48
#PBS -l mem=48gb

set -euo pipefail

REPO_DIR=/misc/raid/zchen/Code/NeUF
PYTHON_BIN=/home/zchen/.conda/envs/neuf/bin/python
RUN_ID="${RUN_ID:?通过 qsub -v RUN_ID=YYYYMMDD_trainNN 传入运行编号}"
RESULT_DIR="${REPO_DIR}/logs/${RUN_ID}/cerebral/index_0/HashObservedEdgeGatedSharp"
LOG_DIR="${REPO_DIR}/qsub/logs/neuf/${RUN_ID}"
KERNEL_PREFIX="${TMPDIR:-/tmp}/neuf-sag-highres-${PBS_JOBID}"

[[ "${RUN_ID}" =~ ^[0-9]{8}_train[0-9]{2,}$ ]]
[[ -x "${PYTHON_BIN}" && ! -e "${RESULT_DIR}" ]]
mkdir -p "${LOG_DIR}" "${KERNEL_PREFIX}"
printf '%s\n' "${PBS_JOBID:-unknown}" > "${LOG_DIR}/job_id.txt"
finish() {
    local status=$?
    printf 'exit_code=%s\n' "${status}" > "${LOG_DIR}/status.txt"
    printf '结果目录: %s\n实验日志: %s/metrics\nqsub日志: %s\n' \
        "${RESULT_DIR}" "${RESULT_DIR}" "${LOG_DIR}"
    printf 'qsub脚本: %s/qsub/neuf/render_sagittal_highres.sh\nJob ID/状态: %s / exit_code=%s\n' \
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
export HIGHRES_SAGITTAL_RESULT_DIR="${RESULT_DIR}"

printf 'Host: %s\nCheckpoint: %s\nNative spacing: checkpoint acquisition pixel spacing\n' \
    "$(hostname)" "${REPO_DIR}/logs/20260916_train02/cerebral/index_all/HashObservedEdgeGatedSharp/checkpoints/latest.pt"
"${PYTHON_BIN}" -m ipykernel install --prefix "${KERNEL_PREFIX}" \
    --name neuf-sag-highres --display-name 'Python 3 (NeUF sagittal highres)'
export JUPYTER_PATH="${KERNEL_PREFIX}/share/jupyter"
/usr/bin/python3 -u - <<'PY'
from copy import deepcopy
from pathlib import Path

import nbformat
from nbclient import NotebookClient

path = Path("analysis_workbench.ipynb")
notebook = nbformat.read(path, as_version=4)
selected = [deepcopy(cell) for cell in notebook.cells
            if "highres-sagittal" in cell.metadata.get("tags", [])]
assert len(selected) == 2, len(selected)
execution = nbformat.v4.new_notebook(cells=selected, metadata=deepcopy(notebook.metadata))
execution.metadata.setdefault("kernelspec", {})["name"] = "neuf-sag-highres"
NotebookClient(execution, kernel_name="neuf-sag-highres", timeout=1800,
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
cp "${REPO_DIR}/qsub/neuf/render_sagittal_highres.sh" "${RESULT_DIR}/run_config/"
