#!/bin/bash -l
#PBS -N neuf_pose_audit
#PBS -q gpu
#PBS -l walltime=01:00:00
#PBS -l nodes=1:ppn=8:gpus=1:gpu48
#PBS -l mem=128gb

set -euo pipefail

WORKSPACE_ROOT=/misc/raid/zchen/Code
REPO_DIR="${WORKSPACE_ROOT}/NeUF"
PYTHON_ENV=/home/zchen/.conda/envs/neuf
RUN_ID="${RUN_ID:?通过 qsub -v RUN_ID=YYYYMMDD_trainNN 传入运行编号}"
RESULT_DIR="${WORKSPACE_ROOT}/logs/${RUN_ID}/cerebral/index_0/PoseCoordinateAudit"
QSUB_LOG_DIR="${WORKSPACE_ROOT}/qsub/logs/neuf/${RUN_ID}"
KERNEL_PREFIX="${TMPDIR:-/tmp}/neuf-pose-audit-${PBS_JOBID}"

[[ "${RUN_ID}" =~ ^[0-9]{8}_train[0-9]{2,}$ ]]
[[ -x "${PYTHON_ENV}/bin/python" ]]
[[ ! -e "${RESULT_DIR}" ]]
mkdir -p "${QSUB_LOG_DIR}" "${KERNEL_PREFIX}"
printf '%s\n' "${PBS_JOBID:-unknown}" > "${QSUB_LOG_DIR}/job_id.txt"

finish() {
    local status=$?
    printf 'exit_code=%s\n' "${status}" > "${QSUB_LOG_DIR}/status.txt"
    printf '结果目录: %s\n实验日志: %s/metrics\nqsub日志: %s\n' \
        "${RESULT_DIR}" "${RESULT_DIR}" "${QSUB_LOG_DIR}"
    printf 'qsub脚本: %s/qsub/neuf/pose_coordinate_audit.sh\nJob ID/状态: %s / exit_code=%s\n' \
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
export POSE_AUDIT_RESULT_DIR="${RESULT_DIR}"
export MPLCONFIGDIR="${KERNEL_PREFIX}/matplotlib"
export IPYTHONDIR="${KERNEL_PREFIX}/ipython"
export JUPYTER_RUNTIME_DIR="${KERNEL_PREFIX}/runtime"

"${PYTHON_ENV}/bin/python" -m ipykernel install \
    --prefix "${KERNEL_PREFIX}" --name neuf-pose-audit \
    --display-name "Python 3 (NeUF pose coordinate audit)"
export JUPYTER_PATH="${KERNEL_PREFIX}/share/jupyter"

printf 'Host: %s\nCommand: execute analysis_workbench.ipynb cells tagged pose-coordinate-audit\n' "$(hostname)"
printf 'Input dataset: %s\n' "${REPO_DIR}/data/cerebral_data/Pre_traitement_echo_v2/Recalage/Patient0/us_recal_original/baked_dataset_physical.pkl"
printf 'Result directory: %s\n' "${RESULT_DIR}"

/usr/bin/python3 -u - "${REPO_DIR}/analysis_workbench.ipynb" <<'PY'
from copy import deepcopy
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

notebook_path = Path(sys.argv[1])
notebook = nbformat.read(notebook_path, as_version=4)
selected = [deepcopy(cell) for cell in notebook.cells
            if "pose-coordinate-audit" in cell.get("metadata", {}).get("tags", [])]
if len(selected) != 2:
    raise RuntimeError(f"Expected 2 pose-coordinate-audit cells, found {len(selected)}")
execution = nbformat.v4.new_notebook(cells=selected, metadata=deepcopy(notebook.metadata))
execution.metadata.setdefault("kernelspec", {})["name"] = "neuf-pose-audit"
try:
    NotebookClient(
        execution,
        kernel_name="neuf-pose-audit",
        timeout=1800,
        resources={"metadata": {"path": str(notebook_path.parent.resolve())}},
    ).execute()
finally:
    notebook = nbformat.read(notebook_path, as_version=4)
    executed = {cell["id"]: cell for cell in execution.cells}
    for cell in notebook.cells:
        replacement = executed.get(cell.get("id"))
        if replacement is not None and cell.cell_type == "code":
            cell["outputs"] = replacement.get("outputs", [])
            cell["execution_count"] = replacement.get("execution_count")
    nbformat.write(notebook, notebook_path)
PY

mkdir -p "${RESULT_DIR}/run_config/source"
cp "${REPO_DIR}/qsub/neuf/pose_coordinate_audit.sh" "${RESULT_DIR}/run_config/source/"
