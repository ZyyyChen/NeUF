#!/bin/bash -l
#PBS -N neuf_landmark_overlay
#PBS -q gpu
#PBS -l walltime=00:30:00
#PBS -l nodes=1:ppn=4:gpus=1:gpu48
#PBS -l mem=32gb

set -euo pipefail

# 复用已运行的 notebook 环境，仅执行本次获批的标注单元。
REPO_DIR="/misc/raid/zchen/Code/NeUF"
PYTHON_ENV="/home/zchen/.conda/envs/neuf"
RUN_ID="20260908_train01"
RESULT_DIR="${REPO_DIR}/logs/${RUN_ID}/cerebral_patient0/index_0/landmark_overlay"
QSUB_LOG_DIR="${REPO_DIR}/qsub/logs/neuf/${RUN_ID}"
KERNEL_PREFIX="${TMPDIR:-/tmp}/neuf-landmark-overlay-${PBS_JOBID}"
cd "${REPO_DIR}"
export PATH="${PYTHON_ENV}/bin:${PATH}"
export MPLCONFIGDIR="${KERNEL_PREFIX}/matplotlib"
export IPYTHONDIR="${KERNEL_PREFIX}/ipython"
export JUPYTER_RUNTIME_DIR="${KERNEL_PREFIX}/runtime"
mkdir -p "${KERNEL_PREFIX}"
"${PYTHON_ENV}/bin/python" -m ipykernel install \
  --prefix "${KERNEL_PREFIX}" --name neuf-landmark-overlay \
  --display-name "Python 3 (NeUF landmark overlay)"
export JUPYTER_PATH="${KERNEL_PREFIX}/share/jupyter"
echo "${PBS_JOBID}" > "${QSUB_LOG_DIR}/job_id.txt"
trap 'status=$?; echo "Exit status: ${status}"; echo "结果目录: ${RESULT_DIR}"; echo "实验日志: ${REPO_DIR}/analysis_workbench.ipynb (landmark-overlay)"; echo "qsub日志: ${QSUB_LOG_DIR}"; echo "qsub脚本: ${REPO_DIR}/qsub/neuf/landmark_overlay.sh"; echo "Job ID: ${PBS_JOBID}"' EXIT
echo "Command: execute analysis_workbench.ipynb cells tagged landmark-overlay"
echo "Inputs: Patient0 J35_2 saved stack/sagittal images and 11 landmark pairs"
echo "Result directory: ${RESULT_DIR}"

/usr/bin/python3 - "${REPO_DIR}/analysis_workbench.ipynb" <<'PY'
from copy import deepcopy
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

notebook_path = Path(sys.argv[1])
notebook = nbformat.read(notebook_path, as_version=4)
selected = [deepcopy(cell) for cell in notebook.cells
            if "landmark-overlay" in cell.get("metadata", {}).get("tags", [])]
if len(selected) != 2:
    raise RuntimeError(f"Expected 2 landmark-overlay cells, found {len(selected)}")
execution_notebook = nbformat.v4.new_notebook(cells=selected, metadata=deepcopy(notebook.metadata))
execution_notebook.metadata.setdefault("kernelspec", {})["name"] = "neuf-landmark-overlay"
try:
    NotebookClient(execution_notebook, kernel_name="neuf-landmark-overlay", timeout=600,
                   resources={"metadata": {"path": str(notebook_path.parent)}}).execute()
finally:
    # 重新读取，避免覆盖运行期间其他分析单元的更新。
    notebook = nbformat.read(notebook_path, as_version=4)
    executed_by_id = {cell["id"]: cell for cell in execution_notebook.cells}
    for cell in notebook.cells:
        executed = executed_by_id.get(cell.get("id"))
        if executed is not None and cell.cell_type == "code":
            cell["outputs"] = executed.get("outputs", [])
            cell["execution_count"] = executed.get("execution_count")
    nbformat.write(notebook, notebook_path)
PY
