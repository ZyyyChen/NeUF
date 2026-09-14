#!/bin/bash -l
#PBS -N neuf_stack_animation
#PBS -q gpu
#PBS -l walltime=00:30:00
#PBS -l nodes=1:ppn=4:gpus=1:gpu48
#PBS -l mem=32gb

set -euo pipefail
REPO_DIR="/misc/raid/zchen/Code/NeUF"
PYTHON_ENV="/home/zchen/.conda/envs/neuf"
RUN_ID="20260910_train01"
RESULT_DIR="${REPO_DIR}/logs/${RUN_ID}/cerebral_patient0/index_0/stack_sagittal_animation"
QSUB_LOG_DIR="${REPO_DIR}/qsub/logs/neuf/${RUN_ID}"
KERNEL_PREFIX="${TMPDIR:-/tmp}/neuf-stack-animation-${PBS_JOBID}"
cd "${REPO_DIR}"
export PATH="${PYTHON_ENV}/bin:${PATH}"
export MPLCONFIGDIR="${KERNEL_PREFIX}/matplotlib"
export IPYTHONDIR="${KERNEL_PREFIX}/ipython"
export JUPYTER_RUNTIME_DIR="${KERNEL_PREFIX}/runtime"
mkdir -p "${KERNEL_PREFIX}"
"${PYTHON_ENV}/bin/python" -m ipykernel install --prefix "${KERNEL_PREFIX}" \
  --name neuf-stack-animation --display-name "Python 3 (NeUF stack animation)"
export JUPYTER_PATH="${KERNEL_PREFIX}/share/jupyter"
echo "${PBS_JOBID}" > "${QSUB_LOG_DIR}/job_id.txt"
trap 'status=$?; echo "Exit status: ${status}"; echo "结果目录: ${RESULT_DIR}"; echo "实验日志: ${RESULT_DIR}/metrics"; echo "qsub日志: ${QSUB_LOG_DIR}"; echo "qsub脚本: ${REPO_DIR}/qsub/neuf/stack_sagittal_animation.sh"; echo "Job ID: ${PBS_JOBID}"' EXIT
echo "Command: execute analysis_workbench.ipynb stack-animation cells, stage=${ANIMATION_STAGE:-render}"

/usr/bin/python3 - "${REPO_DIR}/analysis_workbench.ipynb" <<'PY'
from copy import deepcopy
from pathlib import Path
import os, sys
import nbformat
from nbclient import NotebookClient

path = Path(sys.argv[1])
notebook = nbformat.read(path, as_version=4)
tag = "stack-animation-audit" if os.environ.get("ANIMATION_STAGE") == "audit" else "stack-animation"
selected = [deepcopy(cell) for cell in notebook.cells if tag in cell.get("metadata", {}).get("tags", [])]
if not selected:
    raise RuntimeError(f"No cells for {tag}")
execution = nbformat.v4.new_notebook(cells=selected, metadata=deepcopy(notebook.metadata))
execution.metadata.setdefault("kernelspec", {})["name"] = "neuf-stack-animation"
try:
    NotebookClient(execution, kernel_name="neuf-stack-animation", timeout=1500,
                   resources={"metadata": {"path": str(path.parent)}}).execute()
finally:
    # 仅更新本次执行单元，保留其他分析工作。
    notebook = nbformat.read(path, as_version=4)
    executed = {cell["id"]: cell for cell in execution.cells}
    for cell in notebook.cells:
        if cell.cell_type == "code" and cell.get("id") in executed:
            cell["outputs"] = executed[cell["id"]].get("outputs", [])
            cell["execution_count"] = executed[cell["id"]].get("execution_count")
    nbformat.write(notebook, path)
PY
