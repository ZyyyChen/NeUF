#!/bin/bash -l
# Execute only the Phase 1 native-resolution checkpoint montage notebook cells.

#PBS -N neuf_ckpt_montage
#PBS -q gpu
#PBS -l walltime=00:30:00
#PBS -l nodes=1:ppn=4:gpus=1:gpu48
#PBS -l mem=32gb
#PBS -o /misc/raid/zchen/Code/NeUF/qsub/logs/neuf/20260902_train01/stdout.log
#PBS -e /misc/raid/zchen/Code/NeUF/qsub/logs/neuf/20260902_train01/stderr.log
#PBS -M ziyi.chen@creatis.insa-lyon.fr
#PBS -m ae

set -euo pipefail

REPO_DIR="/misc/raid/zchen/Code/NeUF"
PYTHON_ENV="/home/zchen/.conda/envs/neuf"
NOTEBOOK="${REPO_DIR}/analysis_workbench.ipynb"
RESULT_DIR="${REPO_DIR}/logs/20260902_train01/cerebral_patient0/index_all/phase1_ckpt_native_montage"
QSUB_LOG_DIR="${REPO_DIR}/qsub/logs/neuf/20260902_train01"
KERNEL_PREFIX="${TMPDIR:-/tmp}/neuf-native-montage-kernel-${PBS_JOBID:-manual}"

cd "${REPO_DIR}"
export PATH="${PYTHON_ENV}/bin:${PATH}"
mkdir -p "${KERNEL_PREFIX}"
"${PYTHON_ENV}/bin/python" -m ipykernel install \
  --prefix "${KERNEL_PREFIX}" \
  --name neuf-native-montage \
  --display-name "Python 3 (NeUF native montage)"
export JUPYTER_PATH="${KERNEL_PREFIX}/share/jupyter"
echo "${PBS_JOBID:-manual}" > "${QSUB_LOG_DIR}/job_id.txt"

echo "Notebook: ${NOTEBOOK}"
echo "Result directory: ${RESULT_DIR}"
echo "qsub log directory: ${QSUB_LOG_DIR}"
echo "Command: execute notebook cells tagged phase1-native-montage"

/usr/bin/python3 - "${NOTEBOOK}" <<'PY'
from copy import deepcopy
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

notebook_path = Path(sys.argv[1])
notebook = nbformat.read(notebook_path, as_version=4)
selected = [
    deepcopy(cell)
    for cell in notebook.cells
    if "phase1-native-montage" in cell.get("metadata", {}).get("tags", [])
]
if len(selected) != 3:
    raise RuntimeError(f"Expected 3 tagged montage cells, found {len(selected)}")

execution_notebook = nbformat.v4.new_notebook(
    cells=selected,
    metadata=deepcopy(notebook.metadata),
)
execution_notebook.metadata.setdefault("kernelspec", {})["name"] = "neuf-native-montage"
NotebookClient(
    execution_notebook,
    kernel_name="neuf-native-montage",
    timeout=600,
    resources={"metadata": {"path": str(notebook_path.parent)}},
).execute()

executed_by_id = {cell["id"]: cell for cell in execution_notebook.cells}
for cell in notebook.cells:
    executed = executed_by_id.get(cell.get("id"))
    if executed is not None:
        cell["outputs"] = executed.get("outputs", [])
        cell["execution_count"] = executed.get("execution_count")
nbformat.write(notebook, notebook_path)
PY

echo "Status: complete"
echo "Predictions: ${RESULT_DIR}/plots/all_checkpoints_predictions.png"
echo "Errors: ${RESULT_DIR}/plots/all_checkpoints_absolute_errors.png"
echo "E2 components: ${RESULT_DIR}/plots/e2_all_checkpoints_components.png"
echo "Dimensions: ${RESULT_DIR}/metrics/image_dimensions.csv"
