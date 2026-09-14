#!/bin/bash -l
# Stack all cerebral ultrasound frames and export the central sagittal plane.

#PBS -N neuf_stack_sagittal
#PBS -q gpu
#PBS -l walltime=00:30:00
#PBS -l nodes=1:ppn=4:gpus=1:gpu48
#PBS -l mem=32gb
#PBS -o /misc/raid/zchen/Code/NeUF/qsub/logs/neuf/20260907_train01/stdout.log
#PBS -e /misc/raid/zchen/Code/NeUF/qsub/logs/neuf/20260907_train01/stderr.log
#PBS -M ziyi.chen@creatis.insa-lyon.fr
#PBS -m ae

set -euo pipefail

REPO_DIR="/misc/raid/zchen/Code/NeUF"
PYTHON_ENV="/home/zchen/.conda/envs/neuf"
NOTEBOOK="${REPO_DIR}/analysis_workbench.ipynb"
RESULT_DIR="${REPO_DIR}/logs/20260907_train01/cerebral_patient0/index_all/stacked_sagittal"
QSUB_LOG_DIR="${REPO_DIR}/qsub/logs/neuf/20260907_train01"
KERNEL_PREFIX="${TMPDIR:-/tmp}/neuf-stacked-sagittal-kernel-${PBS_JOBID:-manual}"

cd "${REPO_DIR}"
export PATH="${PYTHON_ENV}/bin:${PATH}"
mkdir -p "${KERNEL_PREFIX}"
"${PYTHON_ENV}/bin/python" -m ipykernel install \
  --prefix "${KERNEL_PREFIX}" \
  --name neuf-stacked-sagittal \
  --display-name "Python 3 (NeUF stacked sagittal)"
export JUPYTER_PATH="${KERNEL_PREFIX}/share/jupyter"
echo "${PBS_JOBID:-manual}" > "${QSUB_LOG_DIR}/job_id.txt"

echo "Notebook: ${NOTEBOOK}"
echo "Input: ${REPO_DIR}/data/cerebral_data/Pre_traitement_echo_v2/Recalage/Patient0/us_recal_original/us*.jpg"
echo "Center sagittal index: x=472 (zero-based)"
echo "Result directory: ${RESULT_DIR}"
echo "qsub log directory: ${QSUB_LOG_DIR}"
echo "Command: execute notebook cells tagged stacked-sagittal"

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
    if "stacked-sagittal" in cell.get("metadata", {}).get("tags", [])
]
if len(selected) != 2:
    raise RuntimeError(f"Expected 2 tagged sagittal cells, found {len(selected)}")

execution_notebook = nbformat.v4.new_notebook(
    cells=selected,
    metadata=deepcopy(notebook.metadata),
)
execution_notebook.metadata.setdefault("kernelspec", {})["name"] = "neuf-stacked-sagittal"
NotebookClient(
    execution_notebook,
    kernel_name="neuf-stacked-sagittal",
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
echo "Image: ${RESULT_DIR}/plots/stacked_sagittal_center_x0472.png"
echo "Metadata: ${RESULT_DIR}/metrics/volume_metadata.json"
