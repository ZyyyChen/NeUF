#!/bin/bash -l
#PBS -N neuf_hash_roi
#PBS -q gpu
#PBS -l walltime=00:30:00
#PBS -l nodes=1:ppn=4:gpus=1:gpu48
#PBS -l mem=32gb

set -euo pipefail
REPO_DIR="/misc/raid/zchen/Code/NeUF"
RUN_ID="20260908_train03"
ROI_STAGE="${ROI_STAGE:-render}"
RESULT_DIR="${REPO_DIR}/logs/${RUN_ID}/cerebral_patient0/index_all/hash_grid_nmax_roi"
QSUB_LOG_DIR="${REPO_DIR}/qsub/logs/neuf/${RUN_ID}"
KERNEL_PREFIX="${TMPDIR:-/tmp}/neuf-hash-roi-${PBS_JOBID}"
cd "${REPO_DIR}"
export PATH="/home/zchen/.conda/envs/neuf/bin:${PATH}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${KERNEL_PREFIX}/matplotlib"
export IPYTHONDIR="${KERNEL_PREFIX}/ipython"
export JUPYTER_RUNTIME_DIR="${KERNEL_PREFIX}/runtime"
export PYTHONDONTWRITEBYTECODE=1
export ROI_STAGE
mkdir -p "${RESULT_DIR}" "${KERNEL_PREFIX}"
exec > >(tee "${RESULT_DIR}/execution_${ROI_STAGE}.log") 2>&1
trap 'status=$?; echo "Exit status: ${status}"; echo "结果目录: ${RESULT_DIR}"; echo "实验日志: ${RESULT_DIR}"; echo "qsub日志: ${QSUB_LOG_DIR}"; echo "qsub脚本: ${REPO_DIR}/qsub/neuf/hash_grid_roi.sh"; echo "Job ID: ${PBS_JOBID}"' EXIT
python -m ipykernel install --prefix "${KERNEL_PREFIX}" --name neuf-hash-roi --display-name "Python 3 (NeUF hash ROI)"
export JUPYTER_PATH="${KERNEL_PREFIX}/share/jupyter"
echo "Command: execute analysis_workbench.ipynb hash-grid-roi setup and ${ROI_STAGE} cells"
# 仅执行本次新增单元，保存输出时重新读取 notebook，保留其他分析。
/usr/bin/python3 -u - <<'PY'
import os
from copy import deepcopy
from pathlib import Path
import nbformat
from nbclient import NotebookClient

path = Path('analysis_workbench.ipynb')
notebook = nbformat.read(path, as_version=4)
tags = {'hash-grid-roi-setup', 'hash-grid-roi-' + os.environ['ROI_STAGE']}
selected = [deepcopy(c) for c in notebook.cells if c.cell_type == 'code' and tags.intersection(c.metadata.get('tags', []))]
assert len(selected) >= 2, len(selected)
execution = nbformat.v4.new_notebook(cells=selected)
try:
    NotebookClient(execution, kernel_name='neuf-hash-roi', timeout=1500, resources={'metadata': {'path': str(Path.cwd())}}).execute()
finally:
    notebook = nbformat.read(path, as_version=4)
    executed = {c.id: c for c in execution.cells}
    for cell in notebook.cells:
        if cell.id in executed:
            cell.outputs = executed[cell.id].get('outputs', [])
            cell.execution_count = executed[cell.id].get('execution_count')
    nbformat.write(notebook, path)
print('Notebook stage completed:', os.environ['ROI_STAGE'])
PY
