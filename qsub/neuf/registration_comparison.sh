#!/bin/bash -l
#PBS -N neuf_registration
#PBS -q route
#PBS -l walltime=04:00:00
#PBS -l nodes=1:ppn=4
#PBS -l mem=32gb

set -euo pipefail
REPO_DIR=/misc/raid/zchen/Code/NeUF
PYTHON_BIN=/home/zchen/.conda/envs/neuf/bin/python
RUN_ID="${RUN_ID:?请传入 RUN_ID}"
MODE="${MODE:-inventory}"
[[ "${RUN_ID}" =~ ^[0-9]{8}_train[0-9]{2,}$ ]]
RESULT_DIR="${REPO_DIR}/logs/${RUN_ID}/registration/index_all/legacy_current"
QSUB_LOG_DIR="${REPO_DIR}/qsub/logs/neuf/${RUN_ID}/${MODE}"
mkdir -p "${QSUB_LOG_DIR}"
printf '%s\n' "${PBS_JOBID}" > "${QSUB_LOG_DIR}/job_id.txt"
finish() {
    local status=$?
    printf 'exit_code=%s\n' "${status}" > "${QSUB_LOG_DIR}/status.txt"
    printf '结果目录: %s\n实验日志: %s/metrics\nqsub日志: %s\nqsub脚本: %s/qsub/neuf/registration_comparison.sh\nJob ID/状态: %s / exit_code=%s\n' \
        "${RESULT_DIR}" "${RESULT_DIR}" "${QSUB_LOG_DIR}" "${REPO_DIR}" "${PBS_JOBID}" "${status}"
}
trap finish EXIT
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/neuf-registration-${PBS_JOBID}/matplotlib"
mkdir -p "${RESULT_DIR}/metrics"
exec > >(tee "${RESULT_DIR}/metrics/${MODE}_execution.log") 2>&1
if [[ "${MODE}" == report ]]; then
    KERNEL_PREFIX="${TMPDIR:-/tmp}/neuf-registration-${PBS_JOBID}/kernel"
    export REGISTRATION_COMPARISON_OUTPUT="${RESULT_DIR}"
    export IPYTHONDIR="${TMPDIR:-/tmp}/neuf-registration-${PBS_JOBID}/ipython"
    export JUPYTER_RUNTIME_DIR="${TMPDIR:-/tmp}/neuf-registration-${PBS_JOBID}/runtime"
    "${PYTHON_BIN}" -m ipykernel install --prefix "${KERNEL_PREFIX}" --name neuf-registration --display-name "NeUF registration comparison"
    export JUPYTER_PATH="${KERNEL_PREFIX}/share/jupyter"
    printf 'Command: execute notebook cells tagged registration-comparison\n'
    /usr/bin/python3 -u - "${REPO_DIR}/analysis_workbench.ipynb" <<'PY'
from copy import deepcopy
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient

path = Path(sys.argv[1])
notebook = nbformat.read(path, as_version=4)
cells = [deepcopy(cell) for cell in notebook.cells if "registration-comparison" in cell.metadata.get("tags", [])]
if len(cells) != 2:
    raise RuntimeError(f"需要两个配准对比单元，实际 {len(cells)}")
execution = nbformat.v4.new_notebook(cells=cells, metadata=deepcopy(notebook.metadata))
execution.metadata["kernelspec"]["name"] = "neuf-registration"
try:
    NotebookClient(execution, kernel_name="neuf-registration", timeout=3600,
                   resources={"metadata": {"path": str(path.parent)}}).execute()
finally:
    notebook = nbformat.read(path, as_version=4)
    executed = {cell["id"]: cell for cell in execution.cells}
    for cell in notebook.cells:
        if cell["id"] in executed and cell.cell_type == "code":
            cell.outputs = executed[cell["id"]].get("outputs", [])
            cell.execution_count = executed[cell["id"]].get("execution_count")
    nbformat.write(notebook, path)
PY
else
    if [[ "${MODE}" == inventory ]]; then
        mkdir -p "${RESULT_DIR}/run_config/source"
        cp "${REPO_DIR}/code_seg_3D/lib/neoBrainSeg/data_registration_v2.py" "${RESULT_DIR}/run_config/source/legacy_original.py"
        cp "${REPO_DIR}/../code_data/data_registration_v2.py" "${RESULT_DIR}/run_config/source/current_original.py"
        cp "${REPO_DIR}/neuf/registration_comparison.py" "${RESULT_DIR}/run_config/source/registration_comparison.py"
        cp "${REPO_DIR}/qsub/neuf/registration_comparison.sh" "${RESULT_DIR}/run_config/source/registration_comparison.sh"
    fi
    printf 'Command: %s -m neuf.registration_comparison --mode %s --output %s\n' "${PYTHON_BIN}" "${MODE}" "${RESULT_DIR}"
    "${PYTHON_BIN}" -u -m neuf.registration_comparison --repo "${REPO_DIR}" --output "${RESULT_DIR}" --mode "${MODE}"
fi
