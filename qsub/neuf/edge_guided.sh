#!/bin/bash -l
#PBS -N neuf_edge_guided
#PBS -q gpu
#PBS -l walltime=02:00:00
#PBS -l nodes=1:ppn=16:gpus=1:gpu48
#PBS -l mem=128gb

set -euo pipefail

WORKSPACE_ROOT=/misc/raid/zchen/Code
REPO_DIR="${WORKSPACE_ROOT}/NeUF"
PYTHON_BIN="${PYTHON_BIN:-/home/zchen/.conda/envs/neuf/bin/python}"
RUN_ID="${RUN_ID:?通过 qsub -v RUN_ID=YYYYMMDD_trainNN 传入运行编号}"
MODE="${MODE:-smoke}"
TARGET_RUN_ID="${TARGET_RUN_ID:-${RUN_ID}}"
RESULT_DIR="${REPO_DIR}/logs/${TARGET_RUN_ID}/cerebral/index_all"
QSUB_LOG_DIR="${REPO_DIR}/qsub/logs/neuf/${RUN_ID}"
DATASET_PATH="${REPO_DIR}/data/cerebral_data/Pre_traitement_echo_v2/Recalage/Patient0/us_recal_original/baked_dataset_physical.pkl"
TEACHER_PATH="${WORKSPACE_ROOT}/NLSTV/logs/20260903_train03/cerebral/index_0/nlstv_lambda009/predictions/cerebral_edges.mat"
SOURCE_IMAGES="${WORKSPACE_ROOT}/UltraNeRF-Studio/data/cerebral/patient_0_convex/images.npy"
VARIANTS=(HashObservedEdgeGuided HashObservedEdgeGated HashObservedEdgeGatedSharp)
MODEL_VARIANT="${MODEL_VARIANT:-}"
if [[ -n "${MODEL_VARIANT}" ]]; then
    [[ "${MODEL_VARIANT}" == HashObservedEdgeGatedSharp ||
       "${MODEL_VARIANT}" == HashObservedEdgeGatedSharpFocused ||
       "${MODEL_VARIANT}" == HashObservedEdgeGatedSharpProfile ]]
    VARIANTS=("${MODEL_VARIANT}")
fi

[[ "${RUN_ID}" =~ ^[0-9]{8}_train[0-9]{2,}$ ]]
[[ "${MODE}" == smoke || "${MODE}" == train || "${MODE}" == report ]]
[[ -x "${PYTHON_BIN}" ]]
if [[ "${MODE}" == report ]]; then
    [[ -d "${RESULT_DIR}" ]]
else
    [[ "${TARGET_RUN_ID}" == "${RUN_ID}" && ! -e "${RESULT_DIR}" ]]
fi
mkdir -p "${QSUB_LOG_DIR}"
printf '%s\n' "${PBS_JOBID:-unknown}" > "${QSUB_LOG_DIR}/job_id.txt"
finish() {
    local status=$?
    printf 'exit_code=%s\n' "${status}" > "${QSUB_LOG_DIR}/status.txt"
    printf '结果目录: %s\n实验日志: %s/<MODEL>/metrics\nqsub日志: %s\n' \
        "${RESULT_DIR}" "${RESULT_DIR}" "${QSUB_LOG_DIR}"
    printf 'qsub脚本: %s/qsub/neuf/edge_guided.sh\nJob ID/状态: %s / exit_code=%s\n' \
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

if [[ "${MODE}" != report ]]; then
    if [[ "${MODE}" == smoke ]]; then
        STEPS=10
        WIDTH=64
        PATCH_SIZE=48
        VOLUME_SPACING=2.0
        HASH_LEVELS=4
        COARSE_HASH_LEVELS=2
        LOG2_HASHMAP_SIZE=15
        HASH_FINEST_RESOLUTION=64
    else
        STEPS=40000
        WIDTH=256
        PATCH_SIZE=64
        VOLUME_SPACING=0.75
        HASH_LEVELS=16
        COARSE_HASH_LEVELS=8
        LOG2_HASHMAP_SIZE=19
        HASH_FINEST_RESOLUTION=512
    fi
    cmd=("${PYTHON_BIN}" -m neuf.edge_field.workflow
         --dataset "${DATASET_PATH}" --teacher "${TEACHER_PATH}"
         --source-images "${SOURCE_IMAGES}" --output "${RESULT_DIR}"
         --steps "${STEPS}" --patches 5 --patch-size "${PATCH_SIZE}"
         --seed 3407 --width "${WIDTH}" --bands 10 --lr 0.001 --encoding hash
         --hash-levels "${HASH_LEVELS}" --hash-features 2
         --coarse-hash-levels "${COARSE_HASH_LEVELS}" --detail-scale 2
         --log2-hashmap-size "${LOG2_HASHMAP_SIZE}"
         --hash-base-resolution 16 --hash-finest-resolution "${HASH_FINEST_RESOLUTION}"
         --guided-fraction 0.4 --sharpness-weight 0.05
         --volume-spacing "${VOLUME_SPACING}"
         --variants "${VARIANTS[@]}")
    if [[ "${MODE}" == smoke ]]; then
        cmd+=(--smoke)
    fi
    printf 'Host: %s\nMode: %s\nCommand:' "$(hostname)" "${MODE}"
    printf ' %q' "${cmd[@]}"
    printf '\n'
    "${cmd[@]}"
else
    printf 'Host: %s\nMode: report\nTarget result: %s\n' "$(hostname)" "${RESULT_DIR}"
fi

if [[ "${MODE}" == train || "${MODE}" == report ]]; then
    export EDGE_GUIDED_RESULT_DIR="${RESULT_DIR}"
    NOTEBOOK_TAG=nlstv-edge-guided
    if [[ "${MODEL_VARIANT}" == HashObservedEdgeGatedSharpFocused ]]; then
        NOTEBOOK_TAG=edge-focused-comparison
    elif [[ "${MODEL_VARIANT}" == HashObservedEdgeGatedSharpProfile ]]; then
        NOTEBOOK_TAG=edge-profile-comparison
    elif [[ -n "${MODEL_VARIANT}" ]]; then
        NOTEBOOK_TAG=pose-mm-training-comparison
    fi
    export EDGE_GUIDED_NOTEBOOK_TAG="${NOTEBOOK_TAG}"
    KERNEL_PREFIX="${TMPDIR:-/tmp}/neuf-edge-guided-${PBS_JOBID}"
    mkdir -p "${KERNEL_PREFIX}"
    "${PYTHON_BIN}" -m ipykernel install --prefix "${KERNEL_PREFIX}" \
        --name neuf-edge-guided --display-name "Python 3 (NeUF edge guided)"
    export JUPYTER_PATH="${KERNEL_PREFIX}/share/jupyter"
    export MPLCONFIGDIR="${KERNEL_PREFIX}/matplotlib"
    /usr/bin/python3 -u - <<'PY'
from copy import deepcopy
from pathlib import Path
import os
import nbformat
from nbclient import NotebookClient

path = Path("analysis_workbench.ipynb")
notebook = nbformat.read(path, as_version=4)
tag = os.environ["EDGE_GUIDED_NOTEBOOK_TAG"]
selected = [deepcopy(cell) for cell in notebook.cells
            if tag in cell.metadata.get("tags", [])]
assert len(selected) == 2, len(selected)
for cell in selected:
    if cell.cell_type != "code":
        cell.pop("outputs", None)
        cell.pop("execution_count", None)
execution = nbformat.v4.new_notebook(cells=selected, metadata=deepcopy(notebook.metadata))
execution.metadata.setdefault("kernelspec", {})["name"] = "neuf-edge-guided"
NotebookClient(execution, kernel_name="neuf-edge-guided", timeout=900,
               resources={"metadata": {"path": str(path.parent.resolve())}}).execute()
notebook = nbformat.read(path, as_version=4)
executed = {cell.id: cell for cell in execution.cells}
for cell in notebook.cells:
    if cell.id in executed and cell.cell_type == "code":
        cell.outputs = executed[cell.id].get("outputs", [])
        cell.execution_count = executed[cell.id].get("execution_count")
nbformat.write(notebook, path)
PY
fi

mkdir -p "${RESULT_DIR}/comparison/run_config/source"
cp "${REPO_DIR}/qsub/neuf/edge_guided.sh" "${RESULT_DIR}/comparison/run_config/source/"
