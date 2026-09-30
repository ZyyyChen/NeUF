#!/bin/bash -l
#PBS -N neuf_pose_mm
#PBS -q gpu
#PBS -l walltime=02:00:00
#PBS -l nodes=1:ppn=8:gpus=1:gpu48
#PBS -l mem=128gb

set -euo pipefail

REPO_DIR=/misc/raid/zchen/Code/NeUF
PYTHON_BIN=/home/zchen/.conda/envs/neuf/bin/python
RUN_ID="${RUN_ID:?通过 qsub -v RUN_ID=YYYYMMDD_trainNN 传入运行编号}"
RESUME="${RESUME:-0}"
RAW_PARENT="${REPO_DIR}/data/cerebral_data/Pre_traitement_echo_v2/Recalage/Patient0"
RAW_DIR="${RAW_PARENT}/us_recal_original"
INFOS_PATH="${RAW_DIR}/infos.json"
DATASET_PATH="${RAW_DIR}/baked_dataset_physical.pkl"
RESULT_DIR="${REPO_DIR}/logs/${RUN_ID}/cerebral/index_0/PoseMmCorrection"
QSUB_LOG_DIR="${REPO_DIR}/qsub/logs/neuf/${RUN_ID}"
STAGED_INFOS_DIR="${RESULT_DIR}/run_config/staged_infos"
STAGED_DATASET="${RESULT_DIR}/predictions/baked_dataset_physical.pkl"
LEGACY_INFOS="${RESULT_DIR}/run_config/legacy_infos.json"
LEGACY_DATASET="${RESULT_DIR}/run_config/legacy_baked_dataset_physical.pkl"
KERNEL_PREFIX="${TMPDIR:-/tmp}/neuf-pose-mm-${PBS_JOBID}"

[[ "${RUN_ID}" =~ ^[0-9]{8}_train[0-9]{2,}$ ]]
[[ "${RESUME}" == 0 || "${RESUME}" == 1 ]]
[[ -x "${PYTHON_BIN}" && -f "${INFOS_PATH}" && -f "${DATASET_PATH}" ]]
if [[ "${RESUME}" == 0 ]]; then
    [[ ! -e "${RESULT_DIR}" ]]
else
    [[ -f "${STAGED_DATASET}" && -f "${LEGACY_INFOS}" && -f "${STAGED_INFOS_DIR}/infos.json" ]]
fi
mkdir -p "${QSUB_LOG_DIR}" "${STAGED_INFOS_DIR}" "${RESULT_DIR}/predictions" \
    "${RESULT_DIR}/metrics" "${RESULT_DIR}/plots" "${RESULT_DIR}/run_config/source" "${KERNEL_PREFIX}"
printf '%s\n' "${PBS_JOBID:-unknown}" > "${QSUB_LOG_DIR}/job_id.txt"

finish() {
    local status=$?
    if [[ "${status}" -ne 0 && -f "${LEGACY_DATASET}" ]]; then
        if [[ -f "${DATASET_PATH}" ]]; then
            mv "${DATASET_PATH}" "${STAGED_DATASET}"
        fi
        mv "${LEGACY_DATASET}" "${DATASET_PATH}"
        cp "${LEGACY_INFOS}" "${INFOS_PATH}"
        printf '已恢复原始 infos.json 和 baked dataset\n' >&2
    fi
    printf 'exit_code=%s\n' "${status}" > "${QSUB_LOG_DIR}/status.txt"
    printf '结果目录: %s\n实验日志: %s/metrics\nqsub日志: %s\n' \
        "${RESULT_DIR}" "${RESULT_DIR}" "${QSUB_LOG_DIR}"
    printf 'qsub脚本: %s/qsub/neuf/rebake_cerebral_pose.sh\nJob ID/状态: %s / exit_code=%s\n' \
        "${REPO_DIR}" "${PBS_JOBID:-unknown}" "${status}"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

cd "${REPO_DIR}"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLBACKEND=Agg
export POSE_MM_RESULT_DIR="${RESULT_DIR}"
export POSE_MM_LEGACY_INFOS="${LEGACY_INFOS}"
export POSE_MM_STAGED_INFOS="${STAGED_INFOS_DIR}/infos.json"
export POSE_MM_STAGED_DATASET="${STAGED_DATASET}"
export POSE_MM_LEGACY_DATASET="${DATASET_PATH}"
export MPLCONFIGDIR="${KERNEL_PREFIX}/matplotlib"
export IPYTHONDIR="${KERNEL_PREFIX}/ipython"
export JUPYTER_RUNTIME_DIR="${KERNEL_PREFIX}/runtime"

printf 'Host: %s\nLegacy infos: %s\nLegacy baked: %s\nStaged baked: %s\n' \
    "$(hostname)" "${INFOS_PATH}" "${DATASET_PATH}" "${STAGED_DATASET}"
cp "${REPO_DIR}/qsub/neuf/rebake_cerebral_pose.sh" "${RESULT_DIR}/run_config/source/"

if [[ "${RESUME}" == 0 ]]; then
cp "${INFOS_PATH}" "${LEGACY_INFOS}"
"${PYTHON_BIN}" - "${LEGACY_INFOS}" "${STAGED_INFOS_DIR}/infos.json" <<'PY'
import json
import math
from pathlib import Path
import sys

source_path, target_path = map(Path, sys.argv[1:])
data = json.loads(source_path.read_text())
infos = data["infos"]
assert infos.get("position_unit") != "mm", "拒绝重复缩放毫米位姿"
keys = sorted((key for key in data if key.isdigit()), key=int)
assert keys == [str(i) for i in range(242)]
assert abs(data["0"]["y"] - 455.5425008446047) < 1e-7
pixel_width_mm = float(infos["px_size_cm"]["width"]) * 10.0
pixel_height_mm = float(infos["px_size_cm"]["height"]) * 10.0
assert math.isclose(pixel_width_mm, pixel_height_mm, abs_tol=1e-12)
for key in keys:
    for axis in ("x", "y", "z"):
        data[key][axis] = float(data[key][axis]) * pixel_height_mm
infos["scan_dims_mm"] = {
    "width": float(infos["scan_dims_px"]["width"]) * pixel_width_mm,
    "depth": float(infos["scan_dims_px"]["depth"]) * pixel_height_mm,
}
infos["position_unit"] = "mm"
target_path.write_text(json.dumps(data, indent=2) + "\n")
print(f"修正 242 帧位置：{pixel_height_mm:.6f} mm/pixel，四元数保持原值", flush=True)
PY
fi

# 保留旧 baked 的 704 行图像和 sector mask，仅用毫米位姿重建空间缓存。
printf 'Pose rebake command: preserve legacy image/mask geometry, rebuild cached points into %s\n' "${STAGED_DATASET}"
"${PYTHON_BIN}" - "${DATASET_PATH}" "${STAGED_INFOS_DIR}/infos.json" "${STAGED_DATASET}" <<'PY'
import json
from pathlib import Path
import sys

import numpy as np

from neuf.dataset import Dataset

legacy_path, infos_path, output_path = map(Path, sys.argv[1:])
data = json.loads(infos_path.read_text())
dataset = Dataset.open_from_save(legacy_path, map_location="cpu")
assert (dataset.px_height, dataset.px_width) == (704, 944)
records = {}
for slices in (dataset.slices, dataset.slices_valid):
    for item in slices:
        frame_id = int(item.frame_index)
        item.position = np.array([data[str(frame_id)][axis] for axis in ("x", "y", "z")], dtype=np.float32)
        records[frame_id] = item
assert sorted(records) == list(range(242))
dataset.infos_json_path = str(infos_path)
dataset._rebuild_cached_slice_geometry()
dataset._update_scan_metadata(
    [records[index].position for index in range(242)],
    [records[index].rotation for index in range(242)],
)
dataset.save(output_path)
print(f"保留旧图像与 mask，已重建 242 帧空间点：{output_path}", flush=True)
PY

"${PYTHON_BIN}" -m ipykernel install \
    --prefix "${KERNEL_PREFIX}" --name neuf-pose-mm \
    --display-name 'Python 3 (NeUF pose millimetre correction)'
export JUPYTER_PATH="${KERNEL_PREFIX}/share/jupyter"
printf 'Validation command: execute analysis_workbench.ipynb cells tagged pose-mm-correction\n'
/usr/bin/python3 -u - "${REPO_DIR}/analysis_workbench.ipynb" <<'PY'
from copy import deepcopy
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

notebook_path = Path(sys.argv[1])
notebook = nbformat.read(notebook_path, as_version=4)
selected = [deepcopy(cell) for cell in notebook.cells
            if "pose-mm-correction" in cell.get("metadata", {}).get("tags", [])]
if len(selected) != 2:
    raise RuntimeError(f"Expected 2 pose-mm-correction cells, found {len(selected)}")
execution = nbformat.v4.new_notebook(cells=selected, metadata=deepcopy(notebook.metadata))
execution.metadata.setdefault("kernelspec", {})["name"] = "neuf-pose-mm"
try:
    NotebookClient(execution, kernel_name="neuf-pose-mm", timeout=1800,
                   resources={"metadata": {"path": str(notebook_path.parent.resolve())}}).execute()
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

mv "${DATASET_PATH}" "${LEGACY_DATASET}"
mv "${STAGED_DATASET}" "${DATASET_PATH}"
cp "${STAGED_INFOS_DIR}/infos.json" "${INFOS_PATH}.pose-mm-${PBS_JOBID}.tmp"
mv "${INFOS_PATH}.pose-mm-${PBS_JOBID}.tmp" "${INFOS_PATH}"
printf '已切换默认位姿和 baked dataset；旧版保留在 %s\n' "${RESULT_DIR}/run_config"
