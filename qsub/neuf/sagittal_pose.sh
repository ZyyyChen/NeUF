#!/bin/bash -l
#PBS -N neuf_sag_pose
#PBS -q gpu
#PBS -l walltime=02:00:00
#PBS -l nodes=1:ppn=16:gpus=1:gpu48
#PBS -l mem=128gb

set -euo pipefail
REPO_DIR=/misc/raid/zchen/Code/NeUF
PYTHON_BIN=/home/zchen/.conda/envs/neuf/bin/python
RUN_ID="${RUN_ID:?需要传入 YYYYMMDD_trainNN}"
MODE="${MODE:-smoke}"
QSUB_LOG_DIR="${REPO_DIR}/qsub/logs/neuf/${RUN_ID}"
[[ "${RUN_ID}" =~ ^[0-9]{8}_train[0-9]{2,}$ ]]
[[ "${MODE}" == smoke || "${MODE}" == train ]]
DATASET_NAME=synthetic_cerebral
[[ "${MODE}" == smoke ]] || DATASET_NAME=cerebral
RESULT_DIR="${REPO_DIR}/logs/${RUN_ID}/${DATASET_NAME}/index_all"
[[ ! -e "${RESULT_DIR}" ]]
mkdir -p "${RESULT_DIR}" "${QSUB_LOG_DIR}"
printf '%s\n' "${PBS_JOBID}" > "${QSUB_LOG_DIR}/job_id.txt"
finish() {
    local status=$?
    printf 'exit_code=%s\n' "${status}" > "${QSUB_LOG_DIR}/status.txt"
    printf '结果目录: %s\n实验日志: %s/<MODEL>/metrics\nqsub日志: %s\n' "${RESULT_DIR}" "${RESULT_DIR}" "${QSUB_LOG_DIR}"
    printf 'qsub脚本: %s/qsub/neuf/sagittal_pose.sh\nJob ID/状态: %s / exit_code=%s\n' "${REPO_DIR}" "${PBS_JOBID}" "${status}"
}
trap finish EXIT
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLCONFIGDIR="${TMPDIR:-/tmp}/neuf-sagittal-${PBS_JOBID}"
export SAGITTAL_RESULT_DIR="${RESULT_DIR}"
printf 'Mode: %s\nRun: %s\n' "${MODE}" "${RUN_ID}"

if [[ "${MODE}" == smoke ]]; then
    # 同一个任务运行实际训练入口的新旧各 40 步，患者数据完全不参与。
    BASELINE_SOURCE="${RESULT_DIR}/comparison/run_config/source_before"
    BASELINE_COMMIT="${BASELINE_COMMIT:-7f2341f6f3a674a13276c23943420103bd2f139b}"
    mkdir -p "${BASELINE_SOURCE}/neuf/edge_field"
    cp neuf/*.py "${BASELINE_SOURCE}/neuf/"
    while IFS= read -r source_file; do
        git show "${BASELINE_COMMIT}:${source_file}" > "${BASELINE_SOURCE}/${source_file}"
    done < <(git ls-tree -r --name-only "${BASELINE_COMMIT}" neuf/edge_field)
    printf '%s\n' "${BASELINE_COMMIT}" > "${BASELINE_SOURCE}/baseline_commit.txt"
    cmd=("${PYTHON_BIN}" -m neuf.phase1_smoke --sagittal-pose-smoke --output-dir "${RESULT_DIR}/SagittalPose6D")
    printf 'Command: '; printf ' %q' "${cmd[@]}"; printf '\n'
    "${cmd[@]}"
    cd "${BASELINE_SOURCE}"
    export PYTHONPATH="${BASELINE_SOURCE}:${REPO_DIR}"
    cmd=("${PYTHON_BIN}" -m neuf.phase1_smoke --sagittal-pose-smoke --sagittal-pose-baseline --output-dir "${RESULT_DIR}/PreviousSagittalINR")
    printf 'Command: '; printf ' %q' "${cmd[@]}"; printf '\n'
    "${cmd[@]}"
    cd "${REPO_DIR}"
    export PYTHONPATH="${REPO_DIR}"
    "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path
from neuf.edge_field.evaluation import compare_guided_sagittal

root = Path(os.environ["SAGITTAL_RESULT_DIR"])
names = ("PreviousSagittalINR", "SagittalPose6D")
configs = [json.loads((root/name/"run_config/config.json").read_text()) for name in names]
assert configs[0]["initial_field_sha256"] == configs[1]["initial_field_sha256"]
results = [json.loads((root/name/"metrics/sagittal_pose_smoke.json").read_text()) for name in names]
assert results[0]["data"] == results[1]["data"]
assert [item["pose_updates"] for item in results] == [5, 5]
compare_guided_sagittal({name: root/name/"predictions/guided_sagittal_step_000040.npz" for name in names}, root/"comparison")
print("PASS: identical initialization/data/field budget; 5 geometry updates each; synthetic quality not a medical result")
PY
else
    # 正式训练的外部数据与基线必须由调用方明确传入。
    cmd=("${PYTHON_BIN}" -m neuf.edge_field.workflow
         --dataset "${DATASET_PATH:?}" --teacher "${TEACHER_PATH:?}" --source-images "${SOURCE_IMAGES:?}"
         --sagittal-reference "${SAGITTAL_REFERENCE:?}" --sagittal-calibration "${SAGITTAL_CALIBRATION:?}"
         --landmarks "${LANDMARKS:?}" --sagittal-baseline-checkpoint "${BASELINE_CHECKPOINT:?}"
         --output "${RESULT_DIR}" --encoding hash --variants HashSagittalAllPoints
         --steps "${STEPS:-40000}" --pose-every "${POSE_EVERY:-5}"
         --pose-start "${POSE_START:-0.2}" --pose-end "${POSE_END:-0.8}")
    # 主入口拒绝覆盖目录；将它尚未使用的空占位目录移除。
    rmdir "${RESULT_DIR}"
    printf 'Command: '; printf ' %q' "${cmd[@]}"; printf '\n'
    "${cmd[@]}"
fi
