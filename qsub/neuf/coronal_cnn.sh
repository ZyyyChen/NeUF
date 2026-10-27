#!/bin/bash -l
#PBS -N neuf_coronal_cnn
#PBS -q gpu
#PBS -l walltime=02:00:00
#PBS -l nodes=1:ppn=4:gpus=1:gpu32
#PBS -l mem=32gb

set -euo pipefail
WORKSPACE_ROOT=/misc/raid/zchen/Code
REPO_DIR="${WORKSPACE_ROOT}/NeUF"
PYTHON_BIN=/home/zchen/.conda/envs/neuf/bin/python
RUN_ID="${RUN_ID:?需要 YYYYMMDD_trainNN}"
MODE="${MODE:-smoke}"
[[ "${RUN_ID}" =~ ^[0-9]{8}_train[0-9]{2,}$ ]]
[[ "${MODE}" == smoke || "${MODE}" == train ]]
QSUB_LOG_DIR="${WORKSPACE_ROOT}/qsub/logs/neuf/${RUN_ID}"
DATASET=cerebral
[[ "${MODE}" == train ]] || DATASET=synthetic_cerebral
RESULT_DIR="${WORKSPACE_ROOT}/logs/${RUN_ID}/${DATASET}/index_patient0"
[[ ! -e "${RESULT_DIR}" ]]
mkdir -p "${QSUB_LOG_DIR}"
printf '%s\n' "${PBS_JOBID}" > "${QSUB_LOG_DIR}/job_id.txt"
finish() {
    local status=$?
    printf 'exit_code=%s\n' "${status}" > "${QSUB_LOG_DIR}/status.txt"
    if [[ -d "${RESULT_DIR}" ]]; then
        mkdir -p "${RESULT_DIR}/comparison/run_config"
        cp "${QSUB_LOG_DIR}/execution.log" "${RESULT_DIR}/comparison/execution.log"
        cp "${BASH_SOURCE[0]}" "${RESULT_DIR}/comparison/run_config/coronal_cnn.sh"
    fi
    printf '结果目录: %s\n实验日志: %s/comparison/execution.log\nqsub日志: %s\nqsub脚本: %s/qsub/neuf/coronal_cnn.sh\nJob ID/状态: %s / exit_code=%s\n' \
        "${RESULT_DIR}" "${RESULT_DIR}" "${QSUB_LOG_DIR}" "${REPO_DIR}" "${PBS_JOBID}" "${status}"
}
trap finish EXIT
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLCONFIGDIR="${TMPDIR:-/tmp}/neuf-coronal-${PBS_JOBID}"
if [[ "${FORCE_CPU:-0}" == 1 ]]; then
    [[ "${MODE}" == smoke ]]
    export CUDA_VISIBLE_DEVICES=""
fi
if [[ "${MODE}" == smoke ]]; then
    cmd=("${PYTHON_BIN}" -m neuf.phase1_smoke --coronal-cnn-smoke --output-dir "${RESULT_DIR}/CoronalSmoke")
else
    cmd=("${PYTHON_BIN}" -m neuf.edge_field.workflow --coronal-cnn-comparison
         --output "${RESULT_DIR}" --steps 2000 --patches 4 --patch-size 64 --sagittal-patches 2
         --width 128 --bands 10 --seed 3407 --lr 0.0005 --pose-lr 0.002)
fi
printf 'Mode: %s\nCommand:' "${MODE}"; printf ' %q' "${cmd[@]}"; printf '\n'
git rev-parse HEAD
git status --short
# 先保持正式入口的拒绝覆盖检查，再把完整执行日志归档到实际结果目录。
"${cmd[@]}" 2>&1 | tee "${QSUB_LOG_DIR}/execution.log"
