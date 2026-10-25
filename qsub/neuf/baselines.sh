#!/bin/bash -l
#PBS -N neuf_baselines
#PBS -q route
#PBS -l walltime=02:00:00
#PBS -l nodes=1:ppn=4
#PBS -l mem=32gb

set -euo pipefail
REPO_DIR=/misc/raid/zchen/Code/NeUF
PYTHON_BIN=/home/zchen/.conda/envs/neuf/bin/python
RUN_ID="${RUN_ID:?需要传入 YYYYMMDD_trainNN}"
MODE="${MODE:-inspect}"
MODEL="${MODEL:-all}"
CASE="${CASE:-all}"
[[ "${RUN_ID}" =~ ^[0-9]{8}_train[0-9]{2,}$ ]]
RESULT_DIR="${REPO_DIR}/logs/${RUN_ID}"
QSUB_LOG_DIR="${REPO_DIR}/qsub/logs/neuf/${RUN_ID}/${MODE}_${CASE}_${MODEL}"
mkdir -p "${QSUB_LOG_DIR}"
printf '%s\n' "${PBS_JOBID}" > "${QSUB_LOG_DIR}/job_id.txt"
finish() {
    local status=$?
    printf 'exit_code=%s\n' "${status}" > "${QSUB_LOG_DIR}/status.txt"
    printf '结果目录: %s\n实验日志: %s/<DATASET>/index_<CASE>/<MODEL>/metrics\nqsub日志: %s\nqsub脚本: %s/qsub/neuf/baselines.sh\nJob ID/状态: %s / exit_code=%s\n' \
        "${RESULT_DIR}" "${RESULT_DIR}" "${QSUB_LOG_DIR}" "${REPO_DIR}" "${PBS_JOBID}" "${status}"
}
trap finish EXIT
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/.baseline_deps:${REPO_DIR}:${REPO_DIR}/third_party/NeSVoR"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLCONFIGDIR="${TMPDIR:-/tmp}/neuf-baselines-${PBS_JOBID}"
export TORCH_EXTENSIONS_DIR="${RESULT_DIR}/baselines/index_all/NeSVoR/run_config/extensions"
printf 'Mode=%s Model=%s Case=%s Run=%s Host=%s\n' "${MODE}" "${MODEL}" "${CASE}" "${RUN_ID}" "$(hostname)"
EXECUTION_LOG="${RESULT_DIR}/baselines/index_all/comparison/metrics/${MODE}_${CASE}_${MODEL}_execution.log"
mkdir -p "$(dirname "${EXECUTION_LOG}")"
exec > >(tee "${EXECUTION_LOG}") 2>&1
if [[ "${MODE}" == inspect ]]; then
    "${PYTHON_BIN}" -m neuf.baselines.data --inspect "${RESULT_DIR}/baselines/index_all/comparison/run_config/calibration.json"
elif [[ "${MODE}" == inspect_reconstruction ]]; then
    "${PYTHON_BIN}" -m neuf.baselines.data --inspect-reconstruction "${RESULT_DIR}/baselines/index_all/comparison/run_config/reconstruction_parameters.json"
else
    if [[ "${MODE}" == smoke ]]; then
        # 只补当前环境缺少的纯 Python 依赖，不改动现有 conda 环境。
        "${PYTHON_BIN}" -m pip install --disable-pip-version-check --no-deps --target "${REPO_DIR}/.baseline_deps" nibabel==5.3.2
    fi
    cmd=("${PYTHON_BIN}" -m neuf.baselines.workflow --mode "${MODE}" --run-id "${RUN_ID}" --model "${MODEL}" --case "${CASE}")
    if [[ "${PIXEL_GEOMETRY:-0}" == 1 ]]; then
        cmd+=(--pixel-geometry)
    fi
    if [[ -n "${RESUME:-}" ]]; then
        cmd+=(--resume "${RESUME}")
    fi
    printf 'Command:'; printf ' %q' "${cmd[@]}"; printf '\n'
    "${cmd[@]}"
fi
