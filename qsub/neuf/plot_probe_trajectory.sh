#!/bin/bash -l
#PBS -N neuf_probe_trajectory
#PBS -q gpu
#PBS -l walltime=00:30:00
#PBS -l nodes=1:ppn=4:gpus=1:gpu48
#PBS -l mem=32gb

set -euo pipefail

# 复用项目环境，将图片写入用户指定目录。
REPO_DIR="/misc/raid/zchen/Code/NeUF"
PYTHON_BIN="/home/zchen/.conda/envs/neuf/bin/python"
RUN_ID="20260908_train02"
RESULT_DIR="${REPO_DIR}/trajectory_output"
EXPERIMENT_LOG_DIR="${REPO_DIR}/logs/${RUN_ID}/cerebral_patient0/index_all/probe_trajectory"
QSUB_LOG_DIR="${REPO_DIR}/qsub/logs/neuf/${RUN_ID}"
cd "${REPO_DIR}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/neuf-probe-trajectory-${PBS_JOBID}/matplotlib"
mkdir -p "${EXPERIMENT_LOG_DIR}" "${MPLCONFIGDIR}"
exec > >(tee "${EXPERIMENT_LOG_DIR}/execution.log") 2>&1
trap 'status=$?; echo "Exit status: ${status}"; echo "结果目录: ${RESULT_DIR}"; echo "实验日志: ${EXPERIMENT_LOG_DIR}"; echo "qsub日志: ${QSUB_LOG_DIR}"; echo "qsub脚本: ${REPO_DIR}/qsub/neuf/plot_probe_trajectory.sh"; echo "Job ID: ${PBS_JOBID}"' EXIT
echo "Command: ${PYTHON_BIN} ${RESULT_DIR}/plot_probe_trajectory.py --frames 242 --fps 20 --output-dir ${RESULT_DIR}"
"${PYTHON_BIN}" -u "${RESULT_DIR}/plot_probe_trajectory.py" \
  --frames 242 --fps 20 --output-dir "${RESULT_DIR}"

# 确认静态图可读，动画所有帧完整。
"${PYTHON_BIN}" - "${RESULT_DIR}" <<'PY'
from pathlib import Path
import sys
from PIL import Image

root = Path(sys.argv[1])
for name in ("probe_trajectory_3d.png", "probe_trajectory_yz.png", "probe_pose_with_trajectory.gif"):
    path = root / name
    with Image.open(path) as image:
        frames = getattr(image, "n_frames", 1)
        assert frames == (242 if path.suffix == ".gif" else 1), (name, frames)
        for index in range(frames):
            image.seek(index)
            image.load()
        print(f"Verified: {name}, size={image.size}, frames={frames}, bytes={path.stat().st_size}")
PY
