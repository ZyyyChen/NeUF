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
RUN_ID="${RUN_ID:-20261004_train06}"
RESULT_DIR="${REPO_DIR}/trajectory_output"
CORONAL_IMAGES="${REPO_DIR}/data/cerebral_data/Pre_traitement_echo_v2/Recalage/Patient0/us_recal_original"
EXPERIMENT_LOG_DIR="${REPO_DIR}/logs/${RUN_ID}/cerebral_patient0/index_all/probe_trajectory"
QSUB_LOG_DIR="${REPO_DIR}/qsub/logs/neuf/${RUN_ID}"
cd "${REPO_DIR}"
export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/neuf-probe-trajectory-${PBS_JOBID}/matplotlib"
mkdir -p "${EXPERIMENT_LOG_DIR}" "${MPLCONFIGDIR}"
exec > >(tee "${EXPERIMENT_LOG_DIR}/execution.log") 2>&1
trap 'status=$?; echo "Exit status: ${status}"; echo "结果目录: ${RESULT_DIR}"; echo "实验日志: ${EXPERIMENT_LOG_DIR}"; echo "qsub日志: ${QSUB_LOG_DIR}"; echo "qsub脚本: ${REPO_DIR}/qsub/neuf/plot_probe_trajectory.sh"; echo "Job ID: ${PBS_JOBID}"' EXIT
echo "Command: ${PYTHON_BIN} ${RESULT_DIR}/plot_probe_trajectory.py --combined-only --frames 242 --fps 20 --rectangle-y 50 --coronal-images ${CORONAL_IMAGES} --output-dir ${RESULT_DIR}"
"${PYTHON_BIN}" -u "${RESULT_DIR}/plot_probe_trajectory.py" \
  --combined-only --frames 242 --fps 20 --rectangle-y 50 \
  --coronal-images "${CORONAL_IMAGES}" --output-dir "${RESULT_DIR}"

# 确认静态图可读，动画所有帧完整。
"${PYTHON_BIN}" - "${RESULT_DIR}" <<'PY'
from pathlib import Path
import sys
import numpy as np
from PIL import Image

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
from plot_probe_trajectory import TrajectoryParameters, calculate_trajectories, image_rectangle, image_texture_grid

# 校验贴图四角和探头边：不依赖视觉判断是否上下或左右翻转。
parameters = TrajectoryParameters(rectangle_y=50)
trajectories = calculate_trajectories(parameters)
for index in (0, 121, 241):
    plane = image_rectangle(trajectories, index, parameters.image_width / (2 * parameters.plot_scale))
    grid = image_texture_grid(plane, 160, 120)
    np.testing.assert_allclose(grid[0, 0], plane[3])
    np.testing.assert_allclose(grid[0, -1], plane[2])
    np.testing.assert_allclose(grid[-1, 0], plane[0])
    np.testing.assert_allclose(grid[-1, -1], plane[1])
    np.testing.assert_allclose(grid[0, 80], trajectories.probe[index], atol=1e-12)
    np.testing.assert_allclose(grid[-1, 80], trajectories.image_end[index], atol=1e-12)
print("Verified texture corners, left/right order, top=probe, bottom=image end: frames 0, 121, 241")

for name in ("probe_trajectory_with_coronal.gif",):
    path = root / name
    with Image.open(path) as image:
        frames = getattr(image, "n_frames", 1)
        assert frames == (242 if path.suffix == ".gif" else 1), (name, frames)
        for index in range(frames):
            image.seek(index)
            image.load()
        print(f"Verified: {name}, size={image.size}, frames={frames}, bytes={path.stat().st_size}")
        for index in (0, frames // 2, frames - 1):
            image.seek(index)
            image.convert("RGB").save(root / f"{path.stem}_frame_{index:03d}.png")
PY
