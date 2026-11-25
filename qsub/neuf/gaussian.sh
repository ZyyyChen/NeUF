#!/bin/bash -l
#PBS -N neuf_gaussian
#PBS -q gpu
#PBS -l nodes=1:ppn=4:gpus=1:ampere
#PBS -l mem=24gb
#PBS -l walltime=03:00:00
set -euo pipefail
WORKSPACE_ROOT=/misc/raid/zchen/Code
RUN_ID=${RUN_ID:?}
PHASE=${PHASE:-prepare}
RESULT_DIR=$WORKSPACE_ROOT/logs/$RUN_ID/cerebral/index_patient0/UltraGRay
QSUB_LOG_DIR=$WORKSPACE_ROOT/qsub/logs/neuf/$RUN_ID
PYTHON_BIN=/home/zchen/.conda/envs/neuf/bin/python
cd "$WORKSPACE_ROOT/NeUF"
export PYTHONPATH="$WORKSPACE_ROOT/NeUF:$WORKSPACE_ROOT/NeUF/.baseline_deps:$WORKSPACE_ROOT/UltraG-Ray:$WORKSPACE_ROOT/UltraG-Ray/.runtime_deps"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MAX_JOBS=4
export TORCH_EXTENSIONS_DIR="$WORKSPACE_ROOT/UltraG-Ray/.build"
export CUDA_HOME="$WORKSPACE_ROOT/UltraG-Ray/.cuda"
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX"
export CC="$CUDA_HOME/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CUDA_HOME/bin/x86_64-conda-linux-gnu-c++"
export NVCC_CCBIN="$CXX"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:${CPATH:-}"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-gaussian-${PBS_JOBID}"
export PATH="$CUDA_HOME/bin:$WORKSPACE_ROOT/UltraG-Ray/.runtime_deps/bin:$PATH"
mkdir -p "$RESULT_DIR/run_config" "$QSUB_LOG_DIR"
exec > >(tee "$QSUB_LOG_DIR/${PHASE}_${PBS_JOBID%%.*}.live.log") 2>&1
finish() {
  status=$?
  printf 'exit_code=%s\n' "$status" > "$QSUB_LOG_DIR/$PHASE.status"
  printf 'Result: %s\nJournal: %s/experiment_reports/2026/1125.md\nQsub logs: %s\nScript: %s/qsub/neuf/gaussian.sh\nJob: %s, exit=%s\n' "$RESULT_DIR" "$WORKSPACE_ROOT" "$QSUB_LOG_DIR" "$WORKSPACE_ROOT" "$PBS_JOBID" "$status"
}
trap finish EXIT
printf 'Host=%s Phase=%s Run=%s\n' "$(hostname)" "$PHASE" "$RUN_ID"
# Torque 以 GPUFILE 分配设备，不能默认使用物理卡0。
if [[ -n "${PBS_GPUFILE:-}" && -r "$PBS_GPUFILE" ]]; then
  printf 'Allocated GPUs: '; cat "$PBS_GPUFILE"
  allocated_gpu="$(sed -E 's/.*-gpu([0-9]+)$/\1/' "$PBS_GPUFILE" | paste -sd, -)"
  nvidia-smi -i "$allocated_gpu" --query-gpu=index,uuid,pci.bus_id,compute_mode,memory.used --format=csv
  nvidia-smi -i "$allocated_gpu" --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i "$allocated_gpu" --query-gpu=uuid --format=csv,noheader | paste -sd, -)"
fi
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
if [[ "$PHASE" != prepare && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  qstat -f "$PBS_JOBID"
  printf '缺少调度器 GPU 绑定，停止以免使用其他作业设备。\n' >&2
  exit 2
fi
if [[ "$PHASE" == prepare ]]; then
  if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    /usr/bin/conda create -y -p "$CUDA_HOME" --override-channels -c nvidia/label/cuda-12.6.3 -c conda-forge cuda-nvcc cuda-cudart-dev cuda-cccl cuda-driver-dev 'gcc_linux-64=12' 'gxx_linux-64=12'
  fi
  nvcc --version
  if [[ -f "$TORCH_EXTENSIONS_DIR/gsplat_cuda/build.ninja" && ! -f "$TORCH_EXTENSIONS_DIR/gsplat_cuda/gsplat_cuda.so" ]]; then
    ninja -C "$TORCH_EXTENSIONS_DIR/gsplat_cuda" -j "$MAX_JOBS"
  fi
  "$PYTHON_BIN" - <<'PY'
import importlib.util, json, os, subprocess
from pathlib import Path
import torch
print('torch',torch.__version__,'CUDA',torch.version.cuda,'GPU',torch.cuda.get_device_name(),flush=True)
packages=['rich','ninja','jaxtyping','einops','typing_extensions']
missing=[p for p in packages if importlib.util.find_spec(p) is None]
print('missing',missing,flush=True)
if missing:
    subprocess.run([os.sys.executable,'-m','pip','install','--target','/misc/raid/zchen/Code/UltraG-Ray/.runtime_deps',*missing],check=True)
from gsplat.cuda._backend import _C
assert _C is not None
print('CUDA_EXTENSION_READY',flush=True)
PY
else
  resume_args=()
  if [[ -n "${RESUME_STEP:-}" ]]; then
    printf -v resume_name 'step_%06d.pt' "$RESUME_STEP"
    resume_args=(--resume "$RESULT_DIR/checkpoints/$resume_name")
  fi
  "$PYTHON_BIN" -m neuf.edge_field.gaussian_reconstruction --output "$RESULT_DIR" --mode "$PHASE" --steps "${STEPS:-6000}" "${resume_args[@]}"
fi
