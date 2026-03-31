#!/bin/bash -l
#PBS -N knn_baseline_dist
#PBS -q route
#PBS -l walltime=30:00:00
#PBS -l nodes=2:ppn=16
#PBS -l mem=256gb
#PBS -j oe
#PBS -o /home/zchen/history/knn_baseline_dist.pbs.log
#PBS -M ziyi.chen@creatis.insa-lyon.fr
#PBS -m ae

set -euo pipefail

REPO_DIR="/home/zchen/Code/NeUF"
PYTHON_BIN="/home/zchen/.conda/envs/neuf/bin/python"
SCRIPT_PATH="${REPO_DIR}/export_knn_baseline.py"
OUTPUT_DIR="${REPO_DIR}/exports/knn_baseline_cluster"
CKPT_PATH="${REPO_DIR}/latest/ckpt.pkl"
LOG_DIR="/home/zchen/history/knn_baseline_dist"

KNN_K="${KNN_K:-3}"
KNN_MAX_DIST="${KNN_MAX_DIST:-}"
KNN_CHUNK_SIZE="${KNN_CHUNK_SIZE:-200000}"
KNN_RESOLUTION_SCALE="${KNN_RESOLUTION_SCALE:-1.0}"
KNN_SPACING="${KNN_SPACING:-}"
KNN_BOUNDS_DATASET="${KNN_BOUNDS_DATASET:-}"
KNN_POINT_MIN="${KNN_POINT_MIN:-}"
KNN_POINT_MAX="${KNN_POINT_MAX:-}"
KNN_USE_BBOX_MASK="${KNN_USE_BBOX_MASK:-0}"
KNN_DISABLE_SEQUENCE_PLANE_MASK="${KNN_DISABLE_SEQUENCE_PLANE_MASK:-0}"
KEEP_DISTRIBUTED_ARTIFACTS="${KEEP_DISTRIBUTED_ARTIFACTS:-0}"

mkdir -p "${LOG_DIR}"
JOB_LOG="${LOG_DIR}/${PBS_JOBID:-manual}.log"
exec > >(tee -a "${JOB_LOG}") 2>&1

cd "${REPO_DIR}"

if [[ -n "${PBS_NODEFILE:-}" && -f "${PBS_NODEFILE}" ]]; then
  mapfile -t ALLOCATED_NODES < <(sort -u "${PBS_NODEFILE}")
  TOTAL_SLOTS="$(wc -l < "${PBS_NODEFILE}")"
else
  ALLOCATED_NODES=("$(hostname)")
  TOTAL_SLOTS="$(nproc)"
fi

WORLD_SIZE="${#ALLOCATED_NODES[@]}"
if (( WORLD_SIZE < 1 )); then
  echo "No allocated nodes were detected."
  exit 1
fi

THREADS_PER_RANK=$(( TOTAL_SLOTS / WORLD_SIZE ))
if (( THREADS_PER_RANK < 1 )); then
  THREADS_PER_RANK=1
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${THREADS_PER_RANK}"
export OPENBLAS_NUM_THREADS="${THREADS_PER_RANK}"
export MKL_NUM_THREADS="${THREADS_PER_RANK}"
export NUMEXPR_NUM_THREADS="${THREADS_PER_RANK}"
export VECLIB_MAXIMUM_THREADS="${THREADS_PER_RANK}"
export BLIS_NUM_THREADS="${THREADS_PER_RANK}"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export KMP_AFFINITY=granularity=fine,compact,1,0

DATE_TAG="$(date +%d-%m-%Y)"
CKPT_STEM="$(basename "${CKPT_PATH}")"
CKPT_STEM="${CKPT_STEM%.*}"
RUN_PARENT="${OUTPUT_DIR}/${DATE_TAG}"
mkdir -p "${RUN_PARENT}"

RUN_INDEX=0
while true; do
  FINAL_OUTPUT_DIR="${RUN_PARENT}/${CKPT_STEM}_${RUN_INDEX}"
  if [[ ! -e "${FINAL_OUTPUT_DIR}" ]]; then
    mkdir -p "${FINAL_OUTPUT_DIR}"
    break
  fi
  RUN_INDEX=$(( RUN_INDEX + 1 ))
done

STAGING_ROOT="${RUN_PARENT}/.${CKPT_STEM}_${RUN_INDEX}_distributed_tmp"
SHARD_ROOT="${STAGING_ROOT}/shards"
mkdir -p "${SHARD_ROOT}"

CONFIG_FILE="${STAGING_ROOT}/distributed_knn.env"
{
  printf 'REPO_DIR=%q\n' "${REPO_DIR}"
  printf 'PYTHON_BIN=%q\n' "${PYTHON_BIN}"
  printf 'SCRIPT_PATH=%q\n' "${SCRIPT_PATH}"
  printf 'CKPT_PATH=%q\n' "${CKPT_PATH}"
  printf 'OUTPUT_DIR=%q\n' "${OUTPUT_DIR}"
  printf 'SHARD_ROOT=%q\n' "${SHARD_ROOT}"
  printf 'THREADS_PER_RANK=%q\n' "${THREADS_PER_RANK}"
  printf 'KNN_K=%q\n' "${KNN_K}"
  printf 'KNN_MAX_DIST=%q\n' "${KNN_MAX_DIST}"
  printf 'KNN_CHUNK_SIZE=%q\n' "${KNN_CHUNK_SIZE}"
  printf 'KNN_RESOLUTION_SCALE=%q\n' "${KNN_RESOLUTION_SCALE}"
  printf 'KNN_SPACING=%q\n' "${KNN_SPACING}"
  printf 'KNN_BOUNDS_DATASET=%q\n' "${KNN_BOUNDS_DATASET}"
  printf 'KNN_POINT_MIN=%q\n' "${KNN_POINT_MIN}"
  printf 'KNN_POINT_MAX=%q\n' "${KNN_POINT_MAX}"
  printf 'KNN_USE_BBOX_MASK=%q\n' "${KNN_USE_BBOX_MASK}"
  printf 'KNN_DISABLE_SEQUENCE_PLANE_MASK=%q\n' "${KNN_DISABLE_SEQUENCE_PLANE_MASK}"
} > "${CONFIG_FILE}"

WORKER_SCRIPT="${STAGING_ROOT}/run_knn_rank.sh"
cat > "${WORKER_SCRIPT}" <<'EOF'
#!/bin/bash
set -euo pipefail

CONFIG_FILE="$1"
RANK="${2:-${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-0}}}"
WORLD="${3:-${OMPI_COMM_WORLD_SIZE:-${PMI_SIZE:-1}}}"
source "${CONFIG_FILE}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${THREADS_PER_RANK}"
export OPENBLAS_NUM_THREADS="${THREADS_PER_RANK}"
export MKL_NUM_THREADS="${THREADS_PER_RANK}"
export NUMEXPR_NUM_THREADS="${THREADS_PER_RANK}"
export VECLIB_MAXIMUM_THREADS="${THREADS_PER_RANK}"
export BLIS_NUM_THREADS="${THREADS_PER_RANK}"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export KMP_AFFINITY=granularity=fine,compact,1,0

cd "${REPO_DIR}"
RANK_DIR="${SHARD_ROOT}/rank_${RANK}"

cmd=(
  "${PYTHON_BIN}" "${SCRIPT_PATH}"
  --ckpt "${CKPT_PATH}"
  --output "${OUTPUT_DIR}"
  --exact-output-dir "${RANK_DIR}"
  --shard-rank "${RANK}"
  --shard-world-size "${WORLD}"
  --query-workers "${THREADS_PER_RANK}"
  --chunk-size "${KNN_CHUNK_SIZE}"
  --k "${KNN_K}"
  --resolution-scale "${KNN_RESOLUTION_SCALE}"
)

if [[ -n "${KNN_MAX_DIST}" ]]; then
  cmd+=(--max-dist "${KNN_MAX_DIST}")
fi
if [[ -n "${KNN_SPACING}" ]]; then
  read -r -a spacing_vals <<< "${KNN_SPACING}"
  cmd+=(--spacing "${spacing_vals[@]}")
fi
if [[ -n "${KNN_BOUNDS_DATASET}" ]]; then
  cmd+=(--bounds-dataset "${KNN_BOUNDS_DATASET}")
fi
if [[ -n "${KNN_POINT_MIN}" ]]; then
  read -r -a point_min_vals <<< "${KNN_POINT_MIN}"
  cmd+=(--point-min "${point_min_vals[@]}")
fi
if [[ -n "${KNN_POINT_MAX}" ]]; then
  read -r -a point_max_vals <<< "${KNN_POINT_MAX}"
  cmd+=(--point-max "${point_max_vals[@]}")
fi
if [[ "${KNN_USE_BBOX_MASK}" == "1" ]]; then
  cmd+=(--use-bbox-mask)
fi
if [[ "${KNN_DISABLE_SEQUENCE_PLANE_MASK}" == "1" ]]; then
  cmd+=(--disable-sequence-plane-mask)
fi

echo "Rank ${RANK}/${WORLD}: output -> ${RANK_DIR}"
printf 'Rank %s command:' "${RANK}"
printf ' %q' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
EOF
chmod +x "${WORKER_SCRIPT}"

echo "Host: $(hostname)"
echo "Date: $(date --iso-8601=seconds)"
echo "PBS job id: ${PBS_JOBID:-n/a}"
echo "PBS queue: ${PBS_QUEUE:-n/a}"
echo "Working dir: ${REPO_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "Checkpoint: ${CKPT_PATH}"
echo "Final output dir: ${FINAL_OUTPUT_DIR}"
echo "Allocated nodes (${WORLD_SIZE} ranks): ${ALLOCATED_NODES[*]}"
echo "Total slots: ${TOTAL_SLOTS}"
echo "Threads per rank: ${THREADS_PER_RANK}"
echo "KNN k: ${KNN_K}"
echo "KNN max dist: ${KNN_MAX_DIST:-none}"
echo "Chunk size: ${KNN_CHUNK_SIZE}"
echo "Resolution scale: ${KNN_RESOLUTION_SCALE}"
echo "Spacing override: ${KNN_SPACING:-default}"
echo "Bounds dataset: ${KNN_BOUNDS_DATASET:-checkpoint dataset}"
echo "Point min override: ${KNN_POINT_MIN:-none}"
echo "Point max override: ${KNN_POINT_MAX:-none}"
echo "Use bbox mask: ${KNN_USE_BBOX_MASK}"
echo "Disable sequence plane mask: ${KNN_DISABLE_SEQUENCE_PLANE_MASK}"
echo "Keep distributed staging artifacts: ${KEEP_DISTRIBUTED_ARTIFACTS}"

PIDS=()
RANK_IDS=()
RANK_HOSTS=()

if command -v pbsdsh >/dev/null 2>&1 && [[ -n "${PBS_NODEFILE:-}" ]]; then
  echo "Launch method: pbsdsh"
  for rank in "${!ALLOCATED_NODES[@]}"; do
    host="${ALLOCATED_NODES[$rank]}"
    echo "Launching rank ${rank}/${WORLD_SIZE} on ${host}"
    pbsdsh -h "${host}" -o "${WORKER_SCRIPT}" "${CONFIG_FILE}" "${rank}" "${WORLD_SIZE}" &
    PIDS+=("$!")
    RANK_IDS+=("${rank}")
    RANK_HOSTS+=("${host}")
  done
elif (( WORLD_SIZE == 1 )); then
  echo "Launch method: local single-rank fallback"
  "${WORKER_SCRIPT}" "${CONFIG_FILE}" 0 1 &
  PIDS+=("$!")
  RANK_IDS+=("0")
  RANK_HOSTS+=("$(hostname)")
elif command -v ssh >/dev/null 2>&1; then
  echo "Launch method: ssh fallback"
  for rank in "${!ALLOCATED_NODES[@]}"; do
    host="${ALLOCATED_NODES[$rank]}"
    echo "Launching rank ${rank}/${WORLD_SIZE} on ${host} via ssh"
    ssh "${host}" "${WORKER_SCRIPT}" "${CONFIG_FILE}" "${rank}" "${WORLD_SIZE}" &
    PIDS+=("$!")
    RANK_IDS+=("${rank}")
    RANK_HOSTS+=("${host}")
  done
else
  echo "No supported distributed launcher found. Need pbsdsh inside PBS, or ssh, or a single-node run."
  exit 1
fi

LAUNCH_FAILED=0
for idx in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$idx]}"; then
    echo "Rank ${RANK_IDS[$idx]} on ${RANK_HOSTS[$idx]} failed."
    LAUNCH_FAILED=1
  fi
done
if (( LAUNCH_FAILED != 0 )); then
  echo "At least one distributed rank failed; skipping merge."
  exit 1
fi

export FINAL_OUTPUT_DIR SHARD_ROOT WORLD_SIZE STAGING_ROOT KEEP_DISTRIBUTED_ARTIFACTS
"${PYTHON_BIN}" - <<'PY'
import json
import os
import shutil
from pathlib import Path

final_output_dir = Path(os.environ["FINAL_OUTPUT_DIR"])
shard_root = Path(os.environ["SHARD_ROOT"])
world_size = int(os.environ["WORLD_SIZE"])
staging_root = Path(os.environ["STAGING_ROOT"])
keep_artifacts = os.environ.get("KEEP_DISTRIBUTED_ARTIFACTS", "0") == "1"

shard_dirs = sorted(shard_root.glob("rank_*"), key=lambda p: int(p.name.split("_")[-1]))
if len(shard_dirs) != world_size:
    raise RuntimeError(
        f"Expected {world_size} shard directories under {shard_root}, found {len(shard_dirs)}."
    )

shard_records = []
for shard_dir in shard_dirs:
    metadata_path = shard_dir / "metadata.json"
    raw_path = shard_dir / "volume.raw"
    if not metadata_path.exists() or not raw_path.exists():
        raise FileNotFoundError(f"Shard output is incomplete: {shard_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    shard_records.append((metadata["shard_z_start"], metadata["shard_z_stop"], shard_dir, metadata))

shard_records.sort(key=lambda item: item[0])
global_shape = shard_records[0][3]["global_grid_shape_zyx"]
spacing_xyz = shard_records[0][3]["spacing_mm_xyz"]
expected_start = 0
filled_voxels = 0
active_voxels = 0
for shard_start, shard_stop, shard_dir, metadata in shard_records:
    if shard_start != expected_start:
        raise RuntimeError(
            f"Shard z ranges are not contiguous: expected start {expected_start}, got {shard_start} ({shard_dir})."
        )
    expected_start = shard_stop
    if metadata["global_grid_shape_zyx"] != global_shape:
        raise RuntimeError(f"Mismatched global grid shape in {shard_dir}")
    if metadata["spacing_mm_xyz"] != spacing_xyz:
        raise RuntimeError(f"Mismatched spacing in {shard_dir}")
    filled_voxels += int(metadata["filled_voxels"])
    active_voxels += int(metadata["active_voxels_after_masking"])

if expected_start != int(global_shape[0]):
    raise RuntimeError(
        f"Shard z coverage is incomplete: expected stop {global_shape[0]}, got {expected_start}."
    )

merged_raw_path = final_output_dir / "volume.raw"
with merged_raw_path.open("wb") as fout:
    for _, _, shard_dir, _ in shard_records:
        with (shard_dir / "volume.raw").open("rb") as fin:
            shutil.copyfileobj(fin, fout, length=1024 * 1024 * 16)

dim_z, dim_y, dim_x = [int(v) for v in global_shape]
sx, sy, sz = [float(v) for v in spacing_xyz]
header = "\n".join([
    "ObjectType = Image",
    "NDims = 3",
    "BinaryData = True",
    "BinaryDataByteOrderMSB = False",
    "CompressedData = False",
    "TransformMatrix = 1 0 0 0 1 0 0 0 1",
    "Offset = 0 0 0",
    "CenterOfRotation = 0 0 0",
    "AnatomicalOrientation = RAI",
    f"ElementSpacing = {sy} {sz} {sx}",
    f"DimSize = {dim_x} {dim_y} {dim_z}",
    "ElementType = MET_FLOAT",
    f"ElementDataFile = {merged_raw_path.name}",
    "",
])
(final_output_dir / "volume.mhd").write_text(header, encoding="ascii")

final_metadata = dict(shard_records[0][3])
final_metadata.update({
    "grid_shape_zyx": list(global_shape),
    "total_voxels": int(dim_z * dim_y * dim_x),
    "filled_voxels": int(filled_voxels),
    "active_voxels_after_masking": int(active_voxels),
    "shard_rank": None,
    "shard_z_start": 0,
    "shard_z_stop": int(dim_z),
    "distributed_world_size": int(world_size),
    "distributed_merge": "concatenated rank-local z shards",
    "distributed_staging_retained": bool(keep_artifacts),
})
if keep_artifacts:
    final_metadata["shard_outputs_root"] = str(shard_root)
    final_metadata["distributed_staging_root"] = str(staging_root)
(final_output_dir / "metadata.json").write_text(
    json.dumps(final_metadata, indent=2),
    encoding="utf-8",
)

print(f"Merged {world_size} shards into: {final_output_dir}")
print(f"Final volume shape (z, y, x): {tuple(global_shape)}")
PY

if [[ "${KEEP_DISTRIBUTED_ARTIFACTS}" != "1" ]]; then
  rm -rf "${STAGING_ROOT}"
fi

echo "Finished at: $(date --iso-8601=seconds)"
echo "Merged export: ${FINAL_OUTPUT_DIR}"
