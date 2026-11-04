#!/bin/bash -l
#PBS -N neuf_trajectory
#PBS -q gpu
#PBS -l walltime=02:00:00
#PBS -l nodes=1:ppn=4:gpus=1:gpu32
#PBS -l mem=32gb

set -euo pipefail
WORKSPACE_ROOT=/misc/raid/zchen/Code
REPO_DIR="$WORKSPACE_ROOT/NeUF"
PYTHON_BIN=/home/zchen/.conda/envs/neuf/bin/python
: "${RUN_ID:?需要 YYYYMMDD_trainNN}"
MODE=${MODE:-smoke}
METHOD=${METHOD:-all}
ATTEMPT=${ATTEMPT:-1}
[[ "$RUN_ID" =~ ^[0-9]{8}_train[0-9]{2,}$ ]]
[[ "$MODE" == smoke || "$MODE" == prepare || "$MODE" == train || "$MODE" == compare ]]
[[ "$METHOD" =~ ^(all|FourP|SevenP|Edge|CNN|EdgeDT|Annotation|AnnotationDT)$ ]]
[[ "$ATTEMPT" =~ ^[0-9]+$ ]]
RESULT_DIR="$WORKSPACE_ROOT/logs/$RUN_ID/cerebral/index_patient0"
QSUB_LOG_DIR="$WORKSPACE_ROOT/qsub/logs/neuf/$RUN_ID"
TASK="${MODE}_${METHOD}_attempt${ATTEMPT}"
mkdir -p "$QSUB_LOG_DIR" "$RESULT_DIR/comparison/metrics" "$RESULT_DIR/comparison/run_config/$TASK"
printf '%s\n' "${PBS_JOBID:?需要通过qsub执行}" > "$QSUB_LOG_DIR/${TASK}_job_id.txt"
exec > >(tee "$RESULT_DIR/comparison/metrics/${TASK}_execution.log") 2>&1
finish() {
    local status=$?
    printf 'exit_code=%s\n' "$status" > "$QSUB_LOG_DIR/${TASK}_status.txt"
    printf '结果目录: %s\n实验日志: %s/comparison/metrics/%s_execution.log\nqsub日志: %s\nqsub脚本: %s/qsub/neuf/trajectory_reconstruction.sh\nJob ID/状态: %s / exit_code=%s\n' \
        "$RESULT_DIR" "$RESULT_DIR" "$TASK" "$QSUB_LOG_DIR" "$WORKSPACE_ROOT" "$PBS_JOBID" "$status"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLCONFIGDIR="${TMPDIR:-/tmp}/neuf-trajectory-${PBS_JOBID}"
# 每个作业冻结实际源码，避免多组同时写入同一个来源目录。
cp neuf/edge_field/*.py "$RESULT_DIR/comparison/run_config/$TASK/"
cp neuf/phase1_smoke.py "${BASH_SOURCE[0]}" "$RESULT_DIR/comparison/run_config/$TASK/"
git rev-parse HEAD > "$RESULT_DIR/comparison/run_config/$TASK/git_commit.txt"
git diff -- neuf/edge_field neuf/phase1_smoke.py qsub/neuf/trajectory_reconstruction.sh > "$RESULT_DIR/comparison/run_config/$TASK/source.diff"
run() { printf '命令:'; printf ' %q' "$@"; printf '\n'; "$@"; }
printf '模式=%s 方法=%s 预算=%s 节点=%s Job=%s\n' "$MODE" "$METHOD" "${STEPS:-20000}" "$(hostname)" "$PBS_JOBID"
if [[ "$MODE" == smoke ]]; then
    run "$PYTHON_BIN" -m neuf.phase1_smoke --trajectory-pose-smoke \
        --output-dir "$WORKSPACE_ROOT/logs/$RUN_ID/synthetic_cerebral/index_patient0/PoseSmoke"
    run "$PYTHON_BIN" -m neuf.edge_field.trajectory_reconstruction --mode prepare --output "$RESULT_DIR" --device cpu
    run "$PYTHON_BIN" -m neuf.edge_field.trajectory_reconstruction --mode smoke --output "$RESULT_DIR" \
        --method AnnotationDT --pose-mode all --steps 20
    run "$PYTHON_BIN" -m neuf.edge_field.trajectory_reconstruction --mode compare --output "$RESULT_DIR" \
        --method AnnotationDT --pose-mode all --steps 20
else
    if [[ "$MODE" == train && "$METHOD" == FourP ]]; then
        run "$PYTHON_BIN" -m neuf.edge_field.trajectory_reconstruction --mode prepare --output "$RESULT_DIR" --device cpu
    fi
    run "$PYTHON_BIN" -m neuf.edge_field.trajectory_reconstruction --mode "$MODE" --output "$RESULT_DIR" \
        --method "$METHOD" --pose-mode all --steps "${STEPS:-20000}"
fi

if [[ "$MODE" == compare && "$METHOD" == all ]]; then
    # 汇总仅引用本轮实际产物，向共享工作台追加图表，不复制训练实现。
    export RUN_ID WORKSPACE_ROOT RESULT_DIR
    run "$PYTHON_BIN" - <<'PY'
import json
import os
import re
from pathlib import Path

root = Path(os.environ['WORKSPACE_ROOT'])
result = Path(os.environ['RESULT_DIR'])
run_id = os.environ['RUN_ID']
comparison = json.loads((result/'comparison/metrics/comparison.json').read_text())
rows = [row for row in comparison['global_rows'] if row['region'] == 'Full']
assert len(rows) == 28
paired = {(row['method'], row['mode']): row for row in comparison['within_method_rows']
          if row['region'] == 'Full'}
names = dict(FourP='4P', SevenP='7P', Edge='Edge', CNN='CNN', EdgeDT='Edge+DT',
             Annotation='标注', AnnotationDT='标注+DT')
modes = dict(off='不修正', translation='仅平移', se3='平移＋旋转', sweep='仅扫角')
number = lambda value: '不可用' if value is None else f'{value:.6f}'
table = ['| 方法 | 追加修正 | 全28组共同域 SSIM ↑ | 方法内共同域 SSIM ↑ | SAG覆盖率 | 场/位置更新 | 状态 |',
         '|---|---|---:|---:|---:|---|---|']
details = []
for row in rows:
    name = f"{row['method']}_{row['mode']}"
    folder = result/name
    summary = json.loads((folder/'metrics/training_summary.json').read_text())
    audits = json.loads((folder/'metrics/checkpoint_reload.json').read_text())
    assert summary['status'] == 'complete' and summary['completed_total_steps'] == 20000
    assert [entry['step'] for entry in audits] == [10000, 15000, 20000]
    assert all(entry['independently_loaded'] and entry['reference_grid_independently_loaded'] for entry in audits)
    paths = [str(Path(entry['path'])) for entry in audits]
    assert all(Path(path).is_file() for path in paths)
    table.append(f"| {names[row['method']]} | {modes[row['mode']]} | {number(row['ssim'])} | "
                 f"{number(paired[row['method'], row['mode']]['ssim'])} | {row['reference_mask_coverage']:.2%} | "
                 f"20000 / {summary['pose_updates']} | 完成 |")
    details.append(f"- {name}：50% `{paths[0]}`；75% `{paths[1]}`；结束 `{paths[2]}`。"
                   f"独立重载通过；训练阶段（含阶段评价）{summary['total_wall_time_sec']:.1f}秒。")
table_text = '\n'.join(table)
relative = result.relative_to(root).as_posix()
description = ('以同一Fourier PE+MLP三维隐式灰度场重建七种已保存的完整轨迹，比较关闭、仅平移、6DoF和仅扫角四种追加修正。')
conditions = ('全部242帧参与病例内拟合；每组20000次场更新、seed3407、纯单尺度1−SSIM。'
              '自动与标注方法保留各自原标定，SAG图像708×944、灰度[0,1]及R1–R5固定。'
              '全28组共同域与每方法四模式共同域不同，两列分数不能相减；跨方法比较包含原始标定差异。'
              f"全组共同域{comparison['global_common_pixels']}像素、{rows[0]['complete_windows']}个完整11×11窗口。")
principle = ('项目目标与测试原则：观察固定原生图像的结构、散斑和伪影；坐标、梯度和checkpoint检查只证明实现契约，'
             '不证明重建质量提高，也不提供独立三维定位精度。')
notebook_text = (f'## NeUF：七种轨迹 × 四种位置修正（{run_id}）\n\n{description}\n\n{conditions}\n\n'
                 f'{table_text}\n\n![统一指标]({relative}/comparison/plots/metrics_summary.png)\n\n'
                 f'![原生SAG对比]({relative}/comparison/plots/overview.png)\n\n'
                 + ' · '.join(f'[R{i}]({relative}/comparison/plots/roi_R{i}.png)' for i in range(1, 6))
                 + f'\n\n{principle}\n\n复现入口：`NeUF/neuf/edge_field/trajectory_reconstruction.py`；'
                   f'输入与标定：`{relative}/comparison/run_config/input_audit.json`。\n')
notebook_path = root/'analysis_workbench.ipynb'
before = notebook_path.read_bytes()
notebook = json.loads(before)
cell_id = f'neuf-trajectory-{run_id}'
cell = dict(cell_type='markdown', id=cell_id, metadata={'trajectory_comparison': run_id},
            source=notebook_text.splitlines(keepends=True))
existing = [i for i, item in enumerate(notebook['cells']) if item.get('id') == cell_id]
if existing:
    notebook['cells'][existing[0]] = cell
else:
    notebook['cells'].append(cell)
    # 在已有首个说明单元补充任务索引，保留所有旧单元及输出。
    first = next(item for item in notebook['cells'] if item['cell_type'] == 'markdown')
    first['source'] += [f'\n- NeUF七轨迹×四修正：{run_id}，见本工作台末尾。\n']
temporary = notebook_path.with_name(f'.analysis_workbench.{run_id}.tmp')
temporary.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n')
assert notebook_path.read_bytes() == before, '工作台并发修改，保留临时文件供合并'
temporary.replace(notebook_path)

report_path = root/'experiment_reports'/run_id[:4]/f'{run_id[4:8]}.md'
heading = '## NeUF：七种现有轨迹 × 四种位置修正'
link = f'../../{relative}/comparison'
section = (f'{heading}\n\n{description}\n\n{table_text}\n\n{conditions}\n\n'
           f'![共同域SSIM及覆盖率]({link}/plots/metrics_summary.png)\n\n'
           f'![统一灰度原生SAG对比]({link}/plots/overview.png)\n\n'
           + ' · '.join(f'[R{i}]({link}/plots/roi_R{i}.png)' for i in range(1, 6))
           + f' · [误差图]({link}/plots/absolute_difference.png) · [位姿变化]({link}/plots/pose_corrections.png)\n\n'
           + principle + '\n\n<details><summary>复现、里程碑和版本记录</summary>\n\n'
           + f'- 结果目录：`{result}`；配置、原图SHA、各方法来源和扫角中心均记录于`comparison/run_config/`。'
             'CNN/DT使用已保存200步终态；标注两组为20261104_train01普通SSIM Baseline第0/200步。\n'
           + '- 首帧固定；平移生成元各分量1mm、旋转分量/扫角2°限幅。扫角中心继承保存的初始几何；'
             'Edge/CNN/Edge+DT的中心为原7P初始化的冻结假设。\n'
           + f"- 指标数值核验：`{comparison['metric_validation']}`。\n"
           + f'- 实验日志：`{result}/comparison/metrics/`；qsub日志：`{root}/qsub/logs/neuf/{run_id}/`；'
             f'脚本：`{root}/qsub/neuf/trajectory_reconstruction.sh`。实际Job ID在qsub日志中逐项保留。\n'
           + '\n'.join(details) + '\n\n- 实际提交、推送及图像审阅结论由本轮完成后补记；根报告和工作台无所属Git仓库。'
             '预存用户修改保持不动。未配置新的21:00自动任务。\n\n</details>\n')
old = report_path.read_text() if report_path.exists() else f'# {run_id[:4]}-{run_id[4:6]}-{run_id[6:8]} 实验记录\n'
pattern = re.compile(re.escape(heading)+r'\n.*?(?=\n## |\Z)', re.S)
updated = pattern.sub(lambda match: section.rstrip(), old, count=1) if heading in old else old+'\n'+section
assert report_path.read_text() == old, '日报并发修改，拒绝覆盖'
report_path.write_text(updated.rstrip()+'\n')
print('工作台与日报已加入本轮28组实际结果：', notebook_path, report_path, flush=True)
PY
fi
