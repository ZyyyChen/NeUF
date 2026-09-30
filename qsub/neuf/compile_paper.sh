#!/bin/bash -l
#PBS -N neuf_paper
#PBS -q route
#PBS -l walltime=00:10:00
#PBS -l nodes=1:ppn=1
#PBS -l mem=2gb

set -euo pipefail

# 只编译独立论文稿；沿用本项目 PBS 语法，CPU 作业由 route 分配至 short。
REPO_DIR=/misc/raid/zchen/Code/NeUF
RUN_ID="${RUN_ID:?请传入 RUN_ID=YYYYMMDD_trainNN}"
[[ "${RUN_ID}" =~ ^[0-9]{8}_train[0-9]{2,}$ ]]
PAPER_DIR="${REPO_DIR}/paper"
BUILD_DIR="${PAPER_DIR}/build/${RUN_ID}"
QSUB_LOG_DIR="${REPO_DIR}/qsub/logs/neuf/${RUN_ID}"
mkdir -p "${BUILD_DIR}" "${QSUB_LOG_DIR}"
printf '%s\n' "${PBS_JOBID:-unknown}" > "${QSUB_LOG_DIR}/job_id.txt"

finish() {
    local status=$?
    printf 'exit_code=%s\n' "${status}" > "${QSUB_LOG_DIR}/status.txt"
    printf '结果目录: %s\n编译日志: %s\nqsub日志: %s\n' \
        "${BUILD_DIR}" "${BUILD_DIR}" "${QSUB_LOG_DIR}"
    printf 'qsub脚本: %s/qsub/neuf/compile_paper.sh\nJob ID/状态: %s / exit_code=%s\n' \
        "${REPO_DIR}" "${PBS_JOBID:-unknown}" "${status}"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

cd "${PAPER_DIR}"
printf 'Command: XeLaTeX -> BibTeX -> XeLaTeX -> XeLaTeX, main_en.tex / main_zh.tex\n'
for language in en zh; do
    name="main_${language}"
    xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error \
        -output-directory="${BUILD_DIR}" "${name}.tex" > "${BUILD_DIR}/${name}.pass1.log" 2>&1
    bibtex "build/${RUN_ID}/${name}" > "${BUILD_DIR}/${name}.bibtex.log" 2>&1
    for pass in 2 3; do
        xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error \
            -output-directory="${BUILD_DIR}" "${name}.tex" > "${BUILD_DIR}/${name}.pass${pass}.log" 2>&1
    done
    if grep -E 'There were undefined|Citation .* undefined|Missing character:|Overfull' "${BUILD_DIR}/${name}.log"; then
        printf '编译存在引用、字形或溢出问题，请检查 %s\n' "${BUILD_DIR}/${name}.log"
        exit 1
    fi
    pdfinfo "${BUILD_DIR}/${name}.pdf" | grep -E 'Pages:|Page size:|File size:'
    pdftotext -layout "${BUILD_DIR}/${name}.pdf" "${BUILD_DIR}/${name}.txt"
    printf '已编译并提取文本: %s\n' "${BUILD_DIR}/${name}.pdf"
done

# 预览仅用于检查中英文字形、参考文献及分页，不属于实验图像。
pdftoppm -f 1 -singlefile -scale-to 1400 -png "${BUILD_DIR}/main_en.pdf" "${BUILD_DIR}/preview_en"
pdftoppm -f 1 -singlefile -scale-to 1400 -png "${BUILD_DIR}/main_zh.pdf" "${BUILD_DIR}/preview_zh"
