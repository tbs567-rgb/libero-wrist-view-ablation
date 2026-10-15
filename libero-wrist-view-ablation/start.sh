#!/bin/sh
# LIBERO 腕部视角实验系统 —— Linux/macOS 入口
cd "$(dirname "$0")"
PY=python3; [ -f ".venv/bin/python" ] && PY=.venv/bin/python
echo "1) 环境自检  2) 训练  3) 评测  4) 探针  5) 图表  6) 数字复核  0) 退出"
printf "选择: "; read c
case "$c" in
  1) $PY -c "import mujoco,torch;print('MuJoCo',mujoco.__version__,'| PyTorch',torch.__version__)";;
  2) echo "用法: $PY low_resource_official.py train --h5 <h5> --epochs 600 [--dual-cam|--wrist-cam]";;
  3) echo "用法: LIBERO_EVAL_CAM=256 $PY low_resource_official.py eval --h5 <h5> --task-id <id> [--dual-cam]";;
  4) $PY stage_analysis.py;;
  5) $PY make_figs.py && $PY make_figs_v2.py;;
  6) $PY data_audit.py;;
  *) exit 0;;
esac
