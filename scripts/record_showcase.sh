#!/bin/bash
# Record the demo-video episodes (Kiro brain, SAM 3, real backend) and render composites with make_video.py.
# Usage: scripts/record_showcase.sh            (writes runs/videos/showcase_*.mp4)
cd "$(dirname "$0")/.." || exit 1
PY="env -u PYTHONPATH MUJOCO_GL=egl .venv/bin/python"
R="$PY -m motormem.brain.runner --backend real --perception sam3 --model auto --episodes 1 --record --max-steps 900 --timeout 420"
$R --suite libero_goal --tasks 3,7,8,9 --arm Panda --parallel 4 --name show_goal_panda --index-path data/index/libero_goal.pkl
$R --suite libero_goal --tasks 3,8,9 --arm UR5e --parallel 3 --name show_goal_ur5e --index-path data/index/libero_goal.pkl
$R --suite libero_pro_goal_swap --tasks 8,9 --arm Panda --parallel 2 --name show_pro_swap --index-path data/index/libero_goal.pkl
for run in show_goal_panda show_goal_ur5e show_pro_swap; do
  for ep in runs/$run/ep_*; do
    $PY scripts/make_video.py "$ep" --out "runs/videos/showcase_${run}_$(basename "$ep").mp4" || echo "video failed: $ep"
  done
done
# Scene demos (scripted programs over observed marks, not Kiro): desk sorting and industrial bin sorting
$PY scripts/scene_demo.py desk_sort --layout 0 --arm Panda --motion hybrid --record showcase_desk_panda.mp4 --log runs/videos/showcase.jsonl
$PY scripts/scene_demo.py bin_sort --layout 0 --arm UR5e --motion hybrid --record showcase_bin_ur5e.mp4 --log runs/videos/showcase.jsonl
ls -la runs/videos/showcase_*
