#!/bin/bash
# Run a queue of evaluations sequentially. Usage: eval_queue.sh <queue-name> "<name>|<suite>|<arm>|<motion>" ...
# Each entry: 10 tasks x $EPISODES episodes, Kiro brain (model auto), SAM 3 perception, real backend.
cd "$(dirname "$0")/.." || exit 1
EPISODES=${EPISODES:-5}
q=$1; shift
for entry in "$@"; do
  IFS='|' read -r name suite arm motion <<< "$entry"
  echo "[$q $(date +%H:%M)] start $name"
  env -u PYTHONPATH MUJOCO_GL=egl .venv/bin/python -m motormem.brain.runner \
    --suite "$suite" --tasks 0-9 --episodes "$EPISODES" --arm "$arm" --motion-source "$motion" \
    --model auto --parallel ${PARALLEL:-4} --name "$name" --backend real --perception sam3 \
    --index-path data/index/libero_goal.pkl --max-steps 900 --timeout 420 > "logs/$name.log" 2>&1
  echo "[$q $(date +%H:%M)] done $name: $(tail -1 runs/$name/summary.md 2>/dev/null)"
done
