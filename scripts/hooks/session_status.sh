#!/bin/bash
# SessionStart hook: stdout is added to the agent's context.
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)" || exit 0
echo "## Session status ($(date +%H:%M))"
echo "- Submission deadline: 20:00 today."
echo "- GPU: $(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null)"
echo "- Branch: $(git branch --show-current 2>/dev/null)"
echo "- Worktrees:"; git worktree list 2>/dev/null | sed 's/^/  - /'
echo "- Background jobs (eval/build/download):"
ps -eo pid,etime,args | grep -E 'build_index|runner|run_eval|hf download|sam3_poll' | grep -v grep | cut -c1-140 | sed 's/^/  - /'
if [ -f runs/latest_summary.txt ]; then echo "- Latest eval summary:"; sed 's/^/  /' runs/latest_summary.txt | head -20; fi
exit 0
