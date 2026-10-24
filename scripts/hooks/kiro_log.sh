#!/bin/bash
# Stop hook: append a one-line record of each agent turn to docs/kiro-log.md (evidence for the write-up).
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)" || exit 0
mkdir -p docs
payload=$(cat)
sid=$(printf '%s' "$payload" | python3 -c "import sys,json;print(json.load(sys.stdin).get('session_id','?'))" 2>/dev/null)
stat=$(git diff --shortstat 2>/dev/null | sed 's/^ //')
echo "- $(date '+%Y-%m-%d %H:%M') | $(git branch --show-current 2>/dev/null) | session ${sid:0:16} | ${stat:-no uncommitted changes}" >> docs/kiro-log.md
exit 0
