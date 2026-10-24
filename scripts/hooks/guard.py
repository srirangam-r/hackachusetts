#!/usr/bin/env python3
"""PreToolUse guard for shell commands. Exit 2 blocks the command (stderr goes to the agent).

Protects downloaded data and git history while agents run with --trust-all-tools.
"""
import json
import re
import sys

BLOCK = [
    (r"\brm\s+(-[a-zA-Z]*\s+)*[^|;&]*\bdata/raw\b", "deletes downloaded raw data (data/raw)"),
    (r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*f?\s+(/|~|\$HOME)(\s|$)", "recursive delete of / or home"),
    (r"\bgit\s+push\b[^|;&]*(\s--force\b|\s-f\b|\s--force-with-lease\b)", "force push"),
    (r"\bgit\s+reset\s+--hard\b", "git reset --hard"),
    (r"\bgit\s+clean\s+-[a-zA-Z]*f", "git clean -f"),
    (r"\bgit\s+branch\s+-D\b", "force-deleting a branch"),
    (r"\bgit\s+worktree\s+remove\s+[^|;&]*--force", "force-removing a worktree"),
    (r">\s*\S*data/raw/", "writes into data/raw"),
]


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0
    cmd = (payload.get("tool_input") or {}).get("command") or ""
    for pattern, why in BLOCK:
        if re.search(pattern, cmd):
            print(f"Blocked by guard hook: {why}. Ask the user if this is really intended.", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
