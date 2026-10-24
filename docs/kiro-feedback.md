# Kiro feedback

Friction and suggestions noted while building this project (kept as we go).

- **IDE permissions.** There is no IDE-wide "trust all tools" toggle. We hand-edited `~/.kiro/settings/permissions.yaml` with blanket allows; file tools stopped prompting, but shell still prompted because of an ask rule we could not trace (the log said 33 rules loaded, only 14 were visible in files). `kiro-cli chat -a` fixed it instantly, so we moved to the CLI.
- **The agent cannot write `~/.kiro/settings`** (by design). Good for safety, but the agent then cannot fix permission problems for the user; a guided "allow all for this workspace" action would help.
- **`/agent schema` is referenced in docs but `kiro-cli agent schema` does not exist in 2.34**; we read the bundle to confirm agent fields.
- **Hooks are great under `-a`.** A PreToolUse guard (exit 2) is the safety net that makes `--trust-all-tools` usable; it even blocked our own test command because the command string contained `rm -rf data/raw`. A built-in "dry-run this hook against a sample payload" command would make testing hooks easier.
- **Large MCP outputs** (alphaXiv PDF pages, ~40k chars) are saved to files with a preview. Works well; a per-server output limit setting would help.
- **Hooks created mid-session don't apply to that session.** `createHook` says the hook is active "on the next session start", so our long-running coordinator session never ran its own Stop/SessionStart hooks. An option to hot-load new hooks, as steering and MCP config already reload, would help.
- **Workspace hooks and steering reach custom-agent sessions.** Our `robot-brain` agent (tools restricted to one MCP server) still got the workspace's SessionStart hook and always-included steering. For an evaluation agent that would leak information (our steering contains simulator facts). We worked around it with an isolated workspace per episode. A per-agent `inheritWorkspaceSteering: false` / `hooks: inherit|none` switch would make that unnecessary.
- **Headless agents ignore the workspace tool-search setting**, so robot tools are loaded on demand, which costs one extra model turn per episode. Only the global setting works.
- **`kiro-cli agent validate` accepts only JSON agents**, and `agent list` doesn't show markdown agents, although `--agent <name>` loads them fine.
- **What worked very well:** headless `kiro-cli` as a runtime robot brain over MCP, with images in tool results and `--model` swappable per run. Combined with `-a` plus a PreToolUse guard and git worktrees, this let one coordinator session drive 11 parallel sub-agent streams in an afternoon.
