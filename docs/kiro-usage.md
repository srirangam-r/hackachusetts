# How we used Kiro

Kiro played two roles here. It was the development environment, and at run time headless `kiro-cli` *is* the robot's brain. Everything below points at files in this repo.

## Tools: CLI vs IDE

We started in the Kiro IDE and moved to `kiro-cli` after running into permission friction. There's no workspace-wide "trust all tools" option, so shell commands kept prompting (details in [kiro-feedback.md](kiro-feedback.md)). With `kiro-cli chat -a`, the coordinator and the stream sub-agents ran unattended, and a guard hook (below) provided the safety net.

At run time, `motormem.brain.runner` launches one headless session per episode:

```
kiro-cli chat --no-interactive -a --trust-workspace --require-mcp-startup --agent robot-brain --model <M> "<instruction>"
```

`--parallel N` runs several episodes at once. `--model` swaps the frontier model without touching any code.

## Spec-driven development

We wrote a single spec, [`.kiro/specs/motor-memory-harness/`](../.kiro/specs/motor-memory-harness/): `requirements.md` → `design.md` → `tasks.md`. Tasks are grouped by stream: 0 contracts, then A env, B perception, C memory, D skills and E brain, then integration. The interfaces in `design.md` (function signatures, dataclasses in `motormem/types.py`) were the contract between streams. That let five sub-agents build in five git worktrees (`stream/env` … `stream/brain`) in parallel, each using small fakes for its neighbours until the real module landed. The merges into `main` had no conflicts except in `tasks.md`, where each stream ticked its own boxes. Later streams (`stream/ur5e`, `stream/scenes`, `stream/docs`) followed the same pattern.

Compared with vibe coding: with five agents working at once, an unwritten interface becomes a merge conflict or a silent mismatch. The spec made those decisions explicit up front, for example the pose convention `T_a_b`, xyzw quaternions and the `strip_privileged` contract. The correctness properties in `design.md` became the property tests below.

## Hooks

There are four hooks in [`.kiro/hooks/`](../.kiro/hooks/), with scripts in [`scripts/hooks/`](../scripts/hooks/):

| hook | trigger | what it does |
|---|---|---|
| `guard-destructive` | PreToolUse (`execute_bash`) | `guard.py` exits 2 to block deleting `data/raw`, force push, `reset --hard`, `clean -f`, `branch -D` and forced worktree removal. This is what makes `-a` safe. It even blocked our own test command once, because the command string contained `rm -rf`. |
| `tests-on-save` | PostFileSave (`motormem/.*\.py`) | Runs the fast, non-sim tests for the saved module and reports the failure tail. It's worktree-aware: it sets `PYTHONPATH` to the worktree root, so each stream tests its own `motormem`. |
| `kiro-log` | Stop | Appends branch, session and diff stat to [`kiro-log.md`](kiro-log.md) after every agent turn, which gives a build log for free. |
| `session-status` | SessionStart | Injects GPU usage, worktrees, background jobs and the latest eval summary, so a new session knows the state without asking. |

A fairness finding: workspace hooks and steering also load into custom-agent sessions. Run from the repo, the robot brain would have seen our steering (including simulator facts) and hooks. The runner therefore builds an isolated workspace per episode (`runs/<name>/ep_XXX/ws/`) that contains only the agent file, the robot protocol and settings (`make_workspace` in `motormem/brain/runner.py`). The tool-search setting also had to be handled there, because `kiro-cli` 2.34 headless runs ignore it per workspace.

## Steering

| file | inclusion | purpose |
|---|---|---|
| `product.md` | always | the claim being tested, deliverables, non-goals |
| `tech.md` | always | environment commands and **verified simulator facts**: action scale, image flip, depth conversion, UR5e init-state remap, joint ranges. Writing these down once stopped agents from rediscovering (or getting wrong) the same facts in every worktree. |
| `structure.md` | always | package ownership per stream and worktree rules: edit only your package, contracts only via the coordinator, commit, never push |
| `fairness.md` | always | what the brain and skills may and may not use at test time (summarised in the README) |
| `robot-protocol.md` | **manual** | the brain's operating protocol. Only the `robot-brain` agent loads it (as a resource), so it never pollutes development sessions, and development steering never reaches the brain. |

## MCP

- **alphaXiv** ([`.kiro/settings/mcp.json`](../.kiro/settings/mcp.json)) for the literature review and scoop checks. It surfaced a close competitor (ReCAP) and the Goldberg-lab open Gemini harness whose observe/run interface we adapted.
- **The robot MCP server** (`motormem/brain/mcp_server.py`, FastMCP over stdio) is the brain's *only* interface to the world. It has three tools: `observe(find)`, `run(program)` and `done(summary)`. It owns the sim for one episode, writes `trace.jsonl` and `result.json`, and sanitises every output. A property test checks that no privileged key or success field ever appears in a tool output.

## Custom agents

[`.kiro/agents/robot-brain.md`](../.kiro/agents/robot-brain.md) is the robot brain: `tools: ["@robot"]`, `includeMcpJson: false`, `includePowers: false`, with the robot protocol as its only resource. We checked that it can't use shell, file or web tools. Kiro is therefore the runtime brain, not just the IDE, and the model is swappable per run with `--model`.

## Property-based testing

These test files use Hypothesis (`grep -l hypothesis tests/`):

| file | property guarded |
|---|---|
| `test_geometry_props.py` | pose inverse round-trip; object-frame store/restore round-trip; a segment moves rigidly with its object (the core retargeting assumption); `pose_error` steps reach the target; deproject/project round-trip |
| `test_env_props.py` | `strip_privileged` removes every privileged key and keeps every allowed proprio key, for random observation dicts and all known LIBERO object-key patterns (fairness) |
| `test_brain_program_props.py` | any string either parses into the skill grammar or raises `ProgramError`; valid programs round-trip; the parser never evaluates its input |
| `test_brain_server_props.py` | across random tasks, seeds and actions, no MCP tool output leaks privileged keys or success fields; the sanitiser output is clean |
| `test_memory_index_props.py` | after removing an object or demo, no query result derives from it (editability) |
| `test_memory_segmenting_props.py` | detected gripper events match the generated command transitions; each event is covered by exactly one segment window |
| `test_perception.py` | marks are unique and centroids accurate for random synthetic boxes; mask dedupe keeps the higher-scoring mask |
| `test_skills_articulation_props.py` | line and circle fits recover the generating joint model from noisy samples |

## Powers

None of the registry powers fit a MuJoCo robot harness, so we didn't use any. TODO: package the robot harness (MCP server + `robot-brain` agent + protocol steering) as a Kiro power, so others can drop a frontier-model brain onto their own sim.

## Feedback

See [kiro-feedback.md](kiro-feedback.md) for friction points and suggestions we logged as we built.
