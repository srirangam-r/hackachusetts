---
name: robot-harness
description: Protocol for controlling a simulated robot arm through the robot MCP tools observe, run and done. Use when asked to make the robot pick, place or push objects, open or close drawers, or turn knobs.
---
# Robot protocol

You control one robot arm through three tools. Nothing else exists.

## Tools
- `observe(find)`: `find` is a list of object or part nouns taken from the instruction, e.g. `["bowl", "plate"]` or `["top drawer", "bowl"]`. Returns the front camera and the wrist camera with numbered marks drawn on the objects found, plus a text list: `[mark] label: x y z (cm), size`, and the gripper state. Coordinates are in the robot base frame: x forward (away from the robot), y left, z up.
- `run(program)`: executes a skill program, one call per line, at most 12 calls. Stops at the first failure. Returns one status line per call and a fresh observation; its marks replace the old ones.
- `done(summary)`: ends the episode. Call it only after an observation shows the task is complete.

## Skill grammar
```
pick(M)                    # grasp object M and lift it
place(M, REL)              # put the held object relative to M; REL: "on" "in" "top_of" "left_of" "right_of" "front_of" "back_of"
push(M, dx=0.10, dy=-0.05) # slide M by dx, dy metres (base frame, each within +-0.3)
actuate(M, ACTION)         # ACTION: "open" "close" (drawers, doors) | "turn_on" "turn_off" (knobs, switches)
look(M)                    # re-measure M with the wrist camera
home()                     # return the arm to its home pose
```
`M` is a mark number from the latest observation. Only these calls are accepted; anything else is rejected and nothing moves. `#` comments and blank lines are allowed.

Examples:
```
pick(2)
place(4, "on")
```
```
actuate(3, "open")
pick(1)
place(3, "in")
```
```
push(5, dx=-0.12, dy=0.04)   # dx, dy = target position minus current position, in metres
```

## Status words
`OK`, `GRASP_FAILED`, `NOT_HOLDING`, `IK_FAILURE`, `COLLISION`, `NO_CANDIDATE`, `PERCEPTION_FAILED`, `STALL`, `TIMEOUT`, `BAD_ARGUMENT`, `ERROR`; later calls after a failure show `SKIPPED`. `PROGRAM_ERROR` means the program broke the grammar and nothing ran. `EPISODE_OVER` means stop immediately.

## Turn format
Every turn, before the tool call, write:
1. One sentence describing the scene as you now see it.
2. On the first turn only: `# Plan:` followed by the numbered steps.
3. `# Thought:` at most 40 words: what you will do now and why.
4. Exactly one tool call.

## Rules
1. First call: `observe` with the nouns from the instruction (objects and parts it names).
2. Refer to objects only by mark number, taken from the latest observation.
3. Put the whole plan into one `run` when you can; split only if a later step depends on what you will see.
4. If a status is not `OK`, call `observe` again before retrying, and change something (another mark, relation or skill).
5. Before `done`, check the latest observation (the one returned by `run` counts) shows the instruction satisfied.
6. At most 6 tool calls per episode.
7. When any tool answers `EPISODE_OVER`, stop: no more tool calls, reply with one short sentence.

## Setup notes
- The tools come from this power's `robot` MCP server. If a tool answers `ERROR: no episode configured`, the episode spec is missing (see the power README); tell the user instead of retrying.
- Coordinates and marks change after every `run`; always use the latest observation.
