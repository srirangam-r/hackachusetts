# Demo video (2:00)

Rendered: [`media/motor_memory.mp4`](media/motor_memory.mp4). Source: [`../video/`](../video/) (Remotion).
Captions only, no voice-over, no music, no third-party logos. All footage is from our own runs.

| time | scene | on screen |
|---|---|---|
| 0:00 | Title | "Motor Memory: a robot that plans with Kiro and moves by copying human demos", next to the sim camera of the hero run opening the drawer |
| 0:05 | Problem | Fine-tuned VLAs memorize their training scenes: they score 79 to 98% on LIBERO-Goal, but with a new instruction (LIBERO-Pro) OpenVLA 0%, π0 0%, π0.5 ≈2%. Frontier models reason better but can't move a robot precisely. |
| 0:14 | Three layers | Kiro decides what (a two-line program), SAM 3 finds where (the marked front camera image Kiro got), human demos show how (the LIBERO-Object butter demo playing) |
| 0:26 | Live demo page (34 s) | The real page replaying a recorded run at 2x ("open the drawer, put the butter in there and close it, then put the orange juice in the tray"), turns appearing as they happened. Zooms: the human demo each motion copied; what Kiro sees ("no simulator state"); the orange juice grasp missing, Kiro looking again and retrying; the finished task (67 s, 8 Kiro calls). |
| 1:00 | Same demos, different robot | A UR5e puts the wine bottle on the cabinet with a grasp copied from a Panda demo of a different task ("put the wine bottle on the rack") |
| 1:09 | Editability | "Delete a demo, the skill is gone. Add it back, it returns." The black book's `NO_CANDIDATE` status with its demos removed, and the restored pick |
| 1:18 | Results | One chart across familiar tasks, moved objects and new instructions: the fine-tuned VLA band (OpenVLA, π0, π0.5) starts higher and falls to about zero; Motor Memory starts lower and stays above it (on par with the best VLA on moved objects, about a quarter solved on new instructions). Our gap is in parts we can improve one at a time (grasps, drawer handles, insertion, perception) with no retraining. |
| 1:31 | Built with Kiro, and run by Kiro | Spec, steering, hooks, the PBT counterexample, MCP, Kiro as the runtime brain, with real excerpts from `.kiro/` |
| 1:53 | Close | Repo URL |

Notes:
- The live-demo footage is a replay of a real recorded run in our own page (labelled "Recorded run, 2x" on screen), filmed frame by frame with `video/scripts/capture.mjs`.
- The UR5e clip uses a scripted program (no Kiro call) with real SAM 3 perception and the Panda demos; the shot is about the motion transfer. With Kiro as the brain, the UR5e solved the same task in 3 of 5 episodes ([`results/goal_ur5e`](../results/goal_ur5e/summary.md), task 2).
