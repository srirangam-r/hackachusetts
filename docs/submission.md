# Hackachusetts submission draft

**Track:** Robotics & Hardware
**Project:** Motor Memory: a frontier-model robot harness, built with and run by Kiro
**Repo:** https://github.com/srirangam-r/hackachusetts (make public before submitting)

## Text description

Robot foundation models (VLAs) are fine-tuned per robot and per benchmark, and they mostly memorize. Move the objects on LIBERO-Pro and π0.5 falls to around 13% on average, while OpenVLA and π0 fall to 0%. Frontier vision-language models reason far better but can't move a robot precisely. Systems that let them write code reach high scores only after practicing each test task with the simulator's success signal.

Motor Memory makes the action layer deliberately simple:
- **What:** a frontier model running inside Kiro (headless `kiro-cli`, a custom `robot-brain` agent restricted to one MCP server) reads the instruction and two camera images with numbered object marks. It writes a 2–3 line program such as `pick(2, part="handle")` / `place(5, "in")`. It never generates code.
- **Where:** SAM 3 masks plus depth give each object and part a 3-D frame.
- **How:** recorded human demos stored as gripper poses relative to the object they act on (the motor memory). They're re-targeted onto the live object and executed with operational-space control. Objects with no demo get a grasp computed from their point cloud. Every skill checks its own result (is it holding, did it land on the target, did the plate move far enough) and retries once.

What this gives:
- **Fair by construction:** no simulator state, no success signal and no per-task prompts reach the brain. Kiro's tool restriction plus a property-tested observation filter enforce it.
- **Fast and cheap:** 2–4 model calls per episode.
- **Cross-embodiment:** the same Panda demos run on a UR5e with zero UR5e demos.
- **Editable and traceable:** delete an object's demos and the skill disappears (`NO_CANDIDATE`); add one and it comes back. Every motion traces to a demo id or to "geometry".

Results so far: RESULTS_TABLE

Applications shown: a cluttered desk sort (books into a caddy, mugs on a tray) and an industrial kitting task (sorting parts into bins on a UR5e). Teaching a new part means recording one demo, not retraining a model.

## How Kiro was used
See [kiro-usage.md](kiro-usage.md). Short version:
- **Spec-driven:** one spec whose interfaces let 5, then 11, parallel Kiro agents build in git worktrees.
- **Steering:** fairness rules and verified simulator facts.
- **Hooks:** a guard that makes `--trust-all-tools` safe; worktree-aware tests on save.
- **Property-based tests:** Hypothesis found a real bug in our joint-fit tolerance.
- **MCP:** alphaXiv for the literature review; our own robot MCP server.
- **A custom power** packaging the harness.
- **Kiro as the runtime robot brain,** with swappable models.
- **Transparency:** after our Kiro credits ran out, the demo video, the live demo page and a couple of late fixes were done with Claude Code. Kiro is still the robot's brain in every run.

## Testing instructions
`README.md` → Setup / Run. A sim-free dry run: `python -m motormem.brain.runner --backend fake --suite libero_goal --tasks 8 --episodes 1 --name dry`. A full run needs the LIBERO demos (HF `yifengzhu-hf/LIBERO-datasets`), SAM 3 access (HF `facebook/sam3`) and `kiro-cli`.
