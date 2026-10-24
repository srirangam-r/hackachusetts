---
inclusion: always
---
# Product: Motor Memory

A robot harness where a frontier model (running inside Kiro) decides **what** to do, segmentation and depth decide **where**, and an index of recorded human demos decides **how** the gripper moves. No generated code in the loop, no fine-tuned VLA, no practice on test tasks.

## The claim we are testing
A fixed, simple, fast action layer under a frontier model can solve rigid tabletop manipulation fairly:
- **Motor memory:** demo segments (grasp, place, push, actuate) stored as gripper poses relative to the object they act on. At run time they are transformed onto the live object and executed with operational-space control (OSC). Adding a demo adds a capability; deleting it removes it; every motion traces to a demo id.
- **Two arms:** the same Panda demos run on a UR5e, because segments live in gripper space.
- **Articulated parts** (drawers, knobs, doors) use one joint-agnostic skill that estimates the joint online from the gripper's own motion. Demos only supply how to grab the handle.
- **Hand-written primitive fallback** (top-down grasp) exists for objects with no demo, and doubles as the ablation: same brain, same perception, swap only the motion source.

## Deliverables (priority order)
1. LIBERO-Goal (10 tasks) on Panda, then LIBERO-Pro Goal (moved objects, new instructions).
2. Same tasks on UR5e with zero UR5e demos.
3. Cluttered desk sorting scene (custom LIBERO study scene) incl. an editability demo: remove an object's demos -> fails/falls back; add one -> works.
4. Industrial bin sorting (robosuite PickPlace) on UR5e.
5. Kiro write-up, video, public repo with `.kiro/`.

## Non-goals
Training models, deformables, real hardware, beating fine-tuned VLAs on standard (saturated) LIBERO.

#[[file:.kiro/steering/fairness.md]]
