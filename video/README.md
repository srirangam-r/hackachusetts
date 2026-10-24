# Demo video

The 2-minute submission video, built with [Remotion](https://www.remotion.dev/). The rendered file is
[`../docs/media/motor_memory.mp4`](../docs/media/motor_memory.mp4); the storyboard is in
[`../docs/video-script.md`](../docs/video-script.md).

All footage is our own: recorded runs from `runs/` (sim camera frames, the live demo page replaying a recorded run,
SAM 3 marked images, LIBERO demo frames). No music and no third-party logos.

## Rebuild

`runs/` is git-ignored, so the footage has to be regenerated on a machine that has the recorded runs.

```bash
npm install
# 1. serve the recorded live session the hero shot comes from (repo root)
env -u PYTHONPATH MUJOCO_GL=egl ../.venv/bin/python ../scripts/live.py --name live_184743 --no-browser --port 8790
# 2. film the live page replaying that run, frame by frame (headless Chrome; ?reveal=1 shows turns as they happen)
npm run capture -- --url http://localhost:8790 --ep ep_002 --name page_ep002
# 3. copy the other clips and stills into public/footage/
npm run assets
# 4. render (about a minute), or preview with `npm run studio`
npm run render
```

The UR5e clip is `runs/ur5e_clip_wine/ep_000`, recorded with
`python -m motormem.brain.runner --suite libero_goal --tasks 2 --episodes 2 --arm UR5e --scripted --record --name ur5e_clip_wine`
(scripted program, real SAM 3 perception, Panda demos from the motor memory).

The results chart reads `src/results.json`; update it when the final evaluation numbers change.
