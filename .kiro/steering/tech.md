---
inclusion: always
---
# Tech

## Environment
- Python 3.12 venv at the main checkout: `/home/nail/stuff/sundai/hackachusetts/.venv` (worktrees reuse it by absolute path; do not create new venvs).
- Always run with `env -u PYTHONPATH MUJOCO_GL=egl` (ROS pollutes PYTHONPATH; rendering is headless EGL).
- Tests: `env -u PYTHONPATH MUJOCO_GL=egl /home/nail/stuff/sundai/hackachusetts/.venv/bin/python -m pytest -q`.
- Property-based tests use Hypothesis. Keep each test file fast (<30 s); mark sim-heavy tests `@pytest.mark.sim`.
- Data lives only in the main checkout: `MOTORMEM_DATA` (default `/home/nail/stuff/sundai/hackachusetts/data`). Never write into `data/raw/`.
  - `data/raw/libero_hdf5/{libero_goal,libero_spatial,libero_object,libero_10,libero_90}/*.hdf5` (original LIBERO demos with sim states)
  - `data/raw/mimicgen/source/*.hdf5` (MimicGen source demos, 10 each; includes `pick_place`, `nut_assembly`)
  - `data/raw/libero_pro/` (LIBERO-Pro bddl + init files), `data/raw/LIBERO-PRO-src/` (code)
- GPU: one RTX 5090 laptop (24 GB). SAM 3 weights: `facebook/sam3` (HF cache, gated; may still be pending). Fallback: `facebook/sam2.1-hiera-large` (cached) + `IDEA-Research/grounding-dino-base`.

## Simulator facts (verified)
- LIBERO = robosuite 1.4.0, `OffScreenRenderEnv`, OSC_POSE delta actions (7-D: dpos, drot (world frame, axis-angle), gripper -1 open / +1 close), 20 Hz. Action scale: 1.0 = 0.05 m / 0.5 rad per step.
- `~/.libero/config.yaml` points at `/home/nail/stuff/LIBERO-plus`; our env code must set paths explicitly to the installed `libero` package's bddl/init files (or LIBERO-Pro's), not rely on that file.
- Pass `camera_depths=True`, use 256x256; images are OpenGL-flipped (flip vertically). Convert depth with `robosuite.utils.camera_utils.get_real_depth_map`; K/extrinsics via `get_camera_intrinsic_matrix` / `get_camera_extrinsic_matrix`. `transform_from_pixels_to_world` asserts on (H,W,1) input: deproject manually.
- Wrist F/T: `env.robots[0].ee_force`, `ee_torque` (not in obs by default).
- UR5e in LIBERO: subclass robosuite `UR5e` as `MountedUR5e` (copy `MountedPanda.base_xpos_offset`), register `ROBOT_CLASS_MAPPING["MountedUR5e"]=SingleArm`, pass `robots=["UR5e"]`. LIBERO init states are Panda-sized (98 vs 92): NEVER `set_init_state` directly (corrupts sim, false success). Copy object qpos (`panda_qpos[9:] -> ur5e_qpos[12:]`), keep UR5e default joints, zero qvel.
- Articulated joints: cabinet drawers slide (axis y, range -0.16..0.01; open < -0.14, closed > 0.0); microwave hinge z (-2.094..0; open < -1.3, closed > -0.005); stove knob hinge z (-0.005..2.1; on >= 0.5).
- Gripper control frame is the grip site for every robosuite gripper; gripper action semantics are identical across grippers.

## Conventions
- Poses are 4x4 numpy float64 `T_a_b` = pose of frame b expressed in frame a. Use `motormem.geometry` helpers only.
- Quaternions: robosuite obs are xyzw. Convert only via `motormem.geometry`.
- No new dependencies without pinning them in `pyproject.toml`.
