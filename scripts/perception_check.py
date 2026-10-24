#!/usr/bin/env python
"""Offline perception accuracy check on LIBERO-Goal init frames (task 2.5).

Renders frames from the installed `libero` package's LIBERO-Goal bddl/init files, runs
`Perception.scan` with the task's object names, and compares estimated centroids to ground truth.
Ground truth (sim body poses + MuJoCo segmentation) is used HERE ONLY, offline, to score perception;
it never reaches the perception code under test.

    env PYTHONPATH=$PWD MUJOCO_GL=egl python scripts/perception_check.py [--segmenter auto] [--inits 2]
"""
from __future__ import annotations

import argparse
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from motormem import config
from motormem import geometry as G
from motormem.perception import Perception, describe, make_segmenter, render_overlay
from motormem.perception.core import erode, valid_depth
from motormem.types import CameraFrame, Observation, Proprio

log = logging.getLogger("perception_check")

# Readable prompts for LIBERO object types (offline check only; the brain chooses its own words).
PROMPTS = {
    "akita_black_bowl": "black bowl", "plate": "plate", "cream_cheese": "cream cheese box",
    "wine_bottle": "wine bottle", "wooden_cabinet": "cabinet", "flat_stove": "stove", "wine_rack": "wine rack",
}
CAMS = {"agentview": "agentview", "wrist": "robot0_eye_in_hand"}
SIZE = 256


def obj_type(name: str) -> str:
    import re
    return re.sub(r"_\d+$", "", name)


def libero_root() -> Path:
    import libero.libero as L
    return Path(L.__file__).parent


def make_env(task: str):
    from libero.libero.envs import OffScreenRenderEnv
    bddl = libero_root() / "bddl_files" / "libero_goal" / f"{task}.bddl"
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=SIZE, camera_widths=SIZE,
                             camera_depths=True)
    env.seed(0)
    return env


def init_states(task: str):
    import torch
    return torch.load(libero_root() / "init_files" / "libero_goal" / f"{task}.pruned_init", weights_only=False)


def gt_segmentation(sim, cam: str) -> np.ndarray:
    """Per-pixel MuJoCo body id (-1 = none), image convention. Offline ground truth only."""
    import mujoco
    m, d = sim.model._model, sim.data._data
    r = mujoco.Renderer(m, SIZE, SIZE)
    try:
        r.update_scene(d, camera=cam)
        r.enable_segmentation_rendering()
        s = r.render()
    finally:
        r.close()
    geom = s[..., 0]
    return np.where((s[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & (geom >= 0), m.geom_bodyid[np.clip(geom, 0, None)], -1)


def build_obs(env, raw: dict) -> Observation:
    """Same construction Stream A's adapter is specified to do (flip, metric depth, OpenCV extrinsics)."""
    from robosuite.utils import camera_utils as cu
    sim = env.sim
    cams = {}
    for short, cam in CAMS.items():
        rgb = np.ascontiguousarray(raw[f"{cam}_image"][::-1])
        depth = cu.get_real_depth_map(sim, raw[f"{cam}_depth"])[::-1, :, 0].astype(np.float32)
        K = cu.get_camera_intrinsic_matrix(sim, cam, SIZE, SIZE)
        T = cu.get_camera_extrinsic_matrix(sim, cam)       # camera->world, already OpenCV axes (checked below)
        cams[short] = CameraFrame(short, rgb, np.ascontiguousarray(depth), K, T)
    sid = sim.model.site_name2id("gripper0_grip_site")
    T_eef = G.make_pose(sim.data.site_xpos[sid], sim.data.site_xmat[sid].reshape(3, 3))
    bid = sim.model.body_name2id("robot0_base")
    T_base = G.make_pose(sim.data.body_xpos[bid], sim.data.body_xmat[bid].reshape(3, 3))
    q = np.asarray(raw["robot0_gripper_qpos"])
    opening = float(np.clip((q[0] - q[1]) / 0.08, 0, 1))
    pr = Proprio(T_eef, q, opening, np.asarray(raw["robot0_joint_pos"]), np.zeros(3), np.zeros(3))
    return Observation(cams, pr, env.language_instruction if hasattr(env, "language_instruction") else "", 0, T_base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segmenter", default="auto")
    ap.add_argument("--inits", type=int, default=2, help="init states per task (10 tasks)")
    ap.add_argument("--out", default=str(config.RUNS_DIR / "perception_check"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    seg = make_segmenter(args.segmenter)
    perc = Perception(seg)
    tasks = sorted(p.stem for p in (libero_root() / "bddl_files" / "libero_goal").glob("*.bddl"))

    rows = []                      # per (frame, gt object)
    lat, eef_ok, n_frames, overlays = [], [], 0, []
    for ti, task in enumerate(tasks):
        env = make_env(task)
        inits = init_states(task)
        for k in range(args.inits):
            env.reset()
            env.set_init_state(inits[k])
            for _ in range(10):
                raw, *_ = env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
            obs = build_obs(env, raw)
            sim = env.sim
            cam = obs.cameras["agentview"]

            # extrinsic check: the grip site must project onto a gripper/robot pixel
            seg_gt = gt_segmentation(sim, "agentview")
            uv, z = G.project(obs.proprio.T_world_eef[:3, 3][None], cam.K, cam.T_world_cam)
            u, v = np.round(uv[0]).astype(int)
            win = seg_gt[max(v - 4, 0):v + 5, max(u - 4, 0):u + 5]
            names = [sim.model.body_id2name(b) for b in np.unique(win) if b >= 0]
            eef_ok.append(bool(z[0] > 0 and any(n.startswith(("gripper0", "robot0")) for n in names)))

            objs = list(env.env.objects_dict) + list(env.env.fixtures_dict)
            find = sorted({PROMPTS.get(obj_type(o), obj_type(o).replace("_", " ")) for o in objs})
            t0 = time.time()
            ests = perc.scan(obs, find, "agentview")
            lat.append(time.time() - t0)
            n_frames += 1

            ok_depth = valid_depth(cam.depth)
            for o in objs:
                bids = [b for b in range(sim.model.nbody) if (sim.model.body_id2name(b) or "").startswith(o)]
                if not bids:
                    continue
                gt_mask = np.isin(seg_gt, bids)
                p_body = sim.data.body_xpos[min(bids)].copy()
                core = erode(gt_mask, 2)
                vis = None
                if (core & ok_depth).sum() >= 20:
                    vis = G.deproject(np.where(ok_depth, cam.depth, 0), cam.K, cam.T_world_cam, mask=core).mean(0)
                label = PROMPTS.get(obj_type(o), obj_type(o).replace("_", " "))
                cands = [e for e in ests if e.label == label]
                best, best_iou = None, 0.0
                for e in cands:
                    iou = float((e.mask & gt_mask).sum() / max((e.mask | gt_mask).sum(), 1))
                    if iou > best_iou:
                        best, best_iou = e, iou
                row = dict(task=task, init=k, obj=o, type=obj_type(o), visible_px=int(gt_mask.sum()),
                           detected=best is not None and best_iou > 0.1, iou=best_iou)
                if row["detected"]:
                    c = best.T_world_obj[:3, 3]
                    row["err_body_3d"] = float(np.linalg.norm(c - p_body))
                    row["err_body_xy"] = float(np.linalg.norm(c[:2] - p_body[:2]))
                    row["err_vis_3d"] = float(np.linalg.norm(c - vis)) if vis is not None else np.nan
                rows.append(row)

            if ti % 3 == 0 and k == 0 and len(overlays) < 3:
                p = out / f"overlay_{task}.png"
                render_overlay(obs, ests, "agentview").save(p)
                wr = [perc.refine(obs, e, "wrist") for e in ests[:3]]
                render_overlay(obs, [e for e in wr if e.camera == "wrist"], "wrist").save(
                    out / f"overlay_{task}_wrist.png")
                (out / f"describe_{task}.txt").write_text(describe(ests, obs) + "\n")
                overlays.append(p)
            log.info("%s init %d: %d estimates in %.2fs", task, k, len(ests), lat[-1])
        env.close()

    # ------------------------------------------------------------ report
    by = defaultdict(list)
    for r in rows:
        by[r["type"]].append(r)

    def q(xs, p):
        xs = [x for x in xs if np.isfinite(x)]
        return f"{100 * np.percentile(xs, p):.1f}" if xs else "-"

    lines = [
        "# Perception check: LIBERO-Goal init frames",
        "",
        f"- Segmenter: `{seg.name}`"
        + (f" (Grounding DINO `{getattr(seg, 'gdino_repo', '')}` + `{getattr(seg, 'sam2_repo', '')}`)" if seg.name == "gdino_sam2" else ""),
        f"- Frames: {n_frames} ({len(tasks)} tasks x {args.inits} init states), agentview {SIZE}x{SIZE}, depth used",
        f"- Scan latency per frame (7 prompts): median {np.median(lat):.2f} s, max {np.max(lat):.2f} s (first call includes warm-up)",
        f"- Extrinsic check (grip site projects onto a gripper/robot pixel): {sum(eef_ok)}/{len(eef_ok)}",
        "",
        "Errors in cm. `body` = estimated centroid vs the sim body origin (includes the visible-surface bias, "
        "e.g. bowl origin is at its base). `visible` = vs the centroid of the ground-truth visible surface "
        "(isolates segmentation error). Detected = matched mask IoU > 0.1 with the right label.",
        "",
        "| object type | n | detected | mask IoU (median) | body xy med | body xy p90 | body 3D med | body 3D p90 | visible 3D med | visible 3D p90 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for t in sorted(by):
        rs = by[t]
        det = [r for r in rs if r["detected"]]
        ious = [r["iou"] for r in det]
        lines.append(
            f"| {t} | {len(rs)} | {len(det)}/{len(rs)} | {np.median(ious) if ious else 0:.2f} | "
            f"{q([r['err_body_xy'] for r in det], 50)} | {q([r['err_body_xy'] for r in det], 90)} | "
            f"{q([r['err_body_3d'] for r in det], 50)} | {q([r['err_body_3d'] for r in det], 90)} | "
            f"{q([r['err_vis_3d'] for r in det], 50)} | {q([r['err_vis_3d'] for r in det], 90)} |")
    det_all = [r for r in rows if r["detected"]]
    lines += ["", f"Overall: detected {len(det_all)}/{len(rows)}; visible-centroid error median "
              f"{q([r['err_vis_3d'] for r in det_all], 50)} cm, p90 {q([r['err_vis_3d'] for r in det_all], 90)} cm.",
              "", "Overlays: " + ", ".join(f"`{p.name}`" for p in overlays), ""]
    (out / "report.md").write_text("\n".join(lines))
    import json
    (out / "rows.json").write_text(json.dumps(rows, indent=1, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
