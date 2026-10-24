"""Fast perception tests on synthetic top-down depth scenes (no models, no simulator)."""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from PIL import Image

from motormem import geometry as G
from motormem.perception import segmenters as S
from motormem.perception.core import DEFAULT_PROMPTS, Perception, dedupe, mask_iou, remove_outliers
from motormem.perception.oracle import OraclePerception, match_name
from motormem.perception.overlay import describe, fit_table_plane, render_overlay, table_height
from motormem.perception.segmenters import MaskCandidate, mask_box, nms
from motormem.types import CameraFrame, ObjectEstimate, Observation, Proprio

H = W = 128
F = 120.0
CAM_H = 1.0
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
R_DOWN = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])   # OpenCV camera looking straight down


def render(boxes, cam_pos=(0.0, 0.0, CAM_H)):
    """boxes: list of (cx, cy, sx, sy, top). Returns depth, per-box masks, T_world_cam."""
    T = G.make_pose(cam_pos, R_DOWN)
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    depth = np.full((H, W), cam_pos[2], dtype=np.float64)   # table at z = 0
    masks = []
    for cx, cy, sx, sy, top in boxes:
        z = cam_pos[2] - top
        wx = cam_pos[0] + (u + 0.0 - K[0, 2]) * z / F
        wy = cam_pos[1] - (v - K[1, 2]) * z / F
        m = (np.abs(wx - cx) <= sx / 2) & (np.abs(wy - cy) <= sy / 2)
        depth = np.where(m, np.minimum(depth, z), depth)
        masks.append(m)
    return depth.astype(np.float32), masks, T


def make_obs(boxes, cam_name="agentview", extra=None):
    depth, masks, T = render(boxes)
    rgb = np.full((H, W, 3), 120, dtype=np.uint8)
    for m in masks:
        rgb[m] = (200, 40, 40)
    cams = {cam_name: CameraFrame(cam_name, rgb, depth, K.copy(), T)}
    if extra:
        cams.update(extra)
    eef = G.make_pose([0.0, 0.0, 0.3], R_DOWN)
    pr = Proprio(eef, np.array([0.04, -0.04]), 1.0, np.zeros(7), np.zeros(3), np.zeros(3))
    return Observation(cams, pr, "test", 0, G.make_pose([-0.5, 0.0, 0.0])), masks


class FakeSegmenter:
    name = "fake"

    def __init__(self, cands_by_label):
        self.cands_by_label = cands_by_label
        self.calls = []

    def segment(self, rgb, prompts):
        self.calls.append(list(prompts))
        return [c for p in prompts for c in self.cands_by_label.get(p, [])]


def cand(mask, label="block", score=0.9):
    return MaskCandidate(mask, label, score, mask_box(mask))


BOXES = [(-0.10, 0.05, 0.06, 0.04, 0.05), (0.08, -0.02, 0.04, 0.04, 0.10), (0.0, 0.12, 0.05, 0.03, 0.03)]


# ---------------------------------------------------------------- scan
def test_scan_one_estimate_per_instance_marks_left_to_right():
    obs, masks = make_obs(BOXES)
    seg = FakeSegmenter({"block": [cand(m, score=0.5 + 0.1 * i) for i, m in enumerate(masks)]})
    ests = Perception(seg).scan(obs, ["block"])
    assert sorted(e.mark for e in ests) == [1, 2, 3]
    us = [np.nonzero(e.mask)[1].mean() for e in sorted(ests, key=lambda e: e.mark)]
    assert us == sorted(us)
    for e in ests:
        assert e.points_world.shape[1] == 3 and len(e.points_world) >= 20
        assert e.mask.dtype == bool and e.camera == "agentview"
        # frame convention: z up
        assert np.allclose(e.T_world_obj[:3, 2], [0, 0, 1])


def test_scan_centroid_accuracy_and_top_height():
    obs, masks = make_obs(BOXES)
    seg = FakeSegmenter({"block": [cand(m) for m in masks]})
    ests = Perception(seg).scan(obs, ["block"])
    for (cx, cy, sx, sy, top) in BOXES:
        e = min(ests, key=lambda e: np.linalg.norm(e.T_world_obj[:2, 3] - [cx, cy]))
        assert np.linalg.norm(e.T_world_obj[:2, 3] - [cx, cy]) < 0.005
        assert abs(e.T_world_obj[2, 3] - top) < 0.005
        assert abs(max(e.extent[:2]) - max(sx, sy)) < 0.02


def test_edge_bleed_removed_by_erosion_and_outlier_filter():
    obs, masks = make_obs(BOXES[:1])
    from scipy import ndimage
    fat = ndimage.binary_dilation(masks[0], iterations=2)      # mask bleeds onto the table
    ests = Perception(FakeSegmenter({"block": [cand(fat)]})).scan(obs, ["block"])
    assert len(ests) == 1
    z = ests[0].points_world[:, 2]
    assert z.min() > BOXES[0][4] - 0.01                        # no table points survive


def test_few_depth_points_dropped_and_reported():
    obs, masks = make_obs(BOXES[:2])
    obs.cameras["agentview"].depth[masks[1]] = 0.0              # no valid depth on box 2
    tiny = np.zeros((H, W), bool); tiny[5:8, 5:8] = True         # 9 pixels
    seg = FakeSegmenter({"block": [cand(masks[0]), cand(masks[1])], "speck": [cand(tiny, "speck")]})
    P = Perception(seg)
    ests = P.scan(obs, ["block"])
    assert len(ests) == 1
    assert len(P.dropped) == 1 and "valid depth" in P.dropped[0].reason
    ests = P.scan(obs, ["speck"])
    assert ests == [] and len(P.dropped) == 1


def test_dedupe_keeps_higher_score():
    obs, masks = make_obs(BOXES[:1])
    m2 = masks[0].copy(); m2[np.nonzero(masks[0])[0].min(), :] = False   # near-identical mask
    assert mask_iou(masks[0], m2) > 0.7
    seg = FakeSegmenter({"a": [cand(masks[0], "a", 0.4)], "b": [cand(m2, "b", 0.8)]})
    P = Perception(seg)
    ests = P.scan(obs, ["a", "b"])
    assert len(ests) == 1 and ests[0].label == "b" and ests[0].mark == 1
    assert any("duplicate" in d.reason for d in P.dropped)


def test_find_none_uses_generic_prompts():
    obs, masks = make_obs(BOXES[:1])
    seg = FakeSegmenter({"box": [cand(masks[0], "box")]})
    ests = Perception(seg).scan(obs, None)
    assert seg.calls[-1] == DEFAULT_PROMPTS and len(ests) == 1


def test_refine_picks_nearest_candidate_and_keeps_mark():
    obs, masks = make_obs(BOXES)
    # wrist camera: lower, offset camera seeing the same boxes
    depth, wmasks, T = render(BOXES, cam_pos=(0.02, 0.0, 0.5))
    wrist = CameraFrame("wrist", np.zeros((H, W, 3), np.uint8), depth, K.copy(), T)
    obs.cameras["wrist"] = wrist
    seg = FakeSegmenter({"block": [cand(m) for m in masks]})
    P = Perception(seg)
    ests = P.scan(obs, ["block"])
    target = min(ests, key=lambda e: np.linalg.norm(e.T_world_obj[:2, 3] - BOXES[1][:2]))
    seg.cands_by_label = {"block": [cand(m) for m in wmasks]}
    r = P.refine(obs, target, camera="wrist")
    assert r.camera == "wrist" and r.mark == target.mark
    assert np.linalg.norm(r.T_world_obj[:2, 3] - BOXES[1][:2]) < 0.005
    # nothing near -> unchanged
    seg.cands_by_label = {"block": []}
    assert P.refine(obs, target, camera="wrist") is target


# ---------------------------------------------------------------- properties
@settings(max_examples=25, deadline=None)
@given(st.lists(st.tuples(st.floats(-0.2, 0.2), st.floats(-0.2, 0.2), st.floats(0.03, 0.08),
                          st.floats(0.03, 0.08), st.floats(0.02, 0.15)), min_size=1, max_size=4))
def test_scan_props_marks_unique_and_centroids_accurate(boxes):
    # keep boxes apart so masks do not overlap
    for i in range(len(boxes)):
        for j in range(i):
            if abs(boxes[i][0] - boxes[j][0]) < 0.1 and abs(boxes[i][1] - boxes[j][1]) < 0.1:
                return
    obs, masks = make_obs(boxes)
    seg = FakeSegmenter({"o": [cand(m, "o", float(k + 1) / 10) for k, m in enumerate(masks)]})
    ests = Perception(seg).scan(obs, ["o"])
    marks = [e.mark for e in ests]
    assert sorted(marks) == list(range(1, len(ests) + 1))
    for e in ests:
        d = min(np.linalg.norm(e.T_world_obj[:2, 3] - np.array(b[:2])) for b in boxes)
        assert d < 0.01


@settings(max_examples=40, deadline=None)
@given(st.lists(st.tuples(st.integers(0, 50), st.integers(0, 50), st.integers(3, 30), st.integers(3, 30),
                          st.floats(0.0, 1.0)), min_size=1, max_size=8))
def test_dedupe_props(rects):
    cands = []
    for x, y, w, h, s in rects:
        m = np.zeros((64, 64), bool); m[y:y + h, x:x + w] = True
        cands.append(cand(m, score=s))
    out = dedupe(cands, 0.7)
    for i in range(len(out)):
        for j in range(i):
            assert mask_iou(out[i].mask, out[j].mask) <= 0.7
    best = max(cands, key=lambda c: c.score)
    assert any(c is best for c in out)
    # every dropped candidate overlaps a kept one with a score at least as high
    for c in cands:
        if not any(c is k for k in out):
            assert any(mask_iou(c.mask, k.mask) > 0.7 and k.score >= c.score for k in out)


def test_remove_outliers_drops_far_points():
    rng = np.random.default_rng(0)
    P = rng.normal(0, 0.01, (300, 3))
    P[:10] += [0, 0, 0.5]
    keep = remove_outliers(P, cam_z=P[:, 2])
    assert not keep[:10].any() and keep[10:].mean() > 0.9


# ---------------------------------------------------------------- overlay
def test_table_plane_and_overlay():
    obs, masks = make_obs(BOXES)
    z, inl = table_height(obs.cameras["agentview"])
    assert abs(z) < 1e-3 and len(inl) > 100
    seg = FakeSegmenter({"block": [cand(m) for m in masks]})
    ests = Perception(seg).scan(obs, ["block"])
    img = render_overlay(obs, ests, "agentview")
    assert isinstance(img, Image.Image) and img.size == (512, 512) and img.mode == "RGB"
    a = np.asarray(img)
    assert (np.abs(a.astype(int) - [0, 255, 255]).sum(-1) < 30).sum() > 10   # cyan gripper crosshair drawn
    text = describe(ests, obs)
    assert "[1]" in text and "[3]" in text and "cm" in text and "opening 1.00" in text
    # base frame: base at world x=-0.5 -> object at world x=-0.10 is at base x=40 cm
    e = min(ests, key=lambda e: abs(e.T_world_obj[0, 3] + 0.10))
    assert f"[{e.mark}] block: x=40." in text or f"[{e.mark}] block: x=39.9" in text


def test_fit_table_plane_prefers_lowest_dominant():
    rng = np.random.default_rng(1)
    table = np.c_[rng.uniform(-1, 1, (2000, 2)), np.zeros(2000)]
    shelf = np.c_[rng.uniform(-0.3, 0.3, (1500, 2)), np.full(1500, 0.4)]
    z, _ = fit_table_plane(np.r_[table, shelf])
    assert abs(z) < 1e-6


# ---------------------------------------------------------------- segmenter utils
def test_nms_and_factory():
    b = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]], float)
    assert sorted(nms(b, np.array([0.9, 0.8, 0.7]), 0.5)) == [0, 2]
    with pytest.raises(ValueError):
        S.make_segmenter("nope")
    assert isinstance(FakeSegmenter({}), S.Segmenter)


# ---------------------------------------------------------------- oracle
def test_oracle_perception_from_poses_and_points():
    obs, _ = make_obs([])
    cube = np.array([[x, y, z] for x in (-0.02, 0.02) for y in (-0.02, 0.02) for z in (0.0, 0.04)])
    cube = np.repeat(cube, 5, axis=0) + np.random.default_rng(0).normal(0, 1e-3, (40, 3))
    poses = {"akita_black_bowl_1": G.make_pose([0.05, 0.0, 0.0], G.rot_z(0.3)),
             "plate_1": G.make_pose([-0.1, 0.05, 0.0])}
    orc = OraclePerception(poses, {"akita_black_bowl": cube})
    ests = orc.scan(obs, ["black bowl", "plate", "wine bottle"])
    assert [e.label for e in sorted(ests, key=lambda e: e.mark)] == ["plate", "black bowl"]   # left to right
    bowl = next(e for e in ests if e.label == "black bowl")
    assert np.allclose(bowl.T_world_obj[:2, 3], [0.05, 0.0], atol=2e-3)
    assert bowl.mask.any() and len(bowl.points_world) == 40
    plate = next(e for e in ests if e.label == "plate")
    assert np.allclose(plate.T_world_obj[:3, 3], [-0.1, 0.05, 0.0])
    # callable poses + find=None returns everything
    orc2 = OraclePerception(lambda: poses)
    assert len(orc2.scan(obs, None)) == 2
    r = orc.refine(obs, bowl, camera="agentview")
    assert r.mark == bowl.mark and np.allclose(r.T_world_obj, bowl.T_world_obj)


def test_match_name():
    names = ["akita_black_bowl_1", "akita_black_bowl_2", "plate_1", "wooden_cabinet_1", "cream_cheese_1"]
    assert match_name("black bowl", names) == ["akita_black_bowl_1", "akita_black_bowl_2"]
    assert match_name("plate_1", names) == ["plate_1"]
    assert match_name("cabinet", names) == ["wooden_cabinet_1"]
    assert match_name("cream cheese box", names) == ["cream_cheese_1"]
    assert match_name("wine bottle", names) == []
