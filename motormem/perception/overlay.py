"""Overlays for the brain: numbered masks, a metric grid in the robot base frame, the gripper pinch point."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from motormem import geometry as G
from motormem.types import CameraFrame, ObjectEstimate, Observation

from .core import get_camera, label_anchor, valid_depth

PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
           (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (170, 110, 40), (128, 0, 0)]
CYAN = (0, 255, 255)
JAW_AXIS = 0            # grip-site axis along which the fingers close (robosuite Panda: x; verified in sim)
MAX_JAW_WIDTH = 0.08    # metres at gripper_opening == 1 (Panda); only used to draw the jaw line
GRID_STEP = 0.10


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


# ---------------------------------------------------------------- table plane
def fit_table_plane(points: np.ndarray, iters: int = 200, tol: float = 0.01, min_up: float = 0.95,
                    rng: np.random.Generator | None = None) -> tuple[float, np.ndarray] | None:
    """RANSAC for the dominant near-horizontal plane. Returns (height z, inlier mask) or None.

    Among horizontal planes with >= 60% of the best inlier count, the lowest one wins (the table,
    not the top of a cabinet that happens to fill much of a close view).
    """
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(P) < 50:
        return None
    rng = rng or np.random.default_rng(0)
    found: list[tuple[int, float]] = []
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9 or abs(n[2]) / nn < min_up:
            continue
        n /= nn
        cnt = int((np.abs((P - a) @ n) < tol).sum())
        found.append((cnt, float(np.median(P[np.abs((P - a) @ n) < tol][:, 2]))))
    if not found:
        return None
    best = max(c for c, _ in found)
    z = min(h for c, h in found if c >= 0.6 * best)
    # refine on the horizontal slab around z
    inl = np.abs(P[:, 2] - z) < tol
    z = float(np.median(P[inl, 2]))
    return z, np.abs(P[:, 2] - z) < tol


def table_height(cam: CameraFrame, stride: int = 2) -> tuple[float, np.ndarray] | None:
    """(z, inlier world points) of the table plane seen in `cam`, from depth only."""
    d = np.where(valid_depth(cam.depth, 3.0), cam.depth, 0.0)
    P = G.deproject(d, cam.K, cam.T_world_cam, stride=stride, max_depth=3.0)
    r = fit_table_plane(P)
    if r is None:
        return None
    z, inl = r
    return z, P[inl]


# ---------------------------------------------------------------- drawing helpers
def _polyline_world(draw, pts_w, cam: CameraFrame, s: float, fill, width: int):
    uv, z = G.project(pts_w, cam.K, cam.T_world_cam)
    H, W = cam.depth.shape[:2]
    ok = (z > 0.02) & (uv[:, 0] > -W) & (uv[:, 0] < 2 * W) & (uv[:, 1] > -H) & (uv[:, 1] < 2 * H)
    for i in range(len(uv) - 1):
        if ok[i] and ok[i + 1]:
            draw.line([tuple(uv[i] * s), tuple(uv[i + 1] * s)], fill=fill, width=width)


def _text(draw, xy, txt, font, fill=(255, 255, 255), anchor="mm", stroke=3):
    draw.text(xy, txt, font=font, fill=fill, anchor=anchor, stroke_width=stroke, stroke_fill=(0, 0, 0))


def _draw_grid(draw, obs: Observation, cam: CameraFrame, s: float, font):
    th = table_height(cam)
    if th is None:
        return
    z_tab, inl = th
    T_bw = G.inv(obs.T_world_base)
    Pb = G.transform_points(T_bw, inl)
    lo = np.floor(np.percentile(Pb[:, :2], 2, axis=0) / GRID_STEP) * GRID_STEP
    hi = np.ceil(np.percentile(Pb[:, :2], 98, axis=0) / GRID_STEP) * GRID_STEP
    R, t = obs.T_world_base[:3, :3], obs.T_world_base[:3, 3]

    def to_world(xb, yb):
        # base-frame (x, y) on the table plane: solve for z_b so the world z equals the table height
        zb = (z_tab - t[2] - R[2, 0] * xb - R[2, 1] * yb) / (R[2, 2] if abs(R[2, 2]) > 1e-6 else 1.0)
        return G.transform_points(obs.T_world_base, np.stack([xb, yb, zb], -1))

    H, W = cam.depth.shape[:2]
    col = (255, 255, 255, 110)
    xs = np.arange(lo[0], hi[0] + 1e-9, GRID_STEP)
    ys = np.arange(lo[1], hi[1] + 1e-9, GRID_STEP)
    n = 40
    for x in xs:
        _polyline_world(draw, to_world(np.full(n, x), np.linspace(lo[1], hi[1], n)), cam, s, col, 1)
    for y in ys:
        _polyline_world(draw, to_world(np.linspace(lo[0], hi[0], n), np.full(n, y)), cam, s, col, 1)

    def label_at(pw, txt):
        uv, z = G.project(pw, cam.K, cam.T_world_cam)
        u, v = uv[0]
        if z[0] > 0.02 and 0 <= u < W and 0 <= v < H:
            _text(draw, (u * s, v * s), txt, font, fill=(255, 255, 160), stroke=2)
            return True
        return False

    # x labels along the y edge nearest the camera-visible side, y labels along the low-x edge
    for x in xs:
        for y in (lo[1], hi[1], (lo[1] + hi[1]) / 2):
            if label_at(to_world(np.array([x]), np.array([y])), f"x{round(100 * x):d}"):
                break
    for y in ys:
        for x in (lo[0], hi[0], (lo[0] + hi[0]) / 2):
            if label_at(to_world(np.array([x]), np.array([y])), f"y{round(100 * y):d}"):
                break


def _draw_gripper(draw, obs: Observation, cam: CameraFrame, s: float, font):
    T = obs.proprio.T_world_eef
    p = T[:3, 3]
    uv, z = G.project(p[None], cam.K, cam.T_world_cam)
    if z[0] <= 0.01:
        return
    u, v = uv[0] * s
    half = 0.5 * max(obs.proprio.gripper_opening * MAX_JAW_WIDTH, 0.02)
    axis = T[:3, JAW_AXIS]
    _polyline_world(draw, np.stack([p - half * axis, p + half * axis]), cam, s, CYAN, 3)
    r = 9
    draw.line([(u - r, v), (u + r, v)], fill=CYAN, width=2)
    draw.line([(u, v - r), (u, v + r)], fill=CYAN, width=2)
    draw.ellipse([u - 4, v - 4, u + 4, v + 4], outline=CYAN, width=2)


# ---------------------------------------------------------------- public
def render_overlay(obs: Observation, objects: list[ObjectEstimate], camera: str = "agentview",
                   size: int = 512, grid: bool = True, gripper: bool = True) -> Image.Image:
    cam = get_camera(obs, camera)
    H, W = cam.rgb.shape[:2]
    s = size / W
    base = Image.fromarray(cam.rgb).convert("RGBA").resize((size, int(round(H * s))), Image.BILINEAR)
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    font = _font(max(12, size // 22))
    small = _font(max(9, size // 40))

    # mask fills
    fill = np.zeros((H, W, 4), dtype=np.uint8)
    for o in objects:
        if o.camera in (cam.name, camera) and o.mask.shape == (H, W):
            fill[o.mask] = (*PALETTE[(o.mark - 1) % len(PALETTE)], 100)
    masks = Image.fromarray(fill, "RGBA").resize(base.size, Image.NEAREST)
    img = Image.alpha_composite(base, masks)

    if grid:
        _draw_grid(draw, obs, cam, s, small)
    if gripper:
        _draw_gripper(draw, obs, cam, s, small)
    # marks last so they are on top
    for o in objects:
        if o.camera in (cam.name, camera) and o.mask.shape == (H, W) and o.mask.any():
            u, v = label_anchor(o.mask)
            u, v = (u + 0.5) * s, (v + 0.5) * s
        else:   # estimate from another camera: put the mark at its projected centroid
            uv, z = G.project(o.T_world_obj[:3, 3][None], cam.K, cam.T_world_cam)
            if z[0] <= 0.01 or not (0 <= uv[0, 0] < W and 0 <= uv[0, 1] < H):
                continue
            u, v = uv[0] * s
        col = PALETTE[(o.mark - 1) % len(PALETTE)]
        r = font.size * 0.75
        draw.ellipse([u - r, v - r, u + r, v + r], fill=(0, 0, 0, 170), outline=col + (255,), width=2)
        _text(draw, (u, v), str(o.mark), font, stroke=2)
    return Image.alpha_composite(img, layer).convert("RGB")


def describe(objects: list[ObjectEstimate], obs: Observation) -> str:
    """Compact text for the brain: marks, labels, base-frame positions and sizes in cm, gripper state."""
    T_bw = G.inv(obs.T_world_base)
    pe = G.transform_points(T_bw, obs.proprio.T_world_eef[:3, 3][None])[0] * 100
    op = obs.proprio.gripper_opening
    lines = [f"gripper: opening {op:.2f} ({'open' if op > 0.6 else 'closed' if op < 0.15 else 'partly open'}), "
             f"pinch at x={pe[0]:.1f} y={pe[1]:.1f} z={pe[2]:.1f} cm",
             "objects (robot base frame: x forward, y left, z up; cm):"]
    if not objects:
        lines.append("  (none found)")
    for o in sorted(objects, key=lambda o: o.mark):
        c = G.transform_points(T_bw, o.T_world_obj[:3, 3][None])[0] * 100
        e = o.extent * 100
        lines.append(f"  [{o.mark}] {o.label}: x={c[0]:.1f} y={c[1]:.1f} z={c[2]:.1f}  "
                     f"size {e[0]:.0f}x{e[1]:.0f}x{e[2]:.0f}  ({o.camera}, score {o.score:.2f})")
    return "\n".join(lines)
