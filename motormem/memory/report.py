"""Index report: segment counts per kind/object, failures, sanity plots -> runs/index_report.md."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from motormem import config
from motormem.memory.index import MotorMemory
from motormem.types import SegmentKind


def _plot(mem: MotorMemory, kind, name, out: Path, n=8, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    segs = mem.query(kind, name, k=n)
    if not segs:
        return False
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
    P = mem.object_points(name)
    for ax, (i, j, lab) in zip(axs, [(0, 1, "x-y (top)"), (0, 2, "x-z (side)")]):
        if len(P):
            ax.scatter(P[:, i], P[:, j], s=1, c="0.7")
        for s in segs:
            p = s.T_obj_eef[:, :3, 3]
            closed = s.gripper > 0
            ax.plot(p[:, i], p[:, j], lw=1)
            ax.scatter(p[closed, i], p[closed, j], s=4, c="r")
        ax.set_title(f"{lab}")
        ax.set_aspect("equal")
    fig.suptitle(title or f"{kind.value} segments, gripper path in {name} frame (red = closed)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return True


def write_report(index_paths: list[Path], out_md: Path):
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Motor-memory index report", ""]
    for p in index_paths:
        if not p.exists():
            continue
        mem = MotorMemory.load(p)
        rep_p = p.with_suffix(".report.json")
        rep = json.loads(rep_p.read_text()) if rep_p.exists() else {}
        lines += [f"## `{p.name}` ({p.stat().st_size / 1e6:.1f} MB)", "",
                  f"Sources: {rep.get('sources')}; demos: {rep.get('demos')}; segments: {len(mem)}; "
                  f"demos with zero segments: {len(rep.get('failed', {}))}; build time {rep.get('seconds', 0):.0f} s", "",
                  "| kind | object | segments |", "|---|---|---|"]
        for kind, objs in sorted(mem.summary().items()):
            for o, n in sorted(objs.items(), key=lambda x: -x[1]):
                lines.append(f"| {kind} | {o} | {n} |")
        rel = Counter((s.kind.value, s.relation) for s in mem.segments)
        lines += ["", "Relations: " + ", ".join(f"{k}/{r}: {n}" for (k, r), n in sorted(rel.items(), key=str)), ""]
        issues = Counter()
        for v in rep.get("issues", {}).values():
            for i in v:
                issues[i.split(": ", 1)[-1].split(" (")[0].split(" '")[0]] += 1
        if issues:
            lines += ["Labelling issues (events skipped, by reason):", ""]
            lines += [f"- {k}: {n}" for k, n in issues.most_common(15)]
        if rep.get("failed"):
            lines += ["", "Demos with zero segments:", ""]
            lines += [f"- `{k}`: {v}" for k, v in list(rep["failed"].items())[:15]]
        lines.append("")
    # sanity plots from the goal index
    goal = config.INDEX_DIR / "libero_goal.pkl"
    if goal.exists():
        mem = MotorMemory.load(goal)
        plots = [(SegmentKind.GRASP, "akita_black_bowl"), (SegmentKind.PLACE, "plate"),
                 (SegmentKind.ACTUATE, "wooden_cabinet")]
        lines += ["## Sanity plots (libero_goal)", ""]
        for kind, name in plots:
            f = out_md.parent / f"index_{kind.value}_{name}.png"
            if _plot(mem, kind, name, f):
                lines.append(f"![{kind.value} {name}]({f.name})")
        lines.append("")
    out_md.write_text("\n".join(lines))
    return out_md


if __name__ == "__main__":
    import sys
    paths = [Path(p) for p in sys.argv[1:]] or sorted(config.INDEX_DIR.glob("*.pkl"))
    print(write_report(paths, config.RUNS_DIR / "index_report.md"))
