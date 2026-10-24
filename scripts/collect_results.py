"""Copy eval summaries (small) from runs/ into results/ so they are committed, and print one overview table.

    collect_results.py goal_panda goal_ur5e pro_swap_panda ...
"""
import glob
import json
import shutil
import sys
from pathlib import Path

from motormem import config

OUT = config.MAIN_CHECKOUT / "results"


def stats(name: str) -> dict:
    eps = [json.load(open(f)) for f in sorted(glob.glob(str(config.RUNS_DIR / name / "ep_*" / "result.json")))]
    n = len(eps)
    ok = sum(bool(e.get("success")) for e in eps)
    calls = [e.get("tool_calls", 0) for e in eps]
    wall = [e.get("brain_wall_s", 0.0) or 0.0 for e in eps]
    fb = sum(e.get("fallbacks", 0) for e in eps)
    segs = sum(len(e.get("segment_ids") or []) for e in eps)
    return {"name": name, "episodes": n, "success": ok, "rate": ok / n if n else 0.0,
            "mean_tool_calls": sum(calls) / n if n else 0, "mean_brain_s": sum(wall) / n if n else 0,
            "index_motions": segs, "primitive_motions": fb}


def main(names):
    OUT.mkdir(exist_ok=True)
    rows = []
    for name in names:
        src = config.RUNS_DIR / name
        if not (src / "summary.md").exists():
            print(f"skip {name}: no summary")
            continue
        d = OUT / name
        d.mkdir(exist_ok=True)
        for f in ("summary.md", "summary.json"):
            if (src / f).exists():
                shutil.copy(src / f, d / f)
        s = stats(name)
        (d / "stats.json").write_text(json.dumps(s, indent=2))
        rows.append(s)
    lines = ["| run | success | mean tool calls | mean brain time (s) | motions from demos / from primitive |",
             "|---|---|---|---|---|"]
    for s in rows:
        lines.append(f"| {s['name']} | {s['success']}/{s['episodes']} ({100 * s['rate']:.0f}%) | "
                     f"{s['mean_tool_calls']:.1f} | {s['mean_brain_s']:.0f} | {s['index_motions']} / {s['primitive_motions']} |")
    table = "\n".join(lines)
    (OUT / "overview.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main(sys.argv[1:])
