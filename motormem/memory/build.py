"""Offline index building: replay demos in parallel, segment them, save a MotorMemory."""
from __future__ import annotations

import json
import multiprocessing as mp
import time
import traceback
from pathlib import Path

from motormem import config
from motormem.memory.index import MotorMemory

LIBERO_SUITES = ("libero_goal", "libero_spatial", "libero_object", "libero_10", "libero_90")


def _libero_jobs(suite: str, max_demos: int | None, chunk: int, scene_filter: tuple[str, ...] = ()):
    import h5py
    d = config.LIBERO_HDF5_DIR / suite
    jobs = []
    for fn in sorted(d.glob("*.hdf5")):
        if scene_filter and not fn.name.startswith(scene_filter):
            continue
        with h5py.File(fn, "r") as f:
            keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        if max_demos:
            keys = keys[:max_demos]
        for i in range(0, len(keys), chunk):
            jobs.append(("libero", suite, str(fn), keys[i:i + chunk]))
    return jobs


def _worker(job):
    kind, suite, fn, keys = job
    out = []
    try:
        if kind == "libero":
            from motormem.memory.replay import LiberoReplayer
            from motormem.memory.segmenting import segment_demo
            r = LiberoReplayer(suite, fn)
        else:
            from motormem.memory.robosuite_replay import RobosuiteReplayer as R
            from motormem.memory.segmenting import segment_demo
            r = R(suite, fn)
    except Exception as e:  # env failed to build: every demo of this job fails
        return [(f"{suite}/{Path(fn).stem}/{k}", [], [f"env build failed: {e!r}"]) for k in keys]
    for k in keys:
        try:
            tr = r.replay(k)
            segs, issues = segment_demo(tr)
            out.append((tr.demo_id, segs, issues))
        except Exception as e:
            out.append((f"{suite}/{Path(fn).stem}/{k}", [], [f"replay failed: {e!r} {traceback.format_exc(limit=2)}"]))
    r.close()
    return out


def build_index(sources: list[str], out: Path, max_demos_per_task: int | None = None, workers: int = 8,
                chunk: int = 7, log=print) -> MotorMemory:
    """sources: LIBERO suite names, "libero_90:STUDY,LIVING_ROOM" (scene-prefix filter) or "mimicgen/<task>"."""
    jobs = []
    for src in sources:
        if src.startswith("mimicgen/"):
            import h5py
            task = src.split("/", 1)[1]
            fn = config.MIMICGEN_DIR / f"{task}.hdf5"
            with h5py.File(fn, "r") as f:
                keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
            if max_demos_per_task:
                keys = keys[:max_demos_per_task]
            for i in range(0, len(keys), max(1, chunk // 2)):
                jobs.append(("robosuite", src, str(fn), keys[i:i + max(1, chunk // 2)]))
        else:
            suite, _, filt = src.partition(":")
            jobs += _libero_jobs(suite, max_demos_per_task, chunk, tuple(f for f in filt.split(",") if f))
    t0 = time.time()
    mem = MotorMemory()
    report = {"sources": sources, "demos": 0, "failed": {}, "issues": {}}
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, maxtasksperchild=4) as pool:
        for n, res in enumerate(pool.imap_unordered(_worker, jobs)):
            for demo_id, segs, issues in res:
                report["demos"] += 1
                mem.add_segments(segs)
                if issues:
                    report["issues"][demo_id] = issues
                if not segs:
                    report["failed"][demo_id] = issues or ["no segments"]
            log(f"[build] {n + 1}/{len(jobs)} jobs, {report['demos']} demos, {len(mem)} segments, "
                f"{time.time() - t0:.0f}s")
    out = Path(out)
    mem.save(out)
    report["segments"] = len(mem)
    report["summary"] = mem.summary()
    report["seconds"] = time.time() - t0
    out.with_suffix(".report.json").write_text(json.dumps(report, indent=1, default=str))
    log(f"[build] saved {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s")
    return mem
