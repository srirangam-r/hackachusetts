"""Trace analysis: episodes where no skill ever ran. Prints what the brain saw and did."""
import glob
import json
import sys

for run in sys.argv[1:]:
    for f in sorted(glob.glob(f"runs/{run}/ep_*/result.json")):
        r = json.load(open(f))
        if r.get("success") or r.get("skill_statuses"):
            continue
        try:
            lines = [json.loads(x) for x in open(f.replace("result.json", "trace.jsonl"))]
        except FileNotFoundError:
            lines = []
        tools = [x["tool"] for x in lines]
        obs = [x for x in lines if x["tool"] == "observe"]
        objs = (obs[-1].get("text") or "").split("objects")[-1].replace("\n", " / ")[:200] if obs else "(no observe)"
        print(run, r["spec"]["task_id"], tools, "|", r.get("instruction", "")[:45], "|", (r.get("error") or "")[:60], "|", objs)
