"""Baselines that keep their own runner scripts (GCBF+, LAS).

They are driven as subprocesses under the interpreter they need, on the map
directory ffbench hands them, and their rows are normalised into the common
schema.  Nothing in ``gcbf_baseline/`` or ``learning_adaptive_safety/`` changes.
"""
from __future__ import annotations

import csv
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
from typing import List

from ffbench.maps.targets.f1tenth import write_f1tenth
from ffbench.paths import BASELINES, GENERATED, PY_FASTFUNNELS, PY_GCBF


def _split_dir(req, label: str) -> pathlib.Path:
    """A one-map 'split' directory the gcbf runners can iterate."""
    map_dir = write_f1tenth(req.src, GENERATED / "f1tenth")
    d = pathlib.Path(tempfile.mkdtemp(prefix="ffbench_split_"))
    os.symlink(map_dir, d / map_dir.name)
    return d


def _run(cmd, cwd) -> None:
    print("[external] " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], cwd=str(cwd), check=True)


def run_gcbf(req, mode: str) -> List[dict]:
    py = PY_GCBF if PY_GCBF.exists() else pathlib.Path(sys.executable)
    if "FFBENCH_PY_GCBF" not in os.environ:
        print("[external] FFBENCH_PY_GCBF not set: running GCBF+ under the current interpreter "
              "(needs jax, pyrobosim, gcbfplus -- see ffbench/env/environment-gcbf.yaml)")
    split = _split_dir(req, req.map_label)
    out = pathlib.Path(req.options.get("out_dir") or GENERATED) / "external"
    out.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time())
    if mode == "native":
        out_path = out / f"gcbf_native_{req.map_label}_n{req.n}_{stamp}.jsonl"
        cmd = [py, "-m", "gcbf_baseline.run_gcbf_eval", "--split", f"{req.map_label}:{split}:1",
               "--n-agents", req.n, "--out", out_path]
    else:
        out_path = out / f"gcbf_f110_{req.map_label}_n{req.n}_{stamp}.jsonl"
        cmd = [py, "-m", "gcbf_baseline.run_gcbf_f110", "--split", f"{req.map_label}:{split}:1",
               "--n-agents", req.n, "--dt", req.proto.dt, "--out", out_path]
    if req.proto.course in ("full", "tunnel"):
        # start at the map start, exit test just past the narrow section
        # (waypoints of the map's own centreline, which the gcbf runners use)
        import numpy as np
        src = req.src; cl = src.centerline; nw = src.nearest_wp(*src.narrow_xy)
        w = np.array([sum(src.lateral_extent(cl[i, 0], cl[i, 1], src.tangent(i))) for i in range(len(cl))])
        narrow = (w > 0) & (w < req.proto.narrow_width_factor * src.gap_width_m)
        hi = nw
        while hi < len(cl) - 1 and narrow[hi + 1]:
            hi += 1
        cmd += ["--pre-wp", nw, "--exit-offset", hi - nw, "--post-wp", min(len(cl) - 1, hi + 2) - nw,
                "--max-steps", 8000 if mode == "native" else 20000]
    for k, v in req.options.items():
        if k.startswith("gcbf_") and v is not None:
            cmd += ["--" + k[5:].replace("_", "-"), v]
    trace_dir = None
    if req.options.get("save_traces"):
        trace_dir = out / f"gcbf_{mode}_traces_{stamp}"
        cmd += ["--trace-dir", trace_dir]
    t0 = time.time()
    _run(cmd, BASELINES)
    rows = []
    with open(out_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if trace_dir is not None:
                import numpy as np
                tp = pathlib.Path(trace_dir) / f"{r.get('name', req.src.name)}.npy"
                if tp.exists():
                    r["_trace"] = np.load(tp)[:, :, :2]
                    r["trace_dt"] = float(req.proto.dt if mode == "f1tenth" else 0.03)
            r.setdefault("success", 1.0 if r.get("cleared") else 0.0)
            n_cleared = r.get("n_cleared", r.get("n_completed", 0)) or 0
            r.setdefault("n_completed", n_cleared)
            r.setdefault("safety_rate", round(n_cleared / max(req.n, 1), 3))
            r.setdefault("n_collided", int(bool(r.get("collided"))))
            r.setdefault("collision", bool(r.get("collided")))
            r.setdefault("terminated", r.get("reason", ""))
            r.update({"baseline": "gcbf", "sim": mode, "map": req.map_label, "course": req.proto.course, "n_agents": req.n,
                      "dynamics": req.dynamics if mode == "f1tenth" else "dubins",
                      "target_speed": float("nan"), "seed": req.proto.seed, "trial": 0,
                      "wall_time_s": round(time.time() - t0, 1),
                      "note": "GCBF+ rollout is deterministic per map: one trial"})
            rows.append(r)
    return rows


def run_las(req, mode: str) -> List[dict]:
    if req.map_label not in ("standard_ON", "open_narrow_obs"):
        raise RuntimeError("LAS only runs on its own open_narrow_obs override (map standard_ON)")
    if req.n != 3:
        raise RuntimeError("LAS is a fixed 1-ego + 2-NPC policy: use --num_agents 3")
    py = PY_FASTFUNNELS if PY_FASTFUNNELS.exists() else pathlib.Path(sys.executable)
    out = pathlib.Path(req.options.get("out_dir") or GENERATED) / "external"
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / f"las_{int(time.time())}.csv"
    cmd = [py, "las_sweep.py", "--n-episodes", req.proto.trials, "--no-render", "--output", out_path]
    t0 = time.time()
    _run(cmd, BASELINES)
    rows = []
    with open(out_path) as f:
        for r in csv.DictReader(f):
            row = {k: (float(v) if v.replace(".", "", 1).replace("-", "", 1).isdigit() else v)
                   for k, v in r.items()}
            row.update({"baseline": "las", "sim": mode, "map": req.map_label, "course": req.proto.course, "n_agents": 3,
                        "dynamics": "st", "seed": req.proto.seed, "trial": 0,
                        "wall_time_s": round(time.time() - t0, 1),
                        "note": f"row is already the mean over {req.proto.trials} episodes"})
            rows.append(row)
    return rows
