"""JSONL rows -> aggregated CSV + printed table (folds sweep_runner/print_table)."""
from __future__ import annotations

import csv
import json
import math
import pathlib
from collections import defaultdict
from typing import Dict, List

from ffbench.eval.metrics import METRIC_KEYS

GROUP_KEYS = ("baseline", "sim", "map", "course", "n_agents", "dynamics", "target_speed")
NUMERIC = ("avg_speed_mps", "success", "time_to_goal_s", "flow_rate", "deformability",
           "n_completed", "n_collided", "safety_rate", "collision", "spread_at_entry_m",
           "spread_at_narrow_m", "finished", "t_course_s", "v_course_mps", "steps", "sim_time_s", "wall_time_s")


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def write_jsonl(rows: List[dict], path) -> pathlib.Path:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    return path


def append_jsonl(row: dict, path) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def read_jsonl(path) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def aggregate(rows: List[dict]) -> List[dict]:
    """Mean per condition, NaN-ignoring; inf survives only if every trial was inf."""
    groups: Dict[tuple, List[dict]] = defaultdict(list)

    def _key(v):
        # NaN != NaN would put every row in its own group (external baselines
        # report target_speed as NaN)
        if isinstance(v, float) and math.isnan(v):
            return "nan"
        return v

    for r in rows:
        groups[tuple(_key(r.get(k)) for k in GROUP_KEYS)].append(r)
    out = []
    for key, rs in groups.items():
        agg = dict(zip(GROUP_KEYS, key))
        agg["trials"] = len(rs)
        for m in NUMERIC:
            vals = [_f(r.get(m)) for r in rs]
            finite = [v for v in vals if math.isfinite(v)]
            if finite:
                agg[m] = round(sum(finite) / len(finite), 4)
            elif any(math.isinf(v) for v in vals):
                agg[m] = float("inf")
            else:
                agg[m] = float("nan")
        out.append(agg)
    out.sort(key=lambda a: tuple(str(a[k]) for k in GROUP_KEYS))
    return out


def write_csv(aggs: List[dict], path) -> pathlib.Path:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(GROUP_KEYS) + ["trials"] + list(NUMERIC)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for a in aggs:
            w.writerow(a)
    return path


def format_table(aggs: List[dict]) -> str:
    cols = [("baseline", 16), ("sim", 8), ("map", 12), ("course", 5), ("n_agents", 3), ("dynamics", 5),
            ("target_speed", 6), ("trials", 6), ("avg_speed_mps", 7), ("success", 6),
            ("time_to_goal_s", 8), ("flow_rate", 7), ("deformability", 7),
            ("safety_rate", 6), ("n_collided", 6), ("finished", 7), ("t_course_s", 8), ("v_course_mps", 7)]
    head = {"baseline": "baseline", "sim": "sim", "map": "map", "n_agents": "N",
            "dynamics": "dyn", "target_speed": "v_tgt", "trials": "trials",
            "avg_speed_mps": "V_bar", "success": "eSR", "time_to_goal_s": "T_zone",
            "flow_rate": "flow", "deformability": "deform", "safety_rate": "aSR",
            "n_collided": "n_coll", "course": "crs", "finished": "finish", "t_course_s": "T_course",
            "v_course_mps": "V_course"}

    def fmt(v):
        if isinstance(v, float):
            if math.isnan(v):
                return "nan"
            if math.isinf(v):
                return "inf"
            return f"{v:.3f}"
        return str(v)

    lines = ["".join(head[c].rjust(w + 1) for c, w in cols)]
    lines.append("-" * len(lines[0]))
    for a in aggs:
        lines.append("".join(fmt(a.get(c, ""))[: w + 1].rjust(w + 1) for c, w in cols))
    return "\n".join(lines)
