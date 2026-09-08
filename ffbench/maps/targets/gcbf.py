"""Oriented wall rectangles (the only obstacle GCBF+ understands)."""
from __future__ import annotations

import json
import pathlib

from ffbench.maps.source import MapSource


def write_gcbf(src: MapSource, out_root, thickness_m: float = 0.6,
               reach_m: float = 12.0) -> pathlib.Path:
    from gcbf_baseline.corridor_geom import wall_rects
    geom = src.corridor_geom()
    window = geom.window()
    rects = wall_rects(geom, window, thickness_m=thickness_m, reach_m=reach_m)
    d = pathlib.Path(out_root) / src.name
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{src.name}_gcbf_rects.json"
    with open(path, "w") as f:
        json.dump({"map": src.name, "scale": src.scale,
                   "window": window.__dict__,
                   "rects_cx_cy_w_h_theta": rects.round(4).tolist()}, f)
    return path
