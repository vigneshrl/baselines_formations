"""Static obstacle polygons for the RVO2 (ORCA) simulator.

RVO2 wants obstacle vertices counter-clockwise; a *bounding* polygon (the
corridor walls seen from inside) must be listed clockwise.  Holes in the
corridor (obstacles clear of the walls) are ordinary obstacles.
"""
from __future__ import annotations

import json
import pathlib
from typing import List, Tuple

from shapely.geometry.polygon import orient

from ffbench.maps.source import MapSource

Poly = List[Tuple[float, float]]


def _ring(coords, clockwise: bool) -> Poly:
    pts = [(float(x), float(y)) for x, y in coords]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    # shapely's orient() gives a CCW exterior and CW interiors; flip on demand
    return pts


def rvo2_polygons(src: MapSource, walls: bool = True, obstacles: bool = True,
                  simplify_m: float = 0.10) -> List[Poly]:
    polys: List[Poly] = []
    if walls:
        corridor = orient(src.corridor(simplify_m), sign=1.0)  # exterior CCW
        ext = _ring(corridor.exterior.coords, clockwise=True)
        polys.append(list(reversed(ext)))                     # bounding -> CW
        for hole in corridor.interiors:                       # holes are CW here
            polys.append(list(reversed(_ring(hole.coords, clockwise=False))))
    elif obstacles:
        # without walls the explicit obstacles are the only geometry ORCA sees
        for o in src.obstacles:
            p = orient(o.polygon(), sign=1.0)
            polys.append(_ring(p.exterior.coords, clockwise=False))
    return polys


def write_rvo2(src: MapSource, out_root, **kw) -> pathlib.Path:
    d = pathlib.Path(out_root) / src.name
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{src.name}_rvo2.json"
    with open(path, "w") as f:
        json.dump({
            "map": src.name, "scale": src.scale,
            "narrow_xy": list(src.narrow_xy), "gap_width_m": src.gap_width_m,
            "centerline": src.centerline.round(4).tolist(),
            "polygons": rvo2_polygons(src, **kw),
        }, f)
    return path
