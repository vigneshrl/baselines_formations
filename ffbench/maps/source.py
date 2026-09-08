"""One map as the single source of truth for every backend.

The FastFunnels map-directory format (``<name>_map.pgm/.yaml``,
``<name>_centerline.csv``, ``<name>_obs_pos.yaml``) is loaded once into a
:class:`MapSource`; the f1tenth, RVO2, Gazebo/ROS and GCBF+ targets are all
derived from it, so a baseline never sees a different corridor than the others.
"""
from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

import numpy as np
import yaml
from PIL import Image
from PIL.Image import Transpose
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon

OCC_THRESH = 128          # f1tenth_gym: pixel <= 128 is occupied
F1TENTH_RADIUS_M = 0.29   # circumscribed radius of an f1tenth car


@dataclass
class Obstacle:
    kind: str                     # rect | disc | tri
    center: Tuple[float, float]
    size: Tuple[float, float]     # (w, h) metres; a disc is (2r, 2r)
    angle_deg: float = 0.0

    @classmethod
    def from_yaml(cls, d: dict) -> "Obstacle":
        """Accept both obstacle schemas found in the repo.

        New (``baselines/maps``): ``kind, center_m, size_m, angle_deg``.
        Old (gym maps / ORCA loader): ``type, center_m|x_m,y_m,
        width_m/height_m|radius_m, angle_deg|yaw_deg``.
        """
        kind = str(d.get("kind") or d.get("type") or "rect").lower()
        if kind in ("circle", "cyl", "cylinder", "pillar"):
            kind = "disc"
        if "center_m" in d:
            cx, cy = d["center_m"][0], d["center_m"][1]
        elif "center" in d:
            cx, cy = d["center"][0], d["center"][1]
        else:
            cx = d.get("x_m", d.get("x", 0.0))
            cy = d.get("y_m", d.get("y", 0.0))
        if "size_m" in d:
            w, h = d["size_m"][0], d["size_m"][1]
        elif "radius_m" in d or "r_m" in d:
            r = float(d.get("radius_m", d.get("r_m")))
            w = h = 2.0 * r
        else:
            w = float(d.get("width_m", d.get("w_m", 0.5)))
            h = float(d.get("height_m", d.get("h_m", w)))
        ang = d.get("angle_deg", d.get("yaw_deg"))
        if ang is None and "angle" in d:
            ang = math.degrees(float(d["angle"]))
        return cls(kind, (float(cx), float(cy)), (float(w), float(h)),
                   float(ang or 0.0))

    def to_yaml(self) -> dict:
        """Emit a record readable by both schemas."""
        w, h = self.size
        out = {
            "kind": self.kind, "type": "circle" if self.kind == "disc" else self.kind,
            "center_m": [round(self.center[0], 4), round(self.center[1], 4)],
            "size_m": [round(w, 4), round(h, 4)],
            "angle_deg": round(self.angle_deg, 3),
            "width_m": round(w, 4), "height_m": round(h, 4),
        }
        if self.kind == "disc":
            out["radius_m"] = round(0.5 * max(w, h), 4)
        return out

    def polygon(self, n_disc: int = 16) -> Polygon:
        w, h = self.size
        cx, cy = self.center
        if self.kind == "disc":
            r = 0.5 * max(w, h)
            return Point(cx, cy).buffer(r, resolution=max(4, n_disc // 4))
        if self.kind == "tri":
            base = Polygon([(-w / 2, -h / 2), (w / 2, -h / 2), (0.0, h / 2)])
        else:
            base = Polygon([(-w / 2, -h / 2), (w / 2, -h / 2),
                            (w / 2, h / 2), (-w / 2, h / 2)])
        base = affinity.rotate(base, self.angle_deg, origin=(0, 0))
        return affinity.translate(base, cx, cy)

    def scaled(self, k: float) -> "Obstacle":
        return Obstacle(self.kind, (self.center[0] * k, self.center[1] * k),
                        (self.size[0] * k, self.size[1] * k), self.angle_deg)


@dataclass
class MapSource:
    name: str
    map_dir: pathlib.Path
    image_path: pathlib.Path
    resolution: float
    origin: Tuple[float, float]
    occupancy: np.ndarray            # bool (H, W); row index grows with +y
    centerline: np.ndarray           # (N, 2) metres
    centerline_widths: np.ndarray    # (N, 2) right/left track width columns
    obstacles: List[Obstacle]
    narrow_xy: Tuple[float, float]
    gap_width_m: float
    meta: dict = field(default_factory=dict)
    scale: float = 1.0               # metres here / metres in the source map
    _corridor: Optional[Polygon] = None
    _corridor_key: Optional[float] = None

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, map_dir, name: Optional[str] = None) -> "MapSource":
        map_dir = pathlib.Path(map_dir).resolve()
        name = name or map_dir.name
        with open(map_dir / f"{name}_map.yaml") as f:
            spec = yaml.safe_load(f)
        image_path = map_dir / spec["image"]
        img = Image.open(image_path).transpose(Transpose.FLIP_TOP_BOTTOM)
        occ = np.asarray(img).astype(np.uint8) <= OCC_THRESH
        raw = np.loadtxt(map_dir / f"{name}_centerline.csv", delimiter=",", comments="#")
        raw = np.atleast_2d(raw)
        cl = raw[:, :2].astype(float)
        widths = raw[:, 2:4].astype(float) if raw.shape[1] >= 4 else np.ones((len(cl), 2))
        meta_path = map_dir / f"{name}_obs_pos.yaml"
        meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = yaml.safe_load(f) or {}
        obstacles = [Obstacle.from_yaml(o) for o in
                     list(meta.get("obstacles") or []) + list(meta.get("toll_pillars") or [])]
        regions = meta.get("narrow_regions") or []
        if regions:
            narrow = (float(regions[0]["center_m"][0]), float(regions[0]["center_m"][1]))
            gap = float(regions[0].get("gap_width_m", 1.0))
        else:
            mid = cl[len(cl) // 2]
            narrow, gap = (float(mid[0]), float(mid[1])), 1.0
        return cls(name, map_dir, image_path, float(spec["resolution"]),
                   (float(spec["origin"][0]), float(spec["origin"][1])),
                   occ, cl, widths, obstacles, narrow, gap, meta)

    # ------------------------------------------------------------- geometry
    @property
    def height_px(self) -> int:
        return int(self.occupancy.shape[0])

    @property
    def width_px(self) -> int:
        return int(self.occupancy.shape[1])

    def world_to_px(self, x: float, y: float) -> Tuple[int, int]:
        return (int((x - self.origin[0]) / self.resolution),
                int((y - self.origin[1]) / self.resolution))

    def is_free(self, x: float, y: float) -> bool:
        c, r = self.world_to_px(x, y)
        if r < 0 or c < 0 or r >= self.height_px or c >= self.width_px:
            return False
        return not bool(self.occupancy[r, c])

    def corridor(self, simplify_m: float = 0.10) -> Polygon:
        """Walkable corridor polygon (obstacles clear of walls become holes)."""
        if self._corridor is not None and self._corridor_key == simplify_m:
            return self._corridor
        from gcbf_baseline.corridor_geom import _trace_free_component
        seed = self.world_to_px(*self.narrow_xy)
        poly = _trace_free_component(self.occupancy, self.resolution, self.origin,
                                     seed, simplify_m)
        self._corridor, self._corridor_key = poly, simplify_m
        return poly

    def corridor_geom(self, simplify_m: float = 0.10):
        """The GCBF+ baseline's :class:`CorridorGeom` view of this map."""
        from gcbf_baseline.corridor_geom import CorridorGeom
        seg = np.hypot(np.diff(self.centerline[:, 0]), np.diff(self.centerline[:, 1]))
        s = np.concatenate([[0.0], np.cumsum(seg)])
        return CorridorGeom(name=self.name, walkable=self.corridor(simplify_m),
                            centerline=self.centerline, arclength=s,
                            narrow_xy=self.narrow_xy, gap_width_m=self.gap_width_m,
                            resolution=self.resolution)

    def nearest_wp(self, x: float, y: float) -> int:
        return int(np.argmin(np.hypot(self.centerline[:, 0] - x, self.centerline[:, 1] - y)))

    def tangent(self, wp: int) -> np.ndarray:
        n = len(self.centerline)
        i0, i1 = max(0, wp - 1), min(n - 1, wp + 1)
        t = self.centerline[i1] - self.centerline[i0]
        nrm = float(np.hypot(t[0], t[1]))
        return t / nrm if nrm > 1e-9 else np.array([1.0, 0.0])

    def lateral_extent(self, x: float, y: float, tangent, max_m: float = 30.0,
                       simplify_m: float = 0.10) -> Tuple[float, float]:
        """Free distance to the wall on the left and right of (x, y).

        Left is the +normal side of ``tangent``.  Returns (0, 0) when the point
        is not inside the corridor.
        """
        poly = self.corridor(simplify_m)
        p = np.array([x, y], dtype=float)
        t = np.asarray(tangent, dtype=float)
        t = t / max(float(np.hypot(t[0], t[1])), 1e-9)
        n = np.array([-t[1], t[0]])
        if not poly.contains(Point(x, y)):
            return 0.0, 0.0
        line = LineString([tuple(p - n * max_m), tuple(p + n * max_m)])
        inter = line.intersection(poly)
        pieces = list(getattr(inter, "geoms", [inter]))
        best = None
        for g in pieces:
            if g.is_empty or not isinstance(g, LineString):
                continue
            if g.distance(Point(x, y)) < 1e-6:
                best = g
                break
        if best is None:
            return 0.0, 0.0
        proj = [float(np.dot(np.array(c) - p, n)) for c in best.coords]
        return max(0.0, max(proj)), max(0.0, -min(proj))

    # ------------------------------------------------------------- rescale
    def rescaled(self, k: float, name: Optional[str] = None) -> "MapSource":
        """Same pixels, everything in metres multiplied by ``k``.

        Used to put a differently sized robot (TurtleBot3 for DEFORM, a
        GCBF+ disc) in a corridor that keeps the same ratio to its body as the
        f1tenth car has to the original.
        """
        if abs(k - 1.0) < 1e-12:
            return self
        tag = f"{k:.3g}".replace(".", "p")
        new_name = name or f"{self.name}_s{tag}"
        meta = dict(self.meta)
        meta["scale_from_source"] = float(self.scale * k)
        meta["source_map"] = self.name
        return replace(
            self, name=new_name,
            resolution=self.resolution * k,
            origin=(self.origin[0] * k, self.origin[1] * k),
            centerline=self.centerline * k,
            centerline_widths=self.centerline_widths * k,
            obstacles=[o.scaled(k) for o in self.obstacles],
            narrow_xy=(self.narrow_xy[0] * k, self.narrow_xy[1] * k),
            gap_width_m=self.gap_width_m * k,
            meta=meta, scale=self.scale * k, _corridor=None, _corridor_key=None,
        )

    def scale_for_robot(self, robot_radius_m: float) -> float:
        """Factor that gives ``robot_radius_m`` the f1tenth car's corridor ratio."""
        return float(robot_radius_m) / F1TENTH_RADIUS_M

    def obs_pos_dict(self) -> dict:
        """The obstacle/narrow-zone sidecar, both schemas, current scale."""
        out = dict(self.meta)
        out["obstacles"] = [o.to_yaml() for o in self.obstacles]
        out["toll_pillars"] = []
        out["narrow_regions"] = [{
            "center_m": [round(self.narrow_xy[0], 4), round(self.narrow_xy[1], 4)],
            "gap_width_m": round(self.gap_width_m, 4),
        }]
        return out
