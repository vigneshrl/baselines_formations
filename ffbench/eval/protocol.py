"""The standard experimental protocol shared by every baseline and backend.

* agents spawn in a rank (abreast, or a column for convoy baselines) at
  ``spawn_up_m`` before the narrow-zone centre, heading along the corridor;
* a straight approach segment is spliced into the centreline through the
  zone (the ORCA / LF trick, generalised to the corridor tangent) because the
  stock ``open_narrow`` centreline runs inside a wall past the pinch;
* the scored zone is ``zone_half_m`` either side of the pinch;
* a trial ends when every agent has cleared the zone, or the plant reports a
  collision, or ``max_steps`` elapse.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ffbench.maps.source import MapSource


@dataclass
class Protocol:
    course: str = "zone"               # zone: spawn before the pinch, score the pinch window
                                       # full: spawn at the track start, score the whole narrow section + start-to-finish
    spawn: str = "zone_entry"          # zone_entry | track_start (forced by course=full)
    narrow_width_factor: float = 1.3   # full course: a waypoint is "narrow" while corridor width < factor * gap
    lateral_gap_full: float = 1.0      # full course: rank spacing at the 9 m wide start line (0.6 m makes cars touch in the bend)
    finish_clearance_m: float = 0.6    # full course: finish = last waypoint with this much wall clearance
    goal_buffer_m: float = 1.5         # an agent has finished once within this distance of the finish line
    formation: str = "auto"            # auto (baseline's own) | abreast | column
    lateral_gap: float = 0.6           # abreast spacing, metres (unscaled)
    column_gap: float = 2.0            # column spacing, metres (unscaled)
    spawn_up_m: float = 7.0            # metres before the pinch (repo/paper convention; inside the zone)
    exit_down_m: float = 20.0          # approach segment continues this far past it
    approach_spacing_m: float = 0.6    # spliced waypoint spacing
    zone_half_m: float = 12.0          # scored zone half-length
    duplicate_lateral_m: float = 8.0   # original waypoints this close to the spliced run are dropped
    dt: float = 0.01                   # f1tenth timestep
    native_dt: float = 0.05            # timestep for native point-mass sims
    max_steps: int = 6000
    trials: int = 20
    seed: int = 42
    target_speed: float = 5.0
    collision_thresh: float = 0.5
    spawn_jitter_m: float = 0.0        # per-seed uniform +-jitter of every spawn pose (m)
    spawn_jitter_rad: float = 0.0      # per-seed uniform +-jitter of the spawn heading

    def scaled(self, k: float) -> "Protocol":
        """Distances in metres follow the map scale."""
        p = Protocol(**self.__dict__)
        for f in ("lateral_gap", "column_gap", "spawn_up_m", "exit_down_m",
                  "approach_spacing_m", "zone_half_m", "collision_thresh", "duplicate_lateral_m",
                  "spawn_jitter_m"):
            setattr(p, f, getattr(self, f) * k)
        p.target_speed = self.target_speed * k
        return p


@dataclass
class Reference:
    xs: np.ndarray                    # extended centreline
    ys: np.ndarray
    idx0: int                         # first spliced waypoint (spawn)
    narrow_idx: int                   # spliced waypoint nearest the pinch
    spawn_poses: np.ndarray           # (n, 3)
    heading: float
    tangent: np.ndarray
    narrow_xy: Tuple[float, float]
    gap_width_m: float
    zone_half_wp: int
    lane_offsets: List[float]
    track_closed: bool = False
    goal_xy: Optional[Tuple[float, float]] = None
    arclength: np.ndarray = field(default_factory=lambda: np.zeros(0))
    zone_entry_idx: Optional[int] = None      # explicit zone bounds (full course); else narrow_idx +- zone_half_wp
    zone_exit_idx: Optional[int] = None
    course: str = "zone"
    goal_buffer_wp: int = 3                   # waypoints before the last one that count as "finished"

    @property
    def n(self) -> int:
        return len(self.spawn_poses)


def _rank_shift(src: MapSource, base: np.ndarray, t: np.ndarray, nrm: np.ndarray, n: int,
                proto: Protocol, formation: str, search_m: float = 3.0, step_m: float = 0.1,
                ahead_m: float = 6.0, min_clear_m: float = 0.5) -> float:
    """Lateral shift of the rank centre that maximises clearance to walls/obstacles.

    The rank and a strip ``ahead_m`` in front of it are sampled; the shift with
    the largest minimum clearance wins (ties -> smallest shift).  Keeps the
    spawn from sitting beside a wall-hugging obstacle at the funnel mouth.
    """
    from shapely.geometry import Point
    if formation == "column":
        return 0.0                      # the column sits on the reference path
    poly = src.corridor()
    half = (n - 1) / 2.0
    pts_local = [(0.0, (i - half) * proto.lateral_gap) for i in range(n)]
    def clearance(shift: float) -> float:
        clear = float("inf")
        for a, l in pts_local:
            for d in np.arange(0.0, ahead_m + 1e-9, 1.0):
                p = base + t * (a + d) + nrm * (l + shift)
                pt = Point(p[0], p[1])
                clear = min(clear, poly.boundary.distance(pt) if poly.contains(pt) else -1.0)
        return clear

    # smallest shift that gives every car `min_clear_m` of room; the rank must
    # stay centred on its lanes, so a shift is a last resort, not an optimisation
    if clearance(0.0) >= min_clear_m:
        return 0.0
    best_shift, best_clear = 0.0, clearance(0.0)
    for mag in np.arange(step_m, search_m + 1e-9, step_m):
        for shift in (-mag, mag):
            c = clearance(shift)
            if c >= min_clear_m:
                return float(shift)
            if c > best_clear:
                best_shift, best_clear = float(shift), c
    return best_shift


def _walk_back(xs, ys, idx0: int, dists) -> np.ndarray:
    """Poses (x, y, heading) at path distances ``dists`` behind waypoint ``idx0``."""
    pts = np.column_stack([xs, ys]).astype(float)
    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    out = []
    for d in dists:
        i, rem = idx0, float(d)
        while i > 0 and rem > seg[i - 1]:
            rem -= seg[i - 1]
            i -= 1
        if i == 0:
            p, tang = pts[0], pts[1] - pts[0]
        elif rem < 1e-9 and i + 1 < len(pts):
            p, tang = pts[i], pts[i + 1] - pts[i]          # exactly on a waypoint: face forward
        else:
            tang = pts[i] - pts[i - 1]
            p = pts[i] - tang / max(seg[i - 1], 1e-9) * rem
        h = math.atan2(tang[1], tang[0])
        out.append([p[0], p[1], h])
    return np.asarray(out, dtype=np.float32)


def _is_closed(xs, ys, thresh=5.0) -> bool:
    return math.hypot(float(xs[-1] - xs[0]), float(ys[-1] - ys[0])) < thresh


def splice_approach(xs: np.ndarray, ys: np.ndarray, src: MapSource, proto: Protocol,
                    extra_up_m: float = 0.0):
    """Insert a straight run through the pinch along the corridor tangent."""
    nx, ny = src.narrow_xy
    t = src.tangent(src.nearest_wp(nx, ny))
    up = max(proto.spawn_up_m + extra_up_m, proto.zone_half_m + 2.0 * proto.approach_spacing_m)
    n_app = int(round((up + proto.exit_down_m) / proto.approach_spacing_m)) + 1
    s = np.linspace(-up, proto.exit_down_m, n_app)
    app = np.array([nx, ny])[None, :] + s[:, None] * t[None, :]
    # Drop the original waypoints that run alongside the spliced segment (the
    # same corridor, plus the stretch that sits inside the wall).  Otherwise a
    # denser original centreline sits closer to the cars than the approach
    # segment and the zone bookkeeping latches onto the wrong index space.
    rel = np.column_stack([xs - nx, ys - ny]).astype(float)
    along = rel @ t
    lateral = np.abs(rel @ np.array([-t[1], t[0]]))
    keep = ~((along >= -up - proto.approach_spacing_m) & (along <= proto.exit_down_m + proto.approach_spacing_m)
             & (lateral < proto.duplicate_lateral_m))
    if keep.all() or not keep.any():
        ins_at = int(np.argmin(np.hypot(xs - nx, ys - ny)))
    else:
        ins_at = int(np.argmax(~keep))          # first dropped waypoint
    xs_pre, ys_pre = xs[:ins_at][keep[:ins_at]], ys[:ins_at][keep[:ins_at]]
    xs_post, ys_post = xs[ins_at:][keep[ins_at:]], ys[ins_at:][keep[ins_at:]]
    ins_at = len(xs_pre)
    xs_ext = np.concatenate([xs_pre, app[:, 0].astype(np.float32), xs_post])
    ys_ext = np.concatenate([ys_pre, app[:, 1].astype(np.float32), ys_post])
    spacing = float((up + proto.exit_down_m) / (n_app - 1))
    spawn_idx = ins_at + int(round((up - proto.spawn_up_m) / spacing))
    narrow_idx = ins_at + int(round(up / spacing))
    return xs_ext, ys_ext, ins_at, spawn_idx, narrow_idx, t, spacing


def build_reference(src: MapSource, xs: np.ndarray, ys: np.ndarray, n: int,
                    proto: Protocol, formation: str = "abreast",
                    seed: Optional[int] = None) -> Reference:
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    closed = _is_closed(xs, ys)
    if proto.course == "full":
        return _build_full_course(src, xs, ys, n, proto, formation, seed)
    if proto.spawn == "zone_entry":
        xs_e, ys_e, ins_at, spawn_idx, narrow_idx, t, spacing = splice_approach(xs, ys, src, proto)
        heading = math.atan2(float(t[1]), float(t[0]))
        base = np.array([xs_e[spawn_idx], ys_e[spawn_idx]], dtype=float)
        idx0 = spawn_idx
    else:  # track start
        xs_e, ys_e = xs, ys
        t = src.tangent(0) if len(src.centerline) > 1 else np.array([1.0, 0.0])
        d = np.array([xs[min(5, len(xs) - 1)] - xs[0], ys[min(5, len(ys) - 1)] - ys[0]], dtype=float)
        if np.hypot(*d) > 1e-6:
            t = d / np.hypot(*d)
        heading = math.atan2(float(t[1]), float(t[0]))
        base = np.array([xs[0], ys[0]], dtype=float) + t * 1.0
        idx0 = 0
        narrow_idx = int(np.argmin(np.hypot(xs - src.narrow_xy[0], ys - src.narrow_xy[1])))
        seg = np.hypot(np.diff(xs), np.diff(ys))
        spacing = float(np.mean(seg[max(0, narrow_idx - 20): narrow_idx + 20])) if len(seg) else 1.0
    nrm = np.array([-t[1], t[0]], dtype=float)
    half = (n - 1) / 2.0
    base = base + nrm * _rank_shift(src, base, t, nrm, n, proto, formation)
    poses = np.zeros((n, 3), dtype=np.float32)
    offsets = []
    if formation == "column":
        # followers sit *on the reference path* behind the leader, so a column
        # follows the corridor's bend upstream instead of a straight tangent
        # that would leave the corridor
        col = _walk_back(xs_e, ys_e, idx0, [i * proto.column_gap for i in range(n)])
        for i in range(n):
            poses[i] = col[i]
            offsets.append(0.0)
    else:
        for i in range(n):
            off = (i - half) * proto.lateral_gap
            p = base + nrm * off
            offsets.append(float(off))
            poses[i] = [p[0], p[1], heading]
    if seed is not None and (proto.spawn_jitter_m > 0 or proto.spawn_jitter_rad > 0):
        # along-track + heading only: lateral jitter would eat into the rank gap
        rng = np.random.default_rng(seed)
        along = rng.uniform(-proto.spawn_jitter_m, proto.spawn_jitter_m, size=n)
        poses[:, 0] += (along * t[0]).astype(np.float32)
        poses[:, 1] += (along * t[1]).astype(np.float32)
        poses[:, 2] += rng.uniform(-proto.spawn_jitter_rad, proto.spawn_jitter_rad, size=n).astype(np.float32)
    seg = np.hypot(np.diff(xs_e), np.diff(ys_e))
    arclength = np.concatenate([[0.0], np.cumsum(seg)])
    zone_half_wp = max(1, int(round(proto.zone_half_m / spacing)))
    goal = (float(xs_e[min(len(xs_e) - 1, narrow_idx + zone_half_wp)]),
            float(ys_e[min(len(ys_e) - 1, narrow_idx + zone_half_wp)]))
    return Reference(xs_e, ys_e, int(idx0), int(narrow_idx), poses, heading, t,
                     src.narrow_xy, src.gap_width_m, zone_half_wp, offsets, closed,
                     goal, arclength)


def _walk_forward(xs, ys, s_target: float):
    """Index of the first waypoint at or past arclength ``s_target``."""
    seg = np.hypot(np.diff(xs), np.diff(ys))
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    return int(min(len(xs) - 1, np.searchsorted(arc, s_target)))


def _build_full_course(src: MapSource, xs, ys, n: int, proto: Protocol, formation: str,
                       seed: Optional[int]) -> Reference:
    """Start line -> finish line.  The pinch splice is kept (the stock
    centreline runs inside a wall there); the scored zone is the whole
    narrow section, i.e. every consecutive waypoint around the pinch whose
    corridor width is below ``narrow_width_factor`` x gap."""
    xs_e, ys_e, ins_at, spawn_idx, narrow_idx, t_pinch, spacing = splice_approach(xs, ys, src, proto)
    # cut the reference at the finish line: the gym's resampled centreline loops
    # back to the start, and a car spawned just behind the start line would
    # otherwise be "nearest" to a waypoint past the finish
    gx, gy = float(src.centerline[-1, 0]), float(src.centerline[-1, 1])
    fin = int(np.argmin(np.hypot(xs_e - gx, ys_e - gy)))
    # the map's last waypoints run past the walls: the finish line is the last
    # waypoint that still has a car's worth of clearance inside the corridor
    from shapely.geometry import Point
    poly = src.corridor()
    while fin > 0:
        pt = Point(float(xs_e[fin]), float(ys_e[fin]))
        if poly.contains(pt) and poly.boundary.distance(pt) >= proto.finish_clearance_m:
            break
        fin -= 1
    xs_e, ys_e = xs_e[: fin + 1], ys_e[: fin + 1]
    nwp = len(xs_e)
    # local tangents
    tx = np.gradient(xs_e.astype(float)); ty = np.gradient(ys_e.astype(float))
    nrm_ = np.maximum(np.hypot(tx, ty), 1e-9); tx, ty = tx / nrm_, ty / nrm_
    # corridor width at every waypoint (0 outside the corridor)
    widths = np.zeros(nwp)
    for i in range(nwp):
        l, r = src.lateral_extent(float(xs_e[i]), float(ys_e[i]), (tx[i], ty[i]))
        widths[i] = l + r
    narrow = (widths > 0) & (widths < proto.narrow_width_factor * src.gap_width_m)
    lo = hi = narrow_idx
    while lo > 0 and narrow[lo - 1]:
        lo -= 1
    while hi < nwp - 1 and narrow[hi + 1]:
        hi += 1
    # spawn: rank (or column) just past the start line, facing along the track
    lead_s = 1.0 + ((n - 1) * proto.column_gap if formation == "column" else 0.0)
    idx0 = _walk_forward(xs_e, ys_e, lead_s)
    t = np.array([tx[idx0], ty[idx0]]); nrm = np.array([-t[1], t[0]])
    heading = math.atan2(float(t[1]), float(t[0]))
    base = np.array([xs_e[idx0], ys_e[idx0]], dtype=float)
    half = (n - 1) / 2.0
    base = base + nrm * _rank_shift(src, base, t, nrm, n, proto, formation)
    poses = np.zeros((n, 3), dtype=np.float32)
    offsets = []
    if formation == "column":
        col = _walk_back(xs_e, ys_e, idx0, [i * proto.column_gap for i in range(n)])
        for i in range(n):
            poses[i] = col[i]; offsets.append(0.0)
    else:
        gap = proto.lateral_gap_full
        for i in range(n):
            off = (i - half) * gap
            p = base + nrm * off
            offsets.append(float(off)); poses[i] = [p[0], p[1], heading]
    if seed is not None and (proto.spawn_jitter_m > 0 or proto.spawn_jitter_rad > 0):
        rng = np.random.default_rng(seed)
        along = rng.uniform(-proto.spawn_jitter_m, proto.spawn_jitter_m, size=n)
        poses[:, 0] += (along * t[0]).astype(np.float32); poses[:, 1] += (along * t[1]).astype(np.float32)
        poses[:, 2] += rng.uniform(-proto.spawn_jitter_rad, proto.spawn_jitter_rad, size=n).astype(np.float32)
    seg = np.hypot(np.diff(xs_e), np.diff(ys_e))
    arclength = np.concatenate([[0.0], np.cumsum(seg)])
    # finish line = the end of the map's own centreline (the gym's resampled
    # centreline may loop back to the start, so never use its last waypoint)
    goal = (float(xs_e[-1]), float(ys_e[-1]))
    end_spacing = float(np.mean(seg[-40:])) if len(seg) >= 40 else float(np.mean(seg))
    goal_buffer_wp = max(1, int(round(proto.goal_buffer_m / max(end_spacing, 1e-6))))
    return Reference(xs_e, ys_e, int(idx0), int(narrow_idx), poses, heading, t_pinch,
                     src.narrow_xy, src.gap_width_m, int(max(hi - narrow_idx, narrow_idx - lo, 1)),
                     offsets, False, goal, arclength,
                     zone_entry_idx=int(lo), zone_exit_idx=int(hi), course="full",
                     goal_buffer_wp=goal_buffer_wp)
