"""The f1tenth-backend trial loop, shared by every baseline that runs on it."""
from __future__ import annotations

import math
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol as _Proto

import numpy as np

from ffbench.eval.metrics import ZoneMetrics
from ffbench.eval.protocol import Protocol, Reference, build_reference
from ffbench.maps.source import MapSource
from ffbench.models.f1tenth import F1TenthSim


@dataclass
class TrialContext:
    src: MapSource
    map_label: str
    map_dir: pathlib.Path
    ref: Reference
    proto: Protocol
    n: int
    dynamics: str
    speed: float
    seed: int
    dt: float
    options: Dict[str, Any] = field(default_factory=dict)


class Controller(_Proto):
    """One baseline's control law on the f1tenth plant."""

    def reset(self, ctx: TrialContext, obs: dict) -> None: ...

    def act(self, obs: dict, step: int) -> np.ndarray:
        """(n, 2) of [steering_angle, speed]."""
        ...


class SimPool:
    """Reuse one gym env per (map, n, dynamics) -- map loading is the slow part."""

    def __init__(self) -> None:
        self._sims: Dict[tuple, F1TenthSim] = {}

    def get(self, map_dir, n: int, dynamics: str, dt: float, render: bool) -> F1TenthSim:
        key = (str(map_dir), n, dynamics, dt, render)
        if key not in self._sims:
            self._sims[key] = F1TenthSim(map_dir, n, dynamics, dt, render)
        return self._sims[key]

    def close(self) -> None:
        for s in self._sims.values():
            s.close()
        self._sims.clear()


def run_f1tenth_trial(baseline: str, controller: Controller, src: MapSource, map_label: str,
                      map_dir, proto: Protocol, n: int, dynamics: str, speed: float,
                      seed: int, trial: int, formation: str = "abreast",
                      control_hz: float = 100.0, render: bool = False,
                      pool: Optional[SimPool] = None, options: Optional[dict] = None,
                      trace: bool = False, record: Optional[str] = None,
                      record_skip: int = 5) -> dict:
    """``record``: path of an .mp4 to write from the gym renderer (rgb_array)."""
    pool = pool or SimPool()
    sim = pool.get(map_dir, n, dynamics, proto.dt, "rgb_array" if record else render)
    writer = None
    if record:
        import cv2
        fps = max(1, int(round(1.0 / (proto.dt * record_skip))))
        pathlib.Path(record).parent.mkdir(parents=True, exist_ok=True)

        def _write(frame):
            nonlocal writer
            if frame is None:
                return
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(record, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
    xs, ys = sim.centerline()
    ref = build_reference(src, xs, ys, n, proto, formation, seed=seed)
    np.random.seed(seed)
    obs = sim.reset(ref.spawn_poses)
    metrics = ZoneMetrics(ref.xs, ref.ys, dt=proto.dt, narrow_center_xy=ref.narrow_xy,
                          gap_width_m=ref.gap_width_m, zone_half_width=ref.zone_half_wp,
                          collision_thresh=proto.collision_thresh, goal_buffer=ref.goal_buffer_wp,
                          goal_xy=(ref.goal_xy if proto.course in ("full", "tunnel") else None),   # zone course: never stop before the zone is cleared
                          zone_entry_idx=ref.zone_entry_idx, zone_exit_idx=ref.zone_exit_idx,
                          finish_line_m=(proto.goal_buffer_m if proto.course in ("full", "tunnel") else None))
    ctx = TrialContext(src, map_label, pathlib.Path(map_dir), ref, proto, n, dynamics,
                       speed, seed, proto.dt, dict(options or {}))
    controller.reset(ctx, obs)
    if record:
        _write(sim.frame())
    hold = max(1, int(round(1.0 / (control_hz * proto.dt))))
    actions = np.zeros((n, 2), dtype=np.float32)
    reason = "max_steps"
    t0 = time.time()
    steps = 0
    gym_collision = False
    positions: List[np.ndarray] = []
    for step in range(proto.max_steps):
        if step % hold == 0:
            actions = np.asarray(controller.act(obs, step), dtype=np.float32).reshape(n, 2)
        obs, done, info = sim.step(actions)
        metrics.step(obs)
        steps = step + 1
        if trace:
            positions.append(np.column_stack([obs["poses_x"], obs["poses_y"]]).copy())
        if record and step % record_skip == 0:
            _write(sim.frame())
        if metrics.all_zone_cleared and proto.course != "full":
            reason = "zone_cleared"
            break
        if metrics.all_done:
            reason = "goal"
            break
        if done:
            gym_collision = bool(np.any(obs.get("collisions", np.zeros(n))))
            reason = "gym_collision" if gym_collision else "gym_done"
            break
    if writer is not None:
        writer.release()
    row = {
        "baseline": baseline, "sim": "f1tenth", "map": map_label, "n_agents": n,
        "dynamics": dynamics, "target_speed": speed, "seed": seed, "trial": trial,
        "formation": formation, "course": proto.course, "zone": proto.zone, "steps": steps, "sim_time_s": round(steps * proto.dt, 3),
        "wall_time_s": round(time.time() - t0, 2), "terminated": reason,
        "gym_collision": gym_collision, "video": record,
    }
    row.update(metrics.summary())
    if formation == "column":
        row["deformability"] = float("nan")     # a convoy has no lateral spread to measure
    if gym_collision:
        # the plant hit a wall or another car: never a success
        row["success"] = 0.0
        row["collision"] = True
    if trace:
        row["_trace"] = np.asarray(positions)
    return row
