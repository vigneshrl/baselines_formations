"""ORCA on the f1tenth plant: the real ``baselines/orca.py`` control law.

The runner shell trick from ``gcbf_baseline/crossover_common.OrcaBrain``:
build ``F1TenthORCARunner`` without ``__init__`` (which would construct its own
gym adapter) and hand it the reference so ``_compute_actions`` -- lane
following + RVO2 + pure-pursuit steering -- runs unchanged.
"""
from __future__ import annotations

import numpy as np

from ffbench.maps.targets.rvo2 import rvo2_polygons
from ffbench.paths import stub_envs_f110


def _orca():
    stub_envs_f110()
    import orca  # baselines/orca.py
    return orca


class OrcaController:
    formation = "abreast"
    control_hz = 100.0

    def __init__(self, walls: bool = True):
        self.walls = walls
        self._r = None

    def reset(self, ctx, obs) -> None:
        orca = _orca()
        n, ref, proto = ctx.n, ctx.ref, ctx.proto
        cfg = orca.OrcaConfig(
            num_cars=n, map_name=ctx.src.name, render=False, seed=ctx.seed,
            target_speed=float(ctx.speed), max_speed=max(10.0, float(ctx.speed)),
            dt=ctx.dt, lateral_gap=proto.lateral_gap,
        )
        walls = bool(ctx.options.get("orca_walls", self.walls))
        r = orca.F1TenthORCARunner.__new__(orca.F1TenthORCARunner)
        r.cfg = cfg
        r.waypoints_x = np.asarray(ref.xs, dtype=np.float32)
        r.waypoints_y = np.asarray(ref.ys, dtype=np.float32)
        r.track_closed = ref.track_closed
        r.lane_offsets = list(ref.lane_offsets)
        r.wp_indices = [ref.idx0] * n
        polys = rvo2_polygons(ctx.src, walls=walls, obstacles=True)
        r.orca = orca.ORCAWrapper(ref.spawn_poses[:, :2], cfg.max_speed, cfg,
                                  obstacle_polygons=polys)
        self._r = r

    def act(self, obs, step) -> np.ndarray:
        return self._r._compute_actions(obs)
