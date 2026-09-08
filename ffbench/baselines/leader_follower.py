"""Leader-follower convoy on the f1tenth plant (``baselines/leader_follower.py``).

Its native simulator *is* f1tenth_gym, so ``--sim native`` runs this too.
"""
from __future__ import annotations

import numpy as np

from ffbench.paths import stub_envs_f110


class LeaderFollowerController:
    formation = "column"
    control_hz = 100.0

    def __init__(self):
        self._r = None

    def reset(self, ctx, obs) -> None:
        stub_envs_f110()
        import leader_follower as lf

        n, ref, proto = ctx.n, ctx.ref, ctx.proto
        speed = float(ctx.speed)
        cfg = lf.LeaderFollowerConfig(
            num_cars=n, map_name=ctx.src.name, render=False, seed=ctx.seed, dt=ctx.dt,
            leader_speed=speed, max_speed=max(10.0, speed), min_speed=min(1.5, 0.5 * speed),
            spawn_gap=proto.column_gap,
        )
        r = lf.F1TenthLeaderFollower.__new__(lf.F1TenthLeaderFollower)
        r.cfg = cfg
        r._track_closed = ref.track_closed
        r._ext_xs = np.asarray(ref.xs, dtype=np.float32)
        r._ext_ys = np.asarray(ref.ys, dtype=np.float32)
        r.car_pp = []
        r.wp_idx_cars = []
        for i in range(n):
            la = cfg.lookahead_dist if i == 0 else cfg.follower_lookahead
            r.car_pp.append(lf.PurePursuit(r._ext_xs, r._ext_ys, la, cfg.wheelbase,
                                           cfg.steer_max, closed=ref.track_closed))
            r.wp_idx_cars.append(ref.idx0)
        r.gap_controllers = [lf.GapController(cfg) for _ in range(n - 1)]
        hazards = [o.center for o in ctx.src.obstacles] + [ctx.src.narrow_xy]
        r._hazards = np.asarray(hazards, dtype=np.float64).reshape(-1, 2)
        self._r = r

    def act(self, obs, step) -> np.ndarray:
        return self._r._compute_actions(obs)
