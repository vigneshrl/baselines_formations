"""ORCA in its native simulator: RVO2 holonomic discs, no vehicle model.

Same reference, spawn, lane offsets and obstacle polygons as the f1tenth
backend; the only difference is that RVO2 integrates the agents itself.
"""
from __future__ import annotations

import math
import time

import numpy as np

from ffbench.eval.metrics import ZoneMetrics
from ffbench.eval.protocol import build_reference
from ffbench.maps.targets.rvo2 import rvo2_polygons
from ffbench.paths import stub_envs_f110


def run_orca_native_trial(req, seed: int, trial: int) -> dict:
    stub_envs_f110()
    import orca

    src, proto, n, speed = req.src, req.proto, req.n, req.speed
    dt = proto.native_dt
    ref = build_reference(src, src.centerline[:, 0], src.centerline[:, 1], n, proto, "abreast", seed=seed)
    cfg = orca.OrcaConfig(num_cars=n, map_name=src.name, render=False, seed=seed,
                          target_speed=float(speed), max_speed=max(10.0, float(speed)),
                          dt=dt, lateral_gap=proto.lateral_gap)
    walls = bool(req.options.get("orca_walls", True))
    wrapper = orca.ORCAWrapper(ref.spawn_poses[:, :2], cfg.max_speed, cfg,
                               obstacle_polygons=rvo2_polygons(src, walls=walls, obstacles=True))
    metrics = ZoneMetrics(ref.xs, ref.ys, dt=dt, narrow_center_xy=ref.narrow_xy,
                             gap_width_m=ref.gap_width_m, zone_half_width=ref.zone_half_wp,
                             collision_thresh=proto.collision_thresh)
    xs, ys = ref.xs, ref.ys
    nwp = len(xs)
    wp_idx = [ref.idx0] * n
    max_steps = int(proto.max_steps * proto.dt / dt)
    reason = "max_steps"
    t0 = time.time()
    steps = 0
    rng = np.random.default_rng(seed)
    positions = []
    for step in range(max_steps):
        pos = np.array([wrapper.sim.getAgentPosition(a) for a in wrapper.agent_ids])
        v_pref = np.zeros((n, 2))
        for i in range(n):
            idx = orca._next_wp_idx(float(pos[i, 0]), float(pos[i, 1]), xs, ys, wp_idx[i],
                                    cfg.lookahead_wp, closed=ref.track_closed)
            wp_idx[i] = idx
            if not ref.track_closed and idx >= nwp - 1:
                continue
            tx, ty = orca._safe_tangent(xs, ys, idx, ref.track_closed, nwp)
            off = ref.lane_offsets[i]
            lx = float(xs[idx]) - ty * off
            ly = float(ys[idx]) + tx * off
            dx, dy = lx - pos[i, 0], ly - pos[i, 1]
            d = math.hypot(dx, dy) + 1e-9
            # tiny jitter breaks the perfect symmetry RVO2 otherwise deadlocks on
            v_pref[i] = [speed * dx / d + rng.normal(0, 1e-3), speed * dy / d + rng.normal(0, 1e-3)]
        wrapper.set_pref_vels(v_pref)
        wrapper.step()
        vel = wrapper.get_velocities()
        pos = np.array([wrapper.sim.getAgentPosition(a) for a in wrapper.agent_ids])
        obs = {
            "poses_x": pos[:, 0], "poses_y": pos[:, 1],
            "poses_theta": np.arctan2(vel[:, 1], vel[:, 0]),
            "linear_vels_x": np.hypot(vel[:, 0], vel[:, 1]),
        }
        metrics.step(obs)
        steps = step + 1
        if req.options.get("save_traces"):
            positions.append(pos.copy())
        if metrics.all_zone_cleared:
            reason = "zone_cleared"
            break
        if metrics.all_done:
            reason = "goal"
            break
    row = {
        "baseline": "orca", "sim": "native", "map": req.map_label, "n_agents": n,
        "dynamics": "rvo2_disc", "target_speed": speed, "seed": seed, "trial": trial,
        "formation": "abreast", "steps": steps, "sim_time_s": round(steps * dt, 3),
        "wall_time_s": round(time.time() - t0, 2), "terminated": reason,
    }
    row.update(metrics.summary())
    if positions:
        row["_trace"] = np.asarray(positions)
    return row


def run_orca_native(req):
    return [run_orca_native_trial(req, req.proto.seed + t, t) for t in range(req.proto.trials)]
