"""FastFunnels (ours): frozen patch (funnel) policy + trained follower policies.

Runs the repo's ``envs.ppo_policy.JointEnv`` with both policies frozen, the
same way ``baselines/run_fastfunnels_layouts.py`` scores the layout splits,
and feeds the followers' poses to the shared ``ZoneMetrics``.  The patch car
(f1tenth index N) is the reference, not an agent, so it is excluded from the
metrics but kept in the trace for the video.

Needs the FastFunnels training code and models: ``FASTFUNNELS_ROOT`` (default:
this repo if ``envs/ppo_policy.py`` exists), torch + stable-baselines3.
JointEnv always spawns at the track start, so ``--course zone`` still starts
there and only the scoring window differs.
"""
from __future__ import annotations

import math
import os
import pathlib
import sys
import time
from typing import List

import numpy as np

from ffbench.eval.metrics import ZoneMetrics
from ffbench.eval.protocol import build_reference
from ffbench.models.f1tenth import _patch_track_lookup
from ffbench.paths import ROOT

DEFAULT_PATCH = "patch_policy_models/run_20260518_151612/checkpoint_18510000"   # the "working patch"
DEFAULT_AGENT = "joint_sb3_models/run_20260810_000919/agents/best_model"           # RL follower, N=1 only
CAR_WB, STEER_MAX = 0.3302, 0.4189


def load_patch_policy(zp: str, env):
    """Frozen patch policy -> centred action for JointEnv.

    Legacy raw-action checkpoints (the May/June family, e.g. checkpoint_18510000)
    emit absolute [steer_rad, v, a, b]; today's env expects centred
    [steer_RATE, v_c, a_c, b_c].  Same shim as mpc_follower_native_n.load_patch.
    """
    import pickle
    from stable_baselines3 import PPO
    zp = zp[:-4] if zp.endswith(".zip") else zp
    d, b = os.path.dirname(zp), os.path.basename(zp)
    vn = os.path.join(d, b + "_vecnormalize.pkl")
    if not os.path.exists(vn):
        vn = os.path.join(d, "best_vecnormalize.pkl")
    model = PPO.load(zp + ".zip", device="cpu")
    v = pickle.load(open(vn, "rb"))
    mean, var, clip = v.obs_rms.mean.astype(np.float32), v.obs_rms.var.astype(np.float32), float(v.clip_obs)
    lo, hi = model.action_space.low, model.action_space.high
    legacy = bool(hi[1] > 1.5)

    def act(o):
        o = np.clip((np.asarray(o, np.float32) - mean) / np.sqrt(var + 1e-8), -clip, clip)
        a, _ = model.predict(o[None], deterministic=True)
        r = np.asarray(a, np.float32).flatten()
        if not legacy:
            return r
        r = np.clip(r, lo, hi)
        cfg = env.cfg
        # JointEnv applies patch_action[0] as an absolute steering angle, while
        # the checkpoint was trained (and reaches 98.9 %) behind PatchCarEnv's
        # rate integrator: at most steer_rate_max rad per step toward the
        # commanded angle.  Reproduce that integrator here.
        cur = float(act.steer_state)
        act.steer_state = float(np.clip(cur + np.clip(float(r[0]) - cur, -act.rate_max, act.rate_max),
                                        -STEER_MAX, STEER_MAX))

        def _c(x, lo_, hi_):
            return (x - 0.5 * (lo_ + hi_)) / (0.5 * (hi_ - lo_))
        return np.array([act.steer_state, _c(float(r[1]), 2.0, 10.0),
                         _c(float(r[2]), cfg.patch_a_cmd_min, cfg.patch_a_cmd_max),
                         _c(float(r[3]), cfg.patch_b_cmd_min, cfg.patch_b_cmd_max)], np.float32)
    act.legacy = legacy
    act.steer_state = 0.0
    act.rate_max = 0.03           # PatchEnvConfig.steer_rate_max (rad per 0.01 s step)
    return act


def respawn_at_slots(env, n: int, slots):
    """Put the followers in their NMPC slots behind the patch car (JointEnv's
    training spawn packs 4 cars 0.37 m apart, which is contact at any yaw)."""
    b = env.current_base_obs
    pidx = env.PATCH_CAR_F110_IDX
    px, py, pth = float(b["poses_x"][pidx]), float(b["poses_y"][pidx]), float(b["poses_theta"][pidx])
    fx, fy, lx, ly = math.cos(pth), math.sin(pth), -math.sin(pth), math.cos(pth)
    poses = [[px + a * fx + l * lx, py + a * fy + l * ly, pth] for a, l in slots[:n]] + [[px, py, pth]]
    base_obs, _ = env.f110.reset(poses=np.asarray(poses, np.float32))
    env.current_base_obs = base_obs
    vx, vy = float(base_obs["linear_vels_x"][pidx]), float(base_obs["linear_vels_y"][pidx])
    env.patch.sync_from_pose(float(base_obs["poses_x"][pidx]), float(base_obs["poses_y"][pidx]),
                             float(base_obs["poses_theta"][pidx]), float(np.hypot(vx, vy)), 0.0)
    env.patch_accel = 0.0
    env._prev_patch_v = float(env.patch.v)
    env._build_all_obs(base_obs)
    env._step_terminated = env._step_truncated = False
    env._step_info = {}


class NMPCFollowers:
    """The paper's decentralised NMPC followers (envs/mpc.py SEMPCSolver) riding
    in wedge slots behind the patch car, ported from mpc_follower_native_n.py
    (kinematic prediction model, patch-at-origin frame, patch car + other
    followers as neighbours, last plan held open-loop on a failed solve)."""

    GAP, SLOT_D, SLOT_LAT = 1.2, 1.2, 0.62     # rows a car length + 0.6 m apart on the gym plant (the script used 0.9 / 0.8)
    # Script values except the speed cap: on the gym's single-track plant the
    # followers cannot take the funnel bend at 12 m/s, so cap at the patch's
    # own 10 m/s.  (A lower accel cap or the hard keep-out made IPOPT hit its
    # iteration limit whenever the patch accelerated -- verified 2026-09-08.)
    V_LO, V_HI, ACCEL_MAX, MIN_DIST = 0.5, 10.0, 9.5, 0.8

    def __init__(self, n: int, dt: float, control_every: int = 5):
        from envs.mpc import MPCConfig, SEMPCSolver
        self.n, self.dt, self.every = n, dt, control_every
        self.solvers = [SEMPCSolver(MPCConfig(
            v_max=self.V_HI, v_min=self.V_LO, accel_max=self.ACCEL_MAX, steering_max=STEER_MAX,
            num_neighbors=max(1, n), horizon_seconds=1.5, horizon_steps=15, w_vel=5.0, w_center=80.0,
            w_contain=1500.0, max_iter=200, min_agent_dist=self.MIN_DIST, model="kinematic",
            w_collision=200.0, collision_radius=0.55, w_collision_slack=4000.0)) for _ in range(n)]
        self.hold = [[0.0, 2.0] for _ in range(n)]
        self.hold_k = [0] * n
        self.fails = 0
        self._pth, self._om, self._pv, self._acc = None, 0.0, None, 0.0
        rows = {1: [(-self.GAP, 0.0)], 2: [(-self.GAP, self.SLOT_LAT), (-self.GAP, -self.SLOT_LAT)],
                3: [(-self.GAP, self.SLOT_LAT), (-self.GAP, -self.SLOT_LAT), (-self.GAP - self.SLOT_D, 0.0)],
                4: [(-self.GAP, self.SLOT_LAT), (-self.GAP, -self.SLOT_LAT),
                    (-self.GAP - self.SLOT_D, self.SLOT_LAT), (-self.GAP - self.SLOT_D, -self.SLOT_LAT)]}
        self.slots = rows.get(n) or [(-self.GAP - (i // 2) * self.SLOT_D, self.SLOT_LAT if i % 2 == 0 else -self.SLOT_LAT)
                                     for i in range(n)]

    def _patch_kin(self, p, dt):
        raw = 0.0 if self._pth is None else ((p.theta - self._pth + math.pi) % (2 * math.pi) - math.pi) / dt
        self._pth = p.theta
        self._om += 0.02 * (float(np.clip(raw, -2.5, 2.5)) - self._om)
        pv = max(float(p.v), 0.5)
        a_raw = 0.0 if self._pv is None else (pv - self._pv) / dt
        self._pv = pv
        self._acc += 0.05 * (float(np.clip(a_raw, -self.ACCEL_MAX, self.ACCEL_MAX)) - self._acc)
        return self._om, self._acc

    def _center_traj(self, p, i, n, dt, omega, accel):
        th0, v0 = p.theta, max(float(p.v), 0.5)
        along_i, lat_i = self.slots[i]
        out = np.zeros((n, 2), np.float32)
        cx = cy = 0.0
        for k in range(n):
            h = th0 + omega * k * dt
            ux, uy = math.cos(h), math.sin(h)
            out[k] = (cx + along_i * ux - lat_i * uy, cy + along_i * uy + lat_i * ux)
            vk = min(max(v0 + accel * k * dt, 0.5), self.V_HI)
            cx += vk * ux * dt
            cy += vk * uy * dt
        return out

    def act(self, base_obs, patch, step: int) -> np.ndarray:
        """(n, 2) of [steer_rad, speed_mps] for the followers (f1tenth cars 0..n-1)."""
        n = self.n
        px, py = np.asarray(base_obs["poses_x"], float), np.asarray(base_obs["poses_y"], float)
        th, vx = np.asarray(base_obs["poses_theta"], float), np.asarray(base_obs["linear_vels_x"], float)
        if step % self.every:
            return np.array(self.hold, np.float32)
        omega, accel = self._patch_kin(patch, self.dt * self.every)
        pvx, pvy = patch.v * math.cos(patch.theta), patch.v * math.sin(patch.theta)

        class _Shim:
            x = y = 0.0
            theta, a, b, v = patch.theta, patch.a, patch.b, patch.v
            vx, vy = pvx, pvy
        for i in range(n):
            x0 = np.array([px[i] - patch.x, py[i] - patch.y, th[i], max(vx[i], 0.0)], np.float32)
            sol = self.solvers[i]
            ct = self._center_traj(patch, i, sol.config.horizon_steps + 1, sol.dt, omega, accel)
            nbr, nbr_v = [[0.0, 0.0]], [[pvx, pvy]]
            for j in range(n):
                if j != i:
                    nbr.append([px[j] - patch.x, py[j] - patch.y])
                    nbr_v.append([vx[j] * math.cos(th[j]), vx[j] * math.sin(th[j])])
            U, ok = sol.solve(x0, _Shim(), nbr, center_traj=ct, neighbor_vels=nbr_v)
            if not ok or U is None:
                self.fails += 1
                self.hold_k[i] += 1
                Us, Xs = sol.prev_U_sol, sol.prev_X_sol
                if Us is not None:
                    k = min(self.hold_k[i], Us.shape[1] - 1); kl = min(self.hold_k[i] + 3, Xs.shape[1] - 1)
                    self.hold[i] = [float(np.clip(Us[1, k], -STEER_MAX, STEER_MAX)),
                                    float(np.clip(Xs[3, kl], self.V_LO, self.V_HI))]
                continue
            self.hold_k[i] = 0
            Xs = sol.prev_X_sol
            v_cmd = float(Xs[3, min(3, sol.config.horizon_steps)]) if Xs is not None else vx[i] + float(U[0]) * sol.dt
            self.hold[i] = [float(np.clip(U[1], -STEER_MAX, STEER_MAX)), float(np.clip(v_cmd, self.V_LO, self.V_HI))]
        return np.array(self.hold, np.float32)


def _ff_root() -> pathlib.Path:
    r = pathlib.Path(os.environ.get("FASTFUNNELS_ROOT", ROOT))
    if not (r / "envs" / "ppo_policy.py").exists():
        raise RuntimeError("FastFunnels training code not found: set FASTFUNNELS_ROOT to the FastFunnels checkout")
    return r


def run_fastfunnels(req, mode: str) -> List[dict]:
    root = _ff_root()
    for p in (str(root), str(root / "presentation_code")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from envs.ppo_policy import JointEnv, JointEnvConfig  # noqa: WPS433
    from mass_eval import load_policy

    src, proto, n = req.src, req.proto, req.n
    patch_path = req.options.get("patch_model") or str(root / DEFAULT_PATCH)
    agent_path = req.options.get("agent_model") or str(root / DEFAULT_AGENT)
    follower = (req.options.get("follower") or "nmpc").lower()      # nmpc (the paper's DMPC) | rl (N=1 checkpoint)
    predict_agent, agent_obs_dim = (load_policy(agent_path) if follower == "rl" else (None, None))
    _patch_track_lookup([src.map_dir.parent])

    rows = []
    for t in range(proto.trials):
        seed = proto.seed + t
        t0 = time.time()
        record = req.options.get("save_traces") and req.options.get("record_video", True)
        env = JointEnv(JointEnvConfig(num_agents=n, map_name=src.name,
                                      render_mode="rgb_array" if record else None,
                                      random_spawn=False, max_steps=proto.max_steps,
                                      agent_use_lidar=True, obs_mode="lidar"))
        writer, video_path = None, None
        if record:
            import cv2
            out_dir = pathlib.Path(req.options.get("out_dir") or (ROOT / "ffbench_results"))
            video_path = str(out_dir / f"fastfunnels_n{n}_{proto.course}_t{t:02d}.mp4")

            def _frame():
                nonlocal writer
                fr = env.f110.render()
                if fr is None:
                    return
                if writer is None:
                    h, w = fr.shape[:2]
                    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
                writer.write(cv2.cvtColor(np.asarray(fr), cv2.COLOR_RGB2BGR))
        env._real_agents_active = True
        env.reset(seed=seed)
        predict_patch = load_patch_policy(patch_path, env)
        if follower == "rl" and env._step_obs[1] is not None and env._step_obs[1].shape[0] != agent_obs_dim:
            env.close()
            raise RuntimeError(f"follower policy expects {agent_obs_dim}-D obs but JointEnv(num_agents={n}) "
                               f"gives {env._step_obs[1].shape[0]}-D: this checkpoint was trained for a different N; "
                               f"use --follower nmpc")
        nmpc = NMPCFollowers(n, proto.dt) if follower == "nmpc" else None
        if nmpc is not None:
            respawn_at_slots(env, n, nmpc.slots)
        v_mid, v_span = 0.5 * (env.cfg.agent_speed_min + env.cfg.agent_speed_max), 0.5 * (env.cfg.agent_speed_max - env.cfg.agent_speed_min)
        track = env.f110.base_env.unwrapped.track
        xs = np.asarray(track.centerline.xs, dtype=np.float32)
        ys = np.asarray(track.centerline.ys, dtype=np.float32)
        ref = build_reference(src, xs, ys, n, proto, "abreast")
        metrics = ZoneMetrics(ref.xs, ref.ys, dt=proto.dt, narrow_center_xy=ref.narrow_xy,
                              gap_width_m=ref.gap_width_m, zone_half_width=ref.zone_half_wp,
                              collision_thresh=proto.collision_thresh, goal_buffer=ref.goal_buffer_wp,
                          goal_xy=(ref.goal_xy if proto.course in ("full", "tunnel") else None),   # zone course: never stop before the zone is cleared
                              zone_entry_idx=ref.zone_entry_idx, zone_exit_idx=ref.zone_exit_idx,
                          finish_line_m=(proto.goal_buffer_m if proto.course in ("full", "tunnel") else None))

        def follower_obs():
            b = env.current_base_obs
            if b is None:
                return None
            return {"poses_x": np.asarray(b["poses_x"])[:n], "poses_y": np.asarray(b["poses_y"])[:n],
                    "poses_theta": np.asarray(b["poses_theta"])[:n],
                    "linear_vels_x": np.asarray(b["linear_vels_x"])[:n]}

        positions, reason, collided, steps = [], "max_steps", False, 0
        out_of_patch_steps = 0
        for step in range(1, proto.max_steps + 1):
            pa = np.asarray(predict_patch(env._step_obs[0]), dtype=np.float32)
            if nmpc is not None:
                cmd = nmpc.act(env.current_base_obs, env.patch, step)      # [steer, speed m/s]
                aa = np.column_stack([cmd[:, 0], (cmd[:, 1] - v_mid) / v_span]).astype(np.float32)
            else:
                aa = np.stack([np.asarray(predict_agent(env._step_obs[1 + i]), dtype=np.float32) for i in range(n)])
            env._step_with_frozen_policy(pa, aa)
            steps = step
            if record and step % 5 == 0:
                _frame()
            o = follower_obs()
            b = env.current_base_obs
            if o is not None:
                metrics.step(o)
                if req.options.get("save_traces"):
                    positions.append(np.column_stack([b["poses_x"], b["poses_y"]]).copy())   # followers + patch car
            if b is not None and np.any(np.asarray(b.get("collisions", 0))[:n]):
                collided, reason = True, "gym_collision"
                break
            if metrics.all_done:
                reason = "goal"
                break
            if proto.course != "full" and metrics.all_zone_cleared:
                reason = "zone_cleared"
                break
            if env._step_terminated or env._step_truncated:
                info = getattr(env, "_step_info", None) or {}
                why = str(info.get("termination_reason", "")) if isinstance(info, dict) else ""
                if "out_of_patch" in why or "patch_wall" in why:
                    # training-time terminations (a follower left the funnel /
                    # the funnel ellipse touched a wall), not car crashes
                    out_of_patch_steps += 1
                    env._step_terminated = env._step_truncated = False
                    continue
                reason = f"env:{why or 'terminated'}"
                break
        if writer is not None:
            writer.release()
        env.close()
        row = {"baseline": "fastfunnels", "sim": "f1tenth", "map": req.map_label, "course": proto.course,
               "video": video_path if writer is not None else None,
               "n_agents": n, "dynamics": "st", "target_speed": float("nan"), "seed": seed, "trial": t,
               "formation": "jointenv", "steps": steps, "sim_time_s": round(steps * proto.dt, 3),
               "wall_time_s": round(time.time() - t0, 2), "terminated": reason, "gym_collision": collided,
               "patch_model": patch_path, "follower": follower,
               "agent_model": agent_path if follower == "rl" else "nmpc", "patch_legacy_shim": bool(getattr(predict_patch, "legacy", False)),
               "nmpc_fails": nmpc.fails if nmpc else 0, "out_of_patch_steps": out_of_patch_steps}
        row.update(metrics.summary())
        if collided:
            row["success"], row["collision"] = 0.0, True
        if positions:
            row["_trace"] = np.asarray(positions)
        print(f"[fastfunnels/{req.map_label}/n{n}/{proto.course}] trial {t + 1}/{proto.trials}: {reason:13s} "
              f"success={row['success']:.0f} cleared={row['n_completed']}/{n} V={row['avg_speed_mps']:.2f} "
              f"T={row['time_to_goal_s']} finished={row.get('finished')} T_course={row.get('t_course_s')} ({row['wall_time_s']}s)")
        rows.append(row)
    return rows
