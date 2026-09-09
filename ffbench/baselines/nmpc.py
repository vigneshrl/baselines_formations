"""Decentralised NMPC baseline: no funnel, no leader.

Every agent solves its own CasADi/IPOPT problem at ``control_hz``:

* **reference** -- its own lane (centreline + fixed lateral offset) sampled
  ahead at the target speed, and a velocity-tracking term along the tangent;
* **corridor** -- left/right half-plane constraints from the traced corridor
  polygon at every horizon step (slacked, heavily penalised);
* **other agents** -- hard slacked keep-out with constant-velocity prediction,
  exactly as ``envs/mpc.py::SEMPCSolver`` handles its neighbours;
* **model** -- the 4-state kinematic bicycle, or the 7-state single-track
  model copied from ``SEMPCSolver._st_rhs`` so it matches the f1tenth plant.

The output is ``[steering_angle, speed]`` for the f1tenth PIDs: the first
planned steering and the planned speed one step ahead.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import casadi as ca
import numpy as np


@dataclass
class NMPCConfig:
    model: str = "kinematic"          # kinematic | st
    horizon_steps: int = 10
    horizon_seconds: float = 1.0
    control_hz: float = 20.0
    v_min: float = 0.3
    v_max: float = 10.0
    accel_max: float = 6.0
    steer_max: float = 0.4189
    steer_rate_max: float = 3.2
    wheelbase: float = 0.3302
    robot_radius: float = 0.29
    wall_margin: float = 0.10
    min_agent_dist: float = 0.5      # hard keep-out; must not exceed the spawn rank gap (0.6 m)
    soft_agent_dist: float = 0.8     # soft penalty radius
    w_center: float = 5.0
    w_vel: float = 50.0
    w_collision: float = 200.0
    w_coll_slack: float = 4000.0
    w_wall_slack: float = 4000.0
    max_iter: int = 150
    st_substeps: int = 3
    # single-track parameters (f1tenth_gym defaults)
    st_mu: float = 1.0489
    st_C_Sf: float = 4.718
    st_C_Sr: float = 5.4562
    st_lf: float = 0.15875
    st_lr: float = 0.17145
    st_h: float = 0.074
    st_m: float = 3.74
    st_Iz: float = 0.04712
    st_v_eps: float = 0.5


class LaneNMPC:
    """One agent's NMPC problem (built once, re-solved with new parameters)."""

    def __init__(self, cfg: NMPCConfig, n_neighbors: int):
        self.cfg = cfg
        self.M = int(n_neighbors)
        self.N = int(cfg.horizon_steps)
        self.dt = cfg.horizon_seconds / self.N
        self.st = cfg.model.lower() == "st"
        self.nx = 7 if self.st else 4
        self.prev_X = None
        self.prev_U = None
        self._build()

    # ---------------------------------------------------------------- model
    def _st_rhs(self, x, u):
        c = self.cfg
        g = 9.81
        DELTA, V, PSI_DOT, BETA = x[2], x[3], x[5], x[6]
        ACCL, STEER_VEL = u[0], u[1]
        Vs = ca.fmax(V, c.st_v_eps)
        lf, lr, hcg, m, Iz = c.st_lf, c.st_lr, c.st_h, c.st_m, c.st_Iz
        Csf, Csr, mu = c.st_C_Sf, c.st_C_Sr, c.st_mu
        gf = g * lr - ACCL * hcg
        gr = g * lf + ACCL * hcg
        psi_ddot = (mu * m / (Iz * (lf + lr))) * (
            lf * Csf * gf * DELTA + (lr * Csr * gr - lf * Csf * gf) * BETA
            - (lf * lf * Csf * gf + lr * lr * Csr * gr) * (PSI_DOT / Vs))
        beta_dot = (mu / (Vs * (lr + lf))) * (
            Csf * gf * DELTA - (Csr * gr + Csf * gf) * BETA
            + (Csr * gr * lr - Csf * gf * lf) * (PSI_DOT / Vs)) - PSI_DOT
        return ca.vertcat(V * ca.cos(x[4] + BETA), V * ca.sin(x[4] + BETA), STEER_VEL,
                          ACCL, PSI_DOT, psi_ddot, beta_dot)

    def _build(self) -> None:
        c, N, M, dt = self.cfg, self.N, self.M, self.dt
        opti = ca.Opti()
        X = opti.variable(self.nx, N + 1)
        U = opti.variable(2, N)
        x, y, accel = X[0, :], X[1, :], U[0, :]
        if self.st:
            delta, v, psi, beta = X[2, :], X[3, :], X[4, :], X[6, :]
            theta = psi + beta
            steer_vel = U[1, :]
        else:
            theta, v, delta = X[2, :], X[3, :], U[1, :]

        p_x0 = opti.parameter(self.nx, 1)
        p_ref = opti.parameter(2, N + 1)
        p_tan = opti.parameter(2, N + 1)
        p_wl = opti.parameter(N + 1, 1)
        p_wr = opti.parameter(N + 1, 1)
        p_vref = opti.parameter(1, 1)
        p_nb = opti.parameter(max(M, 1), 2)
        p_nbv = opti.parameter(max(M, 1), 2)

        # dynamics
        if self.st:
            h = dt / c.st_substeps
            for k in range(N):
                xk, uk = X[:, k], U[:, k]
                for _ in range(c.st_substeps):
                    k1 = self._st_rhs(xk, uk)
                    k2 = self._st_rhs(xk + 0.5 * h * k1, uk)
                    k3 = self._st_rhs(xk + 0.5 * h * k2, uk)
                    k4 = self._st_rhs(xk + h * k3, uk)
                    xk = xk + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                opti.subject_to(X[:, k + 1] == xk)
        else:
            for k in range(N):
                opti.subject_to(x[k + 1] == x[k] + v[k] * ca.cos(theta[k]) * dt)
                opti.subject_to(y[k + 1] == y[k] + v[k] * ca.sin(theta[k]) * dt)
                opti.subject_to(theta[k + 1] == theta[k] + (v[k] / c.wheelbase) * ca.tan(delta[k]) * dt)
                opti.subject_to(v[k + 1] == v[k] + accel[k] * dt)
        opti.subject_to(X[:, 0] == p_x0)

        # bounds (k=0 is pinned by p_x0)
        for k in range(1, N + 1):
            opti.subject_to(v[k] >= c.v_min)
            opti.subject_to(v[k] <= c.v_max)
        for k in range(N):
            opti.subject_to(accel[k] >= -c.accel_max)
            opti.subject_to(accel[k] <= c.accel_max)
        if self.st:
            for k in range(N):
                opti.subject_to(steer_vel[k] >= -c.steer_rate_max)
                opti.subject_to(steer_vel[k] <= c.steer_rate_max)
            for k in range(1, N + 1):
                opti.subject_to(delta[k] >= -c.steer_max)
                opti.subject_to(delta[k] <= c.steer_max)
        else:
            for k in range(N):
                opti.subject_to(delta[k] >= -c.steer_max)
                opti.subject_to(delta[k] <= c.steer_max)

        # corridor half-planes (lateral offset from the lane point, left positive)
        S_w = opti.variable(N + 1, 1)
        r_eff = c.robot_radius + c.wall_margin
        J_wall = 0.0
        for k in range(N + 1):
            dx = x[k] - p_ref[0, k]
            dy = y[k] - p_ref[1, k]
            lat = -p_tan[1, k] * dx + p_tan[0, k] * dy
            opti.subject_to(lat <= p_wl[k] - r_eff + S_w[k])
            opti.subject_to(-lat <= p_wr[k] - r_eff + S_w[k])
            opti.subject_to(S_w[k] >= 0.0)
            J_wall += c.w_wall_slack * (S_w[k] + 10.0 * S_w[k] ** 2)

        # other agents: hard slacked keep-out + soft penalty
        J_coll = 0.0
        S_c = opti.variable(max(M, 1), N + 1)
        r_hard = c.min_agent_dist
        r_soft = c.soft_agent_dist
        for k in range(N + 1):
            for j in range(M):
                nxk = p_nb[j, 0] + p_nbv[j, 0] * (k * dt)
                nyk = p_nb[j, 1] + p_nbv[j, 1] * (k * dt)
                dist_sq = (x[k] - nxk) ** 2 + (y[k] - nyk) ** 2 + 1e-4
                s = S_c[j, k]
                opti.subject_to(dist_sq >= r_hard ** 2 - s)
                opti.subject_to(s >= 0.0)
                J_coll += c.w_coll_slack * (s + 10.0 * s ** 2)
                J_coll += c.w_collision * ca.fmax(0.0, r_soft ** 2 - dist_sq)
        if M == 0:
            opti.subject_to(S_c[0, 0] == 0.0)

        # cost
        J_center = 0.0
        J_vel = 0.0
        for k in range(N + 1):
            J_center += c.w_center * ((x[k] - p_ref[0, k]) ** 2 + (y[k] - p_ref[1, k]) ** 2)
            vx = v[k] * ca.cos(theta[k])
            vy = v[k] * ca.sin(theta[k])
            J_vel += c.w_vel * ((vx - p_vref * p_tan[0, k]) ** 2 + (vy - p_vref * p_tan[1, k]) ** 2)
        J_smooth = 0.0
        for k in range(N - 1):
            J_smooth += (accel[k + 1] - accel[k]) ** 2 + (U[1, k + 1] - U[1, k]) ** 2
        J_effort = 0.0
        for k in range(N):
            J_effort += 0.1 * accel[k] ** 2 + 0.1 * U[1, k] ** 2
        opti.minimize(J_center + J_vel + J_wall + J_coll + J_smooth + J_effort)

        ipopt = {"max_iter": int(c.max_iter), "tol": 1e-3, "acceptable_tol": 2e-2,
                 "acceptable_iter": 12, "acceptable_constr_viol_tol": 1e-2,
                 "print_level": 0, "sb": "yes", "mu_strategy": "adaptive"}
        if self.st:
            ipopt.update({"max_iter": max(int(c.max_iter), 200), "tol": 2e-3,
                          "acceptable_tol": 1e-2, "acceptable_iter": 8,
                          "acceptable_constr_viol_tol": 5e-3, "mu_strategy": "monotone",
                          "nlp_scaling_method": "gradient-based"})
        opti.solver("ipopt", {"expand": True, "print_time": False, "verbose": False}, ipopt)
        self.opti, self.X, self.U = opti, X, U
        self.p = dict(x0=p_x0, ref=p_ref, tan=p_tan, wl=p_wl, wr=p_wr, vref=p_vref,
                      nb=p_nb, nbv=p_nbv)

    # ---------------------------------------------------------------- solve
    def solve(self, x0, ref_pts, tangents, wl, wr, vref, nb_pos, nb_vel
              ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Returns (u0, x1): first input and the planned state one step ahead."""
        N, M, o = self.N, self.M, self.opti
        x0 = np.asarray(x0, float).reshape(self.nx, 1)
        if not np.all(np.isfinite(x0)):
            return None, None
        o.set_value(self.p["x0"], x0)
        o.set_value(self.p["ref"], np.asarray(ref_pts, float).T.reshape(2, N + 1))
        o.set_value(self.p["tan"], np.asarray(tangents, float).T.reshape(2, N + 1))
        o.set_value(self.p["wl"], np.asarray(wl, float).reshape(N + 1, 1))
        o.set_value(self.p["wr"], np.asarray(wr, float).reshape(N + 1, 1))
        o.set_value(self.p["vref"], float(vref))
        nb = np.full((max(M, 1), 2), 1000.0)
        nbv = np.zeros((max(M, 1), 2))
        for j in range(min(M, len(nb_pos))):
            nb[j] = nb_pos[j]
            nbv[j] = nb_vel[j]
        o.set_value(self.p["nb"], nb)
        o.set_value(self.p["nbv"], nbv)
        warm = self.prev_X is not None
        if warm:
            try:
                o.set_initial(self.X, self.prev_X)
                o.set_initial(self.U, self.prev_U)
            except Exception:
                warm = False
        if not warm:
            Xg = np.zeros((self.nx, N + 1))
            v0 = float(np.clip(x0[3, 0], self.cfg.v_min, self.cfg.v_max))
            th = float(x0[4, 0] if self.st else x0[2, 0])
            for k in range(N + 1):
                Xg[0, k] = x0[0, 0] + v0 * math.cos(th) * k * self.dt
                Xg[1, k] = x0[1, 0] + v0 * math.sin(th) * k * self.dt
                Xg[3, k] = v0
                Xg[4 if self.st else 2, k] = th
                if self.st:
                    Xg[2, k] = x0[2, 0]
            try:
                o.set_initial(self.X, Xg)
                o.set_initial(self.U, np.zeros((2, N)))
            except Exception:
                pass
        try:
            sol = o.solve()
            self.prev_X, self.prev_U = sol.value(self.X), sol.value(self.U)
            return np.asarray(sol.value(self.U[:, 0])).ravel(), np.asarray(self.prev_X[:, 1]).ravel()
        except Exception:
            # Do NOT accept IPOPT's last iterate: a near-feasible iterate can
            # still be a spinning trajectory at full steering lock (seen on the
            # full course).  The caller keeps executing the previous plan.
            return None, None


class NMPCController:
    formation = "abreast"

    def __init__(self, cfg: Optional[NMPCConfig] = None):
        self.cfg = cfg or NMPCConfig()
        self.control_hz = self.cfg.control_hz
        self.solvers: List[LaneNMPC] = []

    # ---------------------------------------------------------------- setup
    def reset(self, ctx, obs) -> None:
        cfg = NMPCConfig(**{**self.cfg.__dict__,
                            **{k[5:]: v for k, v in ctx.options.items() if k.startswith("nmpc_")}})
        cfg.v_max = max(cfg.v_max, float(ctx.speed))
        self.cfg = cfg
        self.control_hz = cfg.control_hz
        self.n = ctx.n
        self.speed = float(ctx.speed)
        ref = ctx.ref
        self.xs = np.asarray(ref.xs, float)
        self.ys = np.asarray(ref.ys, float)
        self.s = np.asarray(ref.arclength, float)
        self.offsets = list(ref.lane_offsets)
        nwp = len(self.xs)
        # Tangents from the forward segment, falling back to the backward one
        # across splice jumps (the extended centreline is not uniformly spaced).
        fx = np.append(np.diff(self.xs), 0.0)
        fy = np.append(np.diff(self.ys), 0.0)
        bx = np.insert(np.diff(self.xs), 0, 0.0)
        by = np.insert(np.diff(self.ys), 0, 0.0)
        flen, blen = np.hypot(fx, fy), np.hypot(bx, by)
        jump = 4.0 * ctx.proto.approach_spacing_m
        use_b = (flen > jump) | (flen < 1e-9)
        tx = np.where(use_b, bx, fx)
        ty = np.where(use_b, by, fy)
        nrm = np.maximum(np.hypot(tx, ty), 1e-9)
        self.tx, self.ty = tx / nrm, ty / nrm
        # never reference behind the spawn on the zone course (behind it lies the
        # far-away original centreline); on the full course the path is contiguous
        self.s_min = float(self.s[0] if getattr(ref, "course", "zone") == "full" else self.s[ref.idx0])
        # speed setpoint is read this many horizon steps ahead so the plant's
        # speed PID has something to chase (one step ahead makes the car crawl)
        self.k_cmd = int(min(cfg.horizon_steps, max(1, round(0.3 / (cfg.horizon_seconds / cfg.horizon_steps)))))
        # wall widths where the agents can go: from spawn to well past the zone
        lo = max(0, ref.idx0 - 10)
        hi = min(nwp - 1, ref.narrow_idx + ref.zone_half_wp + 80)
        self.wl = np.zeros(nwp)
        self.wr = np.zeros(nwp)
        for i in range(lo, hi + 1):
            l, r = ctx.src.lateral_extent(self.xs[i], self.ys[i], (self.tx[i], self.ty[i]))
            self.wl[i], self.wr[i] = l, r
        self.hint = [ref.idx0] * self.n
        self.last_delta = np.zeros(self.n)
        self.last_action = np.zeros((self.n, 2), dtype=np.float32)
        self.hold_k = [0] * self.n
        self.solvers = [LaneNMPC(cfg, self.n - 1) for _ in range(self.n)]
        self.st = self.solvers[0].st
        self.dt_ctrl = self.solvers[0].dt
        self.N = cfg.horizon_steps
        self.fails = 0

    def _nearest(self, i: int, x: float, y: float) -> int:
        h = self.hint[i]
        lo, hi = max(0, h - 40), min(len(self.xs) - 1, h + 40)
        d = np.hypot(self.xs[lo:hi + 1] - x, self.ys[lo:hi + 1] - y)
        idx = lo + int(np.argmin(d))
        self.hint[i] = idx
        return idx

    def _horizon(self, i: int, x: float, y: float, v_now: float):
        idx = self._nearest(i, x, y)
        # arclength of the projection onto the local tangent
        s0 = self.s[idx] + (x - self.xs[idx]) * self.tx[idx] + (y - self.ys[idx]) * self.ty[idx]
        v_prog = max(0.5 * self.speed, min(self.speed, v_now))
        sk = s0 + v_prog * self.dt_ctrl * np.arange(self.N + 1)
        sk = np.clip(sk, self.s_min, self.s[-1])
        cx = np.interp(sk, self.s, self.xs)
        cy = np.interp(sk, self.s, self.ys)
        tx = np.interp(sk, self.s, self.tx)
        ty = np.interp(sk, self.s, self.ty)
        nrm = np.maximum(np.hypot(tx, ty), 1e-9)
        tx, ty = tx / nrm, ty / nrm
        # conservative width: the minimum over the waypoints bracketing each
        # sample (+-1), so a wall corner between two samples is never smoothed away
        j = np.searchsorted(self.s, sk)
        lo = np.clip(j - 2, 0, len(self.s) - 1)
        hi = np.clip(j + 1, 0, len(self.s) - 1)
        wl = np.array([self.wl[a:b + 1].min() for a, b in zip(lo, hi)])
        wr = np.array([self.wr[a:b + 1].min() for a, b in zip(lo, hi)])
        off = self.offsets[i]
        lane = np.column_stack([cx - ty * off, cy + tx * off])
        return lane, np.column_stack([tx, ty]), wl - off, wr + off

    # ---------------------------------------------------------------- act
    def act(self, obs, step) -> np.ndarray:
        n = self.n
        px = np.asarray(obs["poses_x"], float)
        py = np.asarray(obs["poses_y"], float)
        th = np.asarray(obs["poses_theta"], float)
        vx = np.asarray(obs["linear_vels_x"], float)
        vy = np.asarray(obs.get("linear_vels_y", np.zeros(n)), float)
        wz = np.asarray(obs.get("ang_vels_z", np.zeros(n)), float)
        vel_world = np.column_stack([vx * np.cos(th) - vy * np.sin(th),
                                     vx * np.sin(th) + vy * np.cos(th)])
        actions = np.zeros((n, 2), dtype=np.float32)
        for i in range(n):
            v_now = float(np.hypot(vx[i], vy[i]))
            lane, tan, wl, wr = self._horizon(i, px[i], py[i], v_now)
            others = [j for j in range(n) if j != i]
            nb_pos = [[px[j], py[j]] for j in others]
            nb_vel = [list(vel_world[j]) for j in others]
            if self.st:
                beta = math.atan2(vy[i], vx[i]) if abs(vx[i]) > 0.5 else 0.0
                x0 = [px[i], py[i], self.last_delta[i], max(v_now, 0.0), th[i], wz[i], beta]
            else:
                x0 = [px[i], py[i], th[i], max(v_now, 0.0)]
            u0, x1 = self.solvers[i].solve(x0, lane, tan, wl, wr, self.speed, nb_pos, nb_vel)
            if u0 is None:
                # failed re-solve: keep executing the last successful plan
                # open-loop (index advances each failed step), like the paper's
                # follower script; brake straight if there is no plan yet
                self.fails += 1
                self.hold_k[i] += 1
                Xs, Us = self.solvers[i].prev_X, self.solvers[i].prev_U
                if Xs is None:
                    actions[i] = [0.0, max(v_now - 1.0, 0.0)]
                    continue
                k = min(self.hold_k[i], Us.shape[1] - 1)
                kl = min(self.hold_k[i] + self.k_cmd, Xs.shape[1] - 1)
                steer = float(Xs[2, k]) if self.st else float(Us[1, k])
                actions[i] = [float(np.clip(steer, -self.cfg.steer_max, self.cfg.steer_max)), max(0.0, float(Xs[3, kl]))]
                continue
            self.hold_k[i] = 0
            plan = self.solvers[i].prev_X
            k = min(self.k_cmd, plan.shape[1] - 1) if plan is not None else 1
            if self.st:
                steer, spd = float(x1[2]), float(plan[3, k] if plan is not None else x1[3])
            else:
                steer, spd = float(u0[1]), float(plan[3, k] if plan is not None else x1[3])
            steer = float(np.clip(steer, -self.cfg.steer_max, self.cfg.steer_max))
            self.last_delta[i] = steer
            actions[i] = [steer, max(0.0, spd)]
        self.last_action = actions
        return actions
