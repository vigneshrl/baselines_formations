"""The f1tenth plant, and the command translators every baseline needs to drive it.

The native models of the baselines live in their own repos (RVO2 discs, DEFORM's
TurtleBot NMPC, GCBF+'s DubinsCar, LAS's own f1tenth env); this module only
owns the *f1tenth backend*: one env builder that takes the vehicle model as an
argument, and the conversions from each baseline's command type to
``[steering_angle, speed]``.
"""
from __future__ import annotations

import math
import pathlib
from typing import Optional, Tuple

import numpy as np

VEHICLE = {
    "wheelbase": 0.3302, "width": 0.31, "length": 0.58, "radius": 0.29,
    "steer_max": 0.4189, "v_max": 20.0,
}
DYNAMICS = {"st": "st", "ks": "ks", "kinematic": "ks", "dynamic": "st", "single_track": "st"}


def _patch_track_lookup(roots) -> None:
    import f1tenth_gym.envs.track.track as _t
    import f1tenth_gym.envs.track.utils as _u
    original = getattr(_u, "_ffbench_original_find", None) or _u.find_track_dir
    _u._ffbench_original_find = original
    paths = [pathlib.Path(r) for r in roots]

    def find(name):
        for root in paths:
            if (root / name).exists():
                return root / name
        return original(name)

    _t.find_track_dir = find
    _u.find_track_dir = find


class F1TenthSim:
    """Bare f1tenth_gym env: model selectable, spawn by pose, no RL wrappers."""

    def __init__(self, map_dir, num_agents: int, dynamics: str = "st",
                 dt: float = 0.01, render=False, mu: float = 1.0):
        import gymnasium as gym
        import f1tenth_gym  # noqa: F401  (registers the env)

        map_dir = pathlib.Path(map_dir).resolve()
        _patch_track_lookup([map_dir.parent])
        self.model = DYNAMICS[str(dynamics).lower()]
        self.dt = float(dt)
        self.num_agents = int(num_agents)
        self.map_dir = map_dir
        self.env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": map_dir.name, "num_agents": self.num_agents, "timestep": self.dt,
                "integrator": "rk4", "control_input": ["speed", "steering_angle"],
                "model": self.model, "observation_config": {"type": "original"},
                "params": {"mu": mu}, "reset_config": {"type": None},
            },
            render_mode=("human" if render is True else (render or None)),
        )
        self.render_mode = "human" if render is True else (render or None)
        self.render_enabled = self.render_mode == "human"
        self.env.reset()
        self.track = self.env.unwrapped.track

    # -- track ---------------------------------------------------------
    def centerline(self) -> Tuple[np.ndarray, np.ndarray]:
        return (np.asarray(self.track.centerline.xs, dtype=np.float32),
                np.asarray(self.track.centerline.ys, dtype=np.float32))

    # -- sim -----------------------------------------------------------
    def reset(self, poses: np.ndarray) -> dict:
        poses = np.asarray(poses, dtype=np.float32).reshape(self.num_agents, 3)
        obs, _ = self.env.reset(options={"poses": poses})
        return obs

    def step(self, actions: np.ndarray):
        """actions: (n, 2) of [steering_angle, speed]. Returns (obs, done, info)."""
        obs, _, done, _, info = self.env.step(np.asarray(actions, dtype=np.float32))
        if self.render_enabled:
            self.env.render()
        return obs, bool(done), info

    def frame(self):
        """Current frame as an RGB array (needs render='rgb_array')."""
        return self.env.render()

    def close(self) -> None:
        self.env.close()


class translators:
    """Command translators onto ``[steering_angle, speed]``."""

    @staticmethod
    def wrap(a: float) -> float:
        return float((a + math.pi) % (2.0 * math.pi) - math.pi)

    @staticmethod
    def twist_to_ackermann(v: float, omega: float, wheelbase: float = VEHICLE["wheelbase"],
                           steer_max: float = VEHICLE["steer_max"], nudge_speed: float = 0.2
                           ) -> Tuple[float, float]:
        """Unicycle (v, w) -> (steer, speed); DEFORM's bridge mapping.

        A diff-drive planner will command w at v == 0 to turn in place; a car
        cannot, so a small forward nudge is applied so the steer takes effect.
        """
        if abs(v) < 1e-3 and abs(omega) > 1e-3:
            v = nudge_speed
        steer = math.atan2(omega * wheelbase, v) if abs(v) > 1e-6 else 0.0
        return float(np.clip(steer, -steer_max, steer_max)), float(v)

    @staticmethod
    def velocity_to_action(vx: float, vy: float, theta: float, lookahead: float = 1.5,
                           wheelbase: float = VEHICLE["wheelbase"],
                           steer_max: float = VEHICLE["steer_max"],
                           v_min: float = 0.0, v_max: float = VEHICLE["v_max"]
                           ) -> Tuple[float, float]:
        """Holonomic velocity -> (steer, speed) by pure-pursuit geometry (ORCA's mapping)."""
        spd = math.hypot(vx, vy)
        if spd > 0.1:
            alpha = translators.wrap(math.atan2(vy, vx) - theta)
            steer = math.atan2(2.0 * wheelbase * math.sin(alpha), lookahead)
        else:
            steer = 0.0
        return (float(np.clip(steer, -steer_max, steer_max)),
                float(np.clip(spd, v_min, v_max)))

    @staticmethod
    def dubins_to_action(omega: float, v_now: float, accel: float, dt: float,
                         wheelbase: float = VEHICLE["wheelbase"],
                         steer_max: float = VEHICLE["steer_max"],
                         min_speed_for_steer: float = 0.5) -> Tuple[float, float]:
        """DubinsCar (yaw rate, accel) -> (steer, speed); GCBF+'s cross-over mapping."""
        v_cmd = max(0.0, v_now + accel * dt)
        steer = math.atan(wheelbase * omega / max(abs(v_now), min_speed_for_steer))
        return float(np.clip(steer, -steer_max, steer_max)), float(v_cmd)
