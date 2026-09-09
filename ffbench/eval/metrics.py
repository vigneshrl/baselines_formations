"""The one metrics collector: ``baselines/eval_metrics.py::FullRunMetrics``.

Loaded by file path so the repo root's unrelated ``eval_metrics`` module can
never shadow it.  :class:`ZoneMetrics` adds ``deformability_multi``: the same
entry-spread / tightest-spread ratio, but the tightest spread is only sampled
while at least two agents are inside the zone.  The stock definition samples a
lone straggler (spread 0) and returns NaN, which is why the old tables are
mostly NaN in that column; the GCBF+ runner applied the same correction.
"""
from __future__ import annotations

import math

from ffbench.paths import BASELINES, load_module_from_file

_mod = load_module_from_file("ffbench_eval_metrics", BASELINES / "eval_metrics.py")
FullRunMetrics = _mod.FullRunMetrics
_lateral_spread = _mod._lateral_spread
CAR_WIDTH_M = 0.31
METRIC_KEYS = ("avg_speed_mps", "success", "time_to_goal_s", "flow_rate", "deformability")
DIAG_KEYS = ("n_agents", "n_completed", "n_collided", "safety_rate", "collision",
             "collision_full_run", "spread_at_entry_m", "spread_at_narrow_m", "narrow_wp_idx")


class ZoneMetrics(FullRunMetrics):
    def __init__(self, *a, zone_entry_idx=None, zone_exit_idx=None, finish_line_m=None, **kw):
        super().__init__(*a, **kw)
        self._spread_min_multi = None
        # finish LINE (full course): an agent has finished once its along-track
        # projection is within ``finish_line_m`` of the last waypoint, measured
        # along the end tangent -- the nearest-waypoint test misses cars that
        # cross the line a few metres to the side of the centreline's end
        self._finish_line = None
        if finish_line_m is not None and len(self.xs) > 20:
            import numpy as np
            t = np.array([self.xs[-1] - self.xs[-20], self.ys[-1] - self.ys[-20]], float)
            t /= max(float(np.hypot(*t)), 1e-9)
            self._finish_line = (np.array([self.xs[-1], self.ys[-1]], float), t, float(finish_line_m))
        if zone_entry_idx is not None and zone_exit_idx is not None and self._narrow_wp is not None:
            self._zone_entry, self._zone_exit = int(zone_entry_idx), int(zone_exit_idx)
        self._run_speed_sum = None
        self._run_speed_n = None
        self._first_step_with_obs = None

    def step(self, obs: dict) -> None:
        super().step(obs)
        import numpy as np
        n = self._n_agents
        if self._run_speed_sum is None:
            self._run_speed_sum = np.zeros(n); self._run_speed_n = np.zeros(n)
            self._first_step_with_obs = self._step - 1
        spd = np.abs(np.asarray(obs["linear_vels_x"], dtype=float))
        if self._finish_line is not None:
            end, t, buf = self._finish_line
            px = np.asarray(obs["poses_x"], float); py = np.asarray(obs["poses_y"], float)
            for i in range(n):
                if not self._reached_goal[i]:
                    d = np.array([px[i] - end[0], py[i] - end[1]])
                    along, lateral = float(d @ t), float(abs(-t[1] * d[0] + t[0] * d[1]))
                    if along >= -buf and lateral < 8.0:
                        self._reached_goal[i] = True
                        self._goal_step[i] = self._step - 1
        for i in range(n):
            if not self._reached_goal[i]:
                self._run_speed_sum[i] += spd[i]; self._run_speed_n[i] += 1
        if self._narrow_wp is None:
            return
        ids = [i for i in range(self._n_agents) if self._in_zone[i]]
        if len(ids) >= 2:
            import numpy as np
            pos = np.column_stack([obs["poses_x"], obs["poses_y"]])[ids]
            s = float(_lateral_spread(pos, self._lateral_axis))
            if self._spread_min_multi is None or s < self._spread_min_multi:
                self._spread_min_multi = s

    def summary(self) -> dict:
        out = super().summary()
        out["deformability_raw"] = out["deformability"]
        entry = out.get("spread_at_entry_m") or 0.0
        # two cars cannot overlap laterally: floor the tightest spread at one
        # car width, otherwise a single-file moment blows the ratio up
        if self._spread_min_multi is not None and entry > 0:
            tight = max(self._spread_min_multi, CAR_WIDTH_M)
            out["deformability_multi"] = round(entry / tight, 3)
        else:
            out["deformability_multi"] = float("nan")
        out["deformability"] = out["deformability_multi"]
        out["spread_at_narrow_multi_m"] = (round(self._spread_min_multi, 3)
                                          if self._spread_min_multi is not None else float("nan"))
        # ---- whole-course metrics (start line -> finish line)
        import numpy as np
        n = self._n_agents
        done = [bool(x) for x in self._reached_goal]
        out["finished"] = round(sum(done) / n, 3) if n else 0.0
        if n and all(done):
            out["t_course_s"] = round((max(self._goal_step) - (self._first_step_with_obs or 0)) * self.dt, 3)
        else:
            out["t_course_s"] = float("inf") if any(done) or self._step > 0 else float("nan")
        if self._run_speed_n is not None and self._run_speed_n.sum() > 0:
            out["v_course_mps"] = round(float(self._run_speed_sum.sum() / self._run_speed_n.sum()), 3)
        else:
            out["v_course_mps"] = 0.0
        out["zone_entry_idx"], out["zone_exit_idx"] = self._zone_entry, self._zone_exit
        return out
