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
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._spread_min_multi = None

    def step(self, obs: dict) -> None:
        super().step(obs)
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
        return out
