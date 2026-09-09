"""Draw the trajectories a run produced on top of the corridor.

    python -m ffbench.eval.plot_traces ffbench_results/orca_standard_ON_n4_both [--trials 4] [--out png]

Reads ``<stem>.jsonl`` and ``<stem>_traces.npz`` (written with
``run_experiment.py --save_traces``).  One panel per (sim, trial): corridor
polygon, obstacles, the scored zone (entry and exit lines), one colour per
agent, start marker and end marker.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ffbench.eval.protocol import Protocol, build_reference
from ffbench.maps.registry import resolve
from ffbench.maps.source import MapSource

COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


def _draw_map(ax, src: MapSource, ref):
    poly = src.corridor()
    xs, ys = poly.exterior.xy
    ax.fill(xs, ys, color="#f2f2f2", zorder=0)
    ax.plot(xs, ys, color="#333", lw=1.2, zorder=1)
    for hole in poly.interiors:
        hx, hy = hole.xy
        ax.fill(hx, hy, color="#666", zorder=1)
    # zone entry / exit lines across the corridor
    t, n = ref.tangent, np.array([-ref.tangent[1], ref.tangent[0]])
    ze = ref.zone_entry_idx if ref.zone_entry_idx is not None else ref.narrow_idx - ref.zone_half_wp
    zx = ref.zone_exit_idx if ref.zone_exit_idx is not None else ref.narrow_idx + ref.zone_half_wp
    for k, label in ((ze, "zone entry"), (zx, "zone exit")):
        c = np.array([ref.xs[k], ref.ys[k]])
        a, b = c - n * 8, c + n * 8
        ax.plot([a[0], b[0]], [a[1], b[1]], "--", color="#2a9d8f", lw=1, zorder=2)
        ax.annotate(label, c + n * 8, fontsize=7, color="#2a9d8f")
    ax.plot(ref.xs[ref.idx0:zx + 5], ref.ys[ref.idx0:zx + 5], ":", color="#999", lw=0.8, zorder=2)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--trials", type=int, default=4, help="panels per sim")
    ap.add_argument("--out", default=None)
    ap.add_argument("--margin", type=float, default=6.0)
    args = ap.parse_args(argv)
    stem = args.stem[:-6] if args.stem.endswith(".jsonl") else args.stem
    rows = [json.loads(l) for l in open(stem + ".jsonl") if l.strip()]
    traces = np.load(stem + "_traces.npz")
    keys = [k for k in traces.files]
    by_sim = {}
    for k in keys:
        sim, mp, trial = k.split("|")
        by_sim.setdefault((sim, mp), []).append((int(trial), k))
    panels = []
    for (sim, mp), lst in sorted(by_sim.items()):
        for trial, k in sorted(lst)[: args.trials]:
            panels.append((sim, mp, trial, k))
    if not panels:
        raise SystemExit("no traces in " + stem)
    ncol = min(4, len(panels))
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 5.2 * nrow), squeeze=False)
    cache = {}
    for ax, (sim, mp, trial, k) in zip(axes.ravel(), panels):
        if mp not in cache:
            src = MapSource.load(resolve(mp)[0][1])
            n = int(rows[0]["n_agents"])
            course = rows[0].get("course", "zone")
            ref = build_reference(src, src.centerline[:, 0], src.centerline[:, 1], n, Protocol(course=course, zone=rows[0].get("zone", "pinch")), "abreast")
            cache[mp] = (src, ref)
        src, ref = cache[mp]
        _draw_map(ax, src, ref)
        tr = traces[k]                       # (T, n, 2)
        for i in range(tr.shape[1]):
            ax.plot(tr[:, i, 0], tr[:, i, 1], color=COLORS[i % len(COLORS)], lw=1.4, zorder=3)
            ax.plot(tr[0, i, 0], tr[0, i, 1], "o", color=COLORS[i % len(COLORS)], ms=4, zorder=4)
            ax.plot(tr[-1, i, 0], tr[-1, i, 1], "x", color=COLORS[i % len(COLORS)], ms=6, mew=1.8, zorder=4)
        row = next((r for r in rows if r["sim"] == sim and r["map"] == mp and int(r["trial"]) == trial), {})
        ax.set_title(f"{row.get('baseline', '')} | {sim} | {mp} | trial {trial}\n"
                     f"{row.get('terminated', '')}, cleared {row.get('n_completed', '?')}/{row.get('n_agents', '?')}, "
                     f"V={row.get('avg_speed_mps', float('nan')):.2f} m/s, T={row.get('time_to_goal_s', float('nan'))}",
                     fontsize=8)
        cx, cy = src.narrow_xy
        allpts = tr.reshape(-1, 2)
        ax.set_xlim(min(allpts[:, 0].min(), cx - args.margin) - 2, max(allpts[:, 0].max(), cx + args.margin) + 2)
        ax.set_ylim(min(allpts[:, 1].min(), cy - 16) - 2, max(allpts[:, 1].max(), cy + 16) + 2)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    out = args.out or stem + "_traces.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
