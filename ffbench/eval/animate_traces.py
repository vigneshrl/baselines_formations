"""Animate a saved position trace over the corridor (mp4 via imageio-ffmpeg).

Used for native runs that have no renderer (RVO2 discs); works for any
``(T, n, 2)`` trace:

    python -m ffbench.eval.animate_traces ffbench_results/verify_orca_n4 --key "native|standard_ON|0"
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ffbench.eval.plot_traces import COLORS, _draw_map
from ffbench.eval.protocol import Protocol, build_reference
from ffbench.maps.registry import resolve
from ffbench.maps.source import MapSource


def animate(trace: np.ndarray, map_label: str, row: dict, out: str, fps: int = 20,
            sim_dt: float = 0.05, radius_m: float = 0.25, margin: float = 6.0) -> str:
    course = row.get("course", "zone")
    import imageio.v2 as imageio

    src = MapSource.load(resolve(map_label)[0][1])
    n = trace.shape[1]
    ref = build_reference(src, src.centerline[:, 0], src.centerline[:, 1], n, Protocol(course=course), "abreast")
    fig, ax = plt.subplots(figsize=(5, 7))
    _draw_map(ax, src, ref)
    cx, cy = src.narrow_xy
    pts = trace.reshape(-1, 2)
    ax.set_xlim(min(pts[:, 0].min(), cx - margin) - 2, max(pts[:, 0].max(), cx + margin) + 2)
    ax.set_ylim(min(pts[:, 1].min(), cy - 16) - 2, max(pts[:, 1].max(), cy + 16) + 2)
    if course == "full":
        ax.set_xlim(pts[:, 0].min() - 3, pts[:, 0].max() + 3)
    ax.set_aspect("equal")
    ax.set_title(f"{row.get('baseline', '')} | {row.get('sim', '')} | {map_label}", fontsize=9)
    trails = [ax.plot([], [], color=COLORS[i % len(COLORS)], lw=1.2, alpha=0.7)[0] for i in range(n)]
    discs = [plt.Circle((0, 0), radius_m, color=COLORS[i % len(COLORS)]) for i in range(n)]
    for d in discs:
        ax.add_patch(d)
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=8)
    step = max(1, int(round(1.0 / (fps * sim_dt))))
    fig.tight_layout()
    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=7, macro_block_size=1)
    for k in range(0, len(trace), step):
        for i in range(n):
            trails[i].set_data(trace[: k + 1, i, 0], trace[: k + 1, i, 1])
            discs[i].center = (trace[k, i, 0], trace[k, i, 1])
        txt.set_text(f"t = {k * sim_dt:5.2f} s")
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        writer.append_data(frame)
    writer.close()
    plt.close(fig)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--key", required=True, help='"sim|map|trial" key inside <stem>_traces.npz')
    ap.add_argument("--out", default=None)
    ap.add_argument("--dt", type=float, default=0.05)
    args = ap.parse_args(argv)
    tr = np.load(args.stem + "_traces.npz")[args.key]
    sim, mp, trial = args.key.split("|")
    rows = [json.loads(l) for l in open(args.stem + ".jsonl") if l.strip()]
    row = next((r for r in rows if r["sim"] == sim and r["map"] == mp and int(r["trial"]) == int(trial)), {})
    print(animate(tr, mp, row, args.out or f"{args.stem}_{sim}_t{trial}.mp4", sim_dt=args.dt))


if __name__ == "__main__":
    main()
