"""Map generator CLI.

    python -m ffbench.maps.generate --map standard_ON --target f1tenth rvo2 ros gcbf
    python -m ffbench.maps.generate --map standard_ON --target ros --num-agents 4 \
        --robot-radius 0.22            # rescale the corridor for a TurtleBot3 waffle
"""
from __future__ import annotations

import argparse
import pathlib

from ffbench.maps.registry import describe, resolve
from ffbench.maps.source import MapSource
from ffbench.maps.targets import TARGETS
from ffbench.paths import GENERATED


def load_source(spec: str, robot_radius: float | None = None, scale: float | None = None):
    """Resolve one map spec to a (possibly rescaled) MapSource list."""
    out = []
    for label, d in resolve(spec):
        src = MapSource.load(d)
        k = scale if scale is not None else (src.scale_for_robot(robot_radius) if robot_radius else 1.0)
        out.append((label, src.rescaled(k)))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default="standard_ON", help="alias, split[:N] or map dir")
    ap.add_argument("--target", nargs="+", default=["f1tenth"], choices=sorted(TARGETS))
    ap.add_argument("--out", default=str(GENERATED))
    ap.add_argument("--num-agents", type=int, default=4, help="ros target: agents to spawn")
    ap.add_argument("--robot-radius", type=float, default=None,
                    help="rescale the map so this robot radius has the f1tenth car's corridor ratio")
    ap.add_argument("--scale", type=float, default=None, help="explicit metre scale factor")
    ap.add_argument("--list", action="store_true", help="print known maps and exit")
    args = ap.parse_args(argv)
    if args.list:
        print(describe())
        return
    for label, src in load_source(args.map, args.robot_radius, args.scale):
        for tgt in args.target:
            kw = {"n_agents": args.num_agents} if tgt == "ros" else {}
            path = TARGETS[tgt](src, pathlib.Path(args.out) / tgt, **kw)
            print(f"[{label}] {tgt:8s} scale={src.scale:.3f} -> {path}")


if __name__ == "__main__":
    main()
