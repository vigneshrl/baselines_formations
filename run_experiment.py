#!/usr/bin/env python
"""Run one baseline on one map, on its native simulator or on f1tenth_gym.

    python run_experiment.py --orca   --num_agents=4 --map=standard_ON
    python run_experiment.py --orca   --num_agents=4 --map=standard_ON --sim=native
    python run_experiment.py --nmpc   --num_agents=4 --map=standard_ON --dynamics=ks
    python run_experiment.py --deform --num_agents=4 --map=standard_ON --sim=native
    python run_experiment.py --baseline leader_follower --num_agents 2 --map eval_matched:10

Results: one JSONL row per trial plus an aggregated CSV and a printed table.
Interpreter: /p/cral/vignesh/envs/fastfunnels/bin/python (GCBF+ re-runs itself
under the gcbf_eval env; DEFORM runs in Docker).
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ffbench  # noqa: E402,F401  (sets sys.path)
from ffbench.baselines.registry import REGISTRY, NativeRequest, get  # noqa: E402
from ffbench.eval.protocol import Protocol  # noqa: E402
from ffbench.eval.report import aggregate, append_jsonl, format_table, write_csv  # noqa: E402
from ffbench.maps.registry import describe, resolve  # noqa: E402
from ffbench.maps.source import MapSource  # noqa: E402
from ffbench.paths import RESULTS  # noqa: E402


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", help="orca | leader_follower | nmpc | deform | gcbf | las")
    for b in REGISTRY:
        ap.add_argument(f"--{b}", action="store_true", help=f"shorthand for --baseline {b}")
    ap.add_argument("--num_agents", "--num-agents", "-n", type=int, default=4)
    ap.add_argument("--map", default="standard_ON")
    ap.add_argument("--sim", choices=["f1tenth", "native", "both"], default="f1tenth")
    ap.add_argument("--dynamics", choices=["st", "ks"], default="st",
                    help="f1tenth plant: single-track (st) or kinematic (ks)")
    ap.add_argument("--speed", type=float, default=5.0, help="target speed m/s")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_steps", "--max-steps", type=int, default=6000)
    ap.add_argument("--formation", choices=["auto", "abreast", "column"], default="auto")
    ap.add_argument("--course", choices=["zone", "full"], default="zone",
                    help="zone: spawn 7 m before the pinch, score the pinch window; "
                         "full: spawn at the start line, score the whole narrow section and start-to-finish")
    ap.add_argument("--zone", choices=["pinch", "section"], default="pinch",
                    help="zone course: score the +-12 m pinch window (pinch) or the whole narrow section (section)")
    ap.add_argument("--spawn", choices=["zone_entry", "track_start"], default="zone_entry")
    ap.add_argument("--lateral_gap", type=float, default=0.6)
    ap.add_argument("--zone_half_m", type=float, default=12.0)
    ap.add_argument("--spawn_up_m", type=float, default=7.0,
                    help="rank spawns this far before the pinch (7 m = the repo's original scripts, inside the 12 m zone)")
    ap.add_argument("--spawn_jitter", type=float, default=0.0,
                    help="+- metres of per-seed spawn jitter (0 = identical trials for deterministic baselines)")
    ap.add_argument("--controller_model", choices=["kinematic", "st"], default=None,
                    help="nmpc: prediction model (default kinematic)")
    ap.add_argument("--orca_no_walls", action="store_true", help="orca: hide the corridor walls from RVO2")
    ap.add_argument("--robot_radius", type=float, default=None,
                    help="deform native: rescale the corridor to this robot radius (default: no rescale, DEFORM's formation parameters are metric)")
    ap.add_argument("--robot_model", default="burger", help="deform native: TurtleBot3 model (DEFORM ships burger + RealSense)")
    ap.add_argument("--patch_model", default=None, help="fastfunnels: patch (funnel) policy path")
    ap.add_argument("--agent_model", default=None, help="fastfunnels: RL follower policy path (N=1 checkpoint)")
    ap.add_argument("--follower", choices=["nmpc", "rl"], default="nmpc", help="fastfunnels: follower controller")
    ap.add_argument("--timeout_s", type=float, default=None, help="deform: per-episode timeout (default 300 s bridge, 600 s native)")
    ap.add_argument("--out", default=None, help="output stem (default ffbench_results/<baseline>_<map>_n<N>_<sim>)")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--record", action="store_true",
                    help="write <out>_<sim>_t<trial>.mp4 (gym renderer for f1tenth runs, trace animation for native)")
    ap.add_argument("--record_trials", type=int, default=1, help="how many trials per condition to record")
    ap.add_argument("--save_traces", action="store_true",
                    help="save per-trial position traces to <out>_traces.npz (plot with ffbench.eval.plot_traces)")
    ap.add_argument("--list", action="store_true", help="list baselines and maps")
    args = ap.parse_args(argv)
    picked = [b for b in REGISTRY if getattr(args, b)]
    if args.baseline:
        picked.append(args.baseline)
    if args.list:
        return args, None
    if len(picked) != 1:
        ap.error("pick exactly one baseline (--orca, --nmpc, ... or --baseline NAME)")
    args.baseline = picked[0]
    return args, get(picked[0])


def main(argv=None) -> int:
    args, spec = parse(argv)
    if args.list:
        print("baselines:")
        for k, s in REGISTRY.items():
            modes = []
            if s.controller or s.external:
                modes.append("f1tenth")
            if s.native or s.external or s.native_note.startswith("its native"):
                modes.append("native")
            print(f"  {k:16s} [{', '.join(modes)}]  {s.description}\n{'':20s}native: {s.native_note}")
        print(describe())
        return 0
    if spec.agent_counts and args.num_agents not in spec.agent_counts:
        print(f"{spec.name} supports num_agents in {spec.agent_counts}")
        return 2
    if args.course == "full" and args.max_steps == 6000:
        args.max_steps = 15000          # 150 s of sim for the 120 m course
    if args.zone == "section" and args.max_steps == 6000:
        args.max_steps = 9000
    proto = Protocol(course=args.course, zone=args.zone, spawn=args.spawn, formation=args.formation, lateral_gap=args.lateral_gap,
                     zone_half_m=args.zone_half_m, spawn_up_m=args.spawn_up_m,
                     max_steps=args.max_steps, trials=args.trials,
                     seed=args.seed, target_speed=args.speed, spawn_jitter_m=args.spawn_jitter,
                     spawn_jitter_rad=0.25 * args.spawn_jitter)
    options = {"orca_walls": not args.orca_no_walls, "controller_model": args.controller_model,
               "save_traces": args.save_traces or args.record,
               "robot_radius": args.robot_radius, "robot_model": args.robot_model,
               "patch_model": args.patch_model, "agent_model": args.agent_model, "follower": args.follower,
               "timeout_s": args.timeout_s}
    sims = ["f1tenth", "native"] if args.sim == "both" else [args.sim]
    maps = resolve(args.map)
    stem = args.out or str(RESULTS / f"{spec.name}_{args.map.replace(':', '-')}_n{args.num_agents}_{args.sim}"
                           + ("_full" if args.course == "full" else ("_section" if args.zone == "section" else "")))
    args.out_stem = stem
    jsonl = pathlib.Path(stem + ".jsonl")
    if jsonl.exists():
        jsonl.unlink()
    rows = []
    traces = {}
    t_all = time.time()
    for sim in sims:
        for label, map_dir in maps:
            src = MapSource.load(map_dir)
            if spec.external is not None:
                req = NativeRequest(src, label, proto, args.num_agents, args.speed, args.dynamics, options)
                new = spec.external(req, sim)
            elif sim == "native" and spec.native is not None:
                req = NativeRequest(src, label, proto, args.num_agents, args.speed, args.dynamics, options)
                new = spec.native(req)
            else:
                if sim == "native":
                    print(f"[{spec.name}] {spec.native_note}; running the f1tenth backend")
                new = _run_f1tenth(spec, src, label, map_dir, proto, args, options)
            for r in new:
                r.setdefault("sim", sim)
                tr = r.pop("_trace", None)
                if tr is not None:
                    traces[f"{r['sim']}|{r['map']}|{r['trial']}"] = tr
                append_jsonl(r, jsonl)
                rows.extend([r])
    if traces:
        import numpy as np
        np.savez_compressed(stem + "_traces.npz", **traces)
        print(f"traces: {stem}_traces.npz  (plot: python -m ffbench.eval.plot_traces {stem})")
        if args.record:
            from ffbench.eval.animate_traces import animate
            for key, tr in traces.items():
                sim_, mp, trial = key.split("|")
                row = next(r for r in rows if r["sim"] == sim_ and r["map"] == mp and int(r["trial"]) == int(trial))
                if row.get("video") or int(trial) >= args.record_trials:
                    continue
                dt_row = proto.native_dt if sim_ == "native" else proto.dt
                print("video:", animate(tr, mp, row, f"{stem}_{sim_}_t{trial}.mp4", sim_dt=dt_row))
    for r in rows:
        if r.get("video"):
            print("video:", r["video"])
    aggs = aggregate(rows)
    csv_path = write_csv(aggs, stem + ".csv")
    print()
    print(format_table(aggs))
    print(f"\nrows: {jsonl}\ntable: {csv_path}\nwall time: {time.time() - t_all:.0f}s")
    return 0


def _run_f1tenth(spec, src, label, map_dir, proto, args, options):
    from ffbench.eval.runner import SimPool, run_f1tenth_trial
    from ffbench.maps.targets.f1tenth import write_f1tenth
    from ffbench.paths import GENERATED

    map_dir = write_f1tenth(src, GENERATED / "f1tenth")
    formation = spec.formation if args.formation == "auto" else args.formation
    pool = SimPool()
    rows = []
    try:
        for t in range(proto.trials):
            controller = spec.controller(options)
            hz = getattr(controller, "control_hz", spec.control_hz)
            row = run_f1tenth_trial(spec.name, controller, src, label, map_dir, proto,
                                    args.num_agents, args.dynamics, args.speed,
                                    proto.seed + t, t, formation=formation, control_hz=hz,
                                    render=args.render, pool=pool, options=options,
                                    trace=args.save_traces or args.record,
                                    record=(f"{args.out_stem}_f1tenth_t{t}.mp4"
                                            if args.record and t < args.record_trials else None))
            print(f"[{spec.name}/{label}/n{args.num_agents}/{args.dynamics}/{proto.course}] trial {t + 1}/{proto.trials}: "
                  f"{row['terminated']:13s} success={row['success']:.0f} cleared={row['n_completed']}/{args.num_agents} "
                  f"V={row['avg_speed_mps']:.2f} T={row['time_to_goal_s']} finished={row.get('finished')} "
                  f"T_course={row.get('t_course_s')} ({row['wall_time_s']}s)")
            rows.append(row)
    finally:
        pool.close()
    return rows


if __name__ == "__main__":
    sys.exit(main())
