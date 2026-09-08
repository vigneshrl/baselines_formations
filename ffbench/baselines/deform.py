"""DEFORM (NeSC-IV) -- runs only inside its ROS Noetic Docker image.

Two backends, both driven through Docker:

* ``native``  -- Gazebo + TurtleBot3 world generated from the map, DEFORM's
                 own planner and TurtleBot NMPC, scored by the ffbench ROS
                 metrics node.  The corridor is rescaled so the TurtleBot has
                 the same body/corridor ratio the f1tenth car has (``--robot-radius``).
* ``f1tenth`` -- DEFORM's planner driving the f1tenth plant through the
                 existing ``f1tenth_deform_bridge`` (``--dynamics`` honoured).

On a host without Docker this module only generates the bundle and prints the
exact commands; nothing is faked.
"""
from __future__ import annotations

import csv
import glob
import os
import re
import pathlib
import shutil
import subprocess
import time
from typing import List

from ffbench.eval.protocol import build_reference
from ffbench.maps.targets.ros import write_ros
from ffbench.paths import DEFORM_DOCKER, GENERATED

IMAGE = "deform_ros:f1tenth_patched"
SIF = pathlib.Path(os.environ.get("FFBENCH_DEFORM_SIF", GENERATED / "deform_ros.sif"))
TB3_RADIUS = {"burger": 0.105, "waffle": 0.22, "waffle_pi": 0.22}


def docker_available() -> bool:
    return shutil.which("docker") is not None


def apptainer_available() -> bool:
    return shutil.which("apptainer") is not None and SIF.exists()


def runtime() -> str:
    """'docker', 'apptainer' or '' -- Docker wins when both exist."""
    if docker_available():
        return "docker"
    if apptainer_available():
        return "apptainer"
    return ""


def prepare_bundle(req, mode: str) -> pathlib.Path:
    """Generate the ROS bundle for this request and return its directory."""
    src = req.src
    if mode == "native":
        robot = req.options.get("robot_model") or "burger"
        r_robot = TB3_RADIUS.get(robot, 0.105)
        # DEFORM's formation parameters are metric (0.8 m interval, 0.35 m
        # clearance) and the 3.2 m corridor is already TurtleBot-scale, so the
        # map is NOT rescaled unless --robot_radius asks for it explicitly
        # (a 0.36x corridor leaves the planner "reducing formation" forever).
        if req.options.get("robot_radius"):
            src = src.rescaled(src.scale_for_robot(float(req.options["robot_radius"])))
        proto = req.proto.scaled(src.scale)
        collision_thresh = 2.0 * r_robot * src.scale      # centre distance at body contact
    else:
        robot = "f1tenth"
        proto = req.proto
        collision_thresh = proto.collision_thresh
    ref = build_reference(src, src.centerline[:, 0], src.centerline[:, 1], req.n, proto, "abreast")
    poses = [(float(x), float(y), float(th)) for x, y, th in ref.spawn_poses]
    goal = (ref.goal_xy[0], ref.goal_xy[1], ref.heading)
    out = pathlib.Path(req.options.get("out_dir") or GENERATED) / "ros"
    return write_ros(
        src, out, n_agents=req.n, poses=poses, goal=goal,
        end_mode="narrow_clear", timeout_s=float(req.options.get("timeout_s") or (600.0 if mode == "native" else 300.0)),
        robot_model=robot if mode == "native" else "burger",
        formation_spacing=proto.lateral_gap, dynamics=req.dynamics,
        zone_half_wp=ref.zone_half_wp, ref_centerline=list(zip(ref.xs.tolist(), ref.ys.tolist())),
        collision_thresh=collision_thresh,
    )


def _docker_cmd(bundle: pathlib.Path, results: pathlib.Path, mode: str, timeout_s: float,
                name: str, map_name: str, n_agents: int = 4) -> List[str]:
    bridge = DEFORM_DOCKER / "f1tenth_deform_bridge"
    vols = [
        f"{bundle / 'track' / map_name}:/opt/f1tenth_gym/maps/{map_name}",
        f"{results}:/tmp/deform_results:rw",
        f"{bundle / 'config' / 'bridge_params.yaml'}:/root/DEFORM/src/utility/f1tenth_deform_bridge/config/bridge_params.yaml:ro",
        f"{bundle / 'config' / 'frm_shape.yaml'}:/root/DEFORM/src/plan_manager/config/frm_shape.yaml:ro",
        f"{bundle / 'launch'}:/root/DEFORM/src/utility/f1tenth_deform_bridge/launch:ro",
        f"{bridge / 'scripts' / 'f1tenth_bridge_node.py'}:/root/DEFORM/devel/lib/f1tenth_deform_bridge/f1tenth_bridge_node.py",
        f"{bridge / 'scripts' / 'eval_metrics.py'}:/root/DEFORM/devel/lib/f1tenth_deform_bridge/eval_metrics.py",
        f"{bundle}:/mnt/ffbench:ro",
    ]
    # one ROS master port per trial: apptainer shares the host network, so two
    # trials (or a stale master) on 11311 would otherwise collide
    port = 11400 + (os.getpid() + int(time.time())) % 500
    if mode == "native":
        inner = (
            f"export ROS_MASTER_URI=http://localhost:{port} && "
            "source /opt/ros/noetic/setup.bash && source /root/DEFORM/devel/setup.bash && "
            f"export TURTLEBOT3_MODEL=$(grep -o 'default=\"[a-z_]*\"' /mnt/ffbench/launch/deform_native.launch | head -1 | cut -d'\"' -f2) && "
            f"(roslaunch -p {port} /mnt/ffbench/launch/deform_native.launch episode_timeout_s:={timeout_s:.0f} gui:=false &) && "
            "sleep 15 && python3 /mnt/ffbench/ffbench_ros/scripts/metrics_node.py "
            f"_config:=/mnt/ffbench/config/metrics.yaml _num_agents:={n_agents} "
            f"_episode_timeout_s:={timeout_s:.0f} _results_dir:=/tmp/deform_results; "
            f"pkill -f 'roslaunch -p {port}'; sleep 3"
        )
    else:
        inner = (
            f"export ROS_MASTER_URI=http://localhost:{port} && "
            "source /opt/ros/noetic/setup.bash && source /root/DEFORM/devel/setup.bash && "
            f"roslaunch -p {port} f1tenth_deform_bridge deform_f1tenth.launch target_episodes:=1 "
            f"episode_timeout_s:={timeout_s:.0f}"
        )
    if runtime() == "apptainer":
        # same mounts, apptainer syntax; writable tmpfs for ROS logs / catkin state
        cmd = ["apptainer", "exec", "--writable-tmpfs", "--cleanenv", "--no-home",
               "--env", f"DISPLAY={os.environ.get('DISPLAY', '')}",
               "--env", "QT_X11_NO_MITSHM=1", "--env", "LIBGL_ALWAYS_SOFTWARE=1",
               "--env", "ROS_HOME=/tmp/ros_home", "--env", "HOME=/tmp/ros_home",
               "--env", "PYTHONPATH=/opt/f1tenth_gym:/mnt/site", "--env", "TURTLEBOT3_MODEL=burger"]
        for v in vols:
            src, dst = v.split(":")[0], v.split(":")[1]
            cmd += ["--bind", f"{src}:{dst}"]
        # the gym as staged for the build (Python 3.8-safe), over the baked copy
        gym_ctx = GENERATED / "build_ctx" / "f1tenth_gym"
        if gym_ctx.exists():
            cmd += ["--bind", f"{gym_ctx}:/opt/f1tenth_gym"]
        # extra pure-python packages installed with `pip3 install --target`
        # inside the container (no rebuild needed), e.g. yamldataclassconfig
        site = GENERATED / "container_site"
        if site.exists():
            cmd += ["--bind", f"{site}:/mnt/site"]
        cmd += [str(SIF), "bash", "-c", "mkdir -p /tmp/ros_home && " + inner]
        return cmd
    cmd = ["docker", "run", "--rm", "--network", "host", "--name", name,
           "--env", f"DISPLAY={os.environ.get('DISPLAY', '')}", "--env", "QT_X11_NO_MITSHM=1",
           "--env", "LIBGL_ALWAYS_SOFTWARE=1", "--volume", "/tmp/.X11-unix:/tmp/.X11-unix:rw"]
    for v in vols:
        cmd += ["--volume", v]
    cmd += [IMAGE, "bash", "-c", inner]
    return cmd


def _read_rows(paths) -> List[dict]:
    rows = []
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                out = {}
                for k, v in r.items():
                    try:
                        out[k] = float(v) if v not in ("True", "False") else (v == "True")
                    except (TypeError, ValueError):
                        out[k] = v
                rows.append(out)
    return rows


def run_deform(req, mode: str) -> List[dict]:
    bundle = prepare_bundle(req, mode)
    results = bundle / "results"
    results.mkdir(exist_ok=True)
    timeout_s = float(req.options.get("timeout_s") or (600.0 if mode == "native" else 300.0))
    map_name = bundle.name.rsplit("_", 1)[0]
    rows: List[dict] = []
    if not runtime():
        cmd = _docker_cmd(bundle, results, mode, timeout_s, "deform_ffbench", map_name, req.n)
        print(f"[deform] bundle generated at {bundle}")
        print("[deform] neither docker nor an apptainer image is available on this host.\n"
              "  Docker host: build the image with baselines/deform_docker (see ffbench/README.md).\n"
              "  Apptainer:   apptainer build --fakeroot ffbench_generated/deform_ros.sif ffbench/ros/deform_ros.def\n"
              "  Then run per trial:")
        print("  " + " ".join(f"'{c}'" if " " in c else c for c in cmd))
        raise RuntimeError("DEFORM needs Docker or an Apptainer image; bundle generated, nothing run")
    print(f"[deform] runtime: {runtime()}")
    for t in range(req.proto.trials):
        before = set(glob.glob(str(results / "zone_ep*.csv")))
        name = f"deform_ffbench_{req.n}_{t}_{os.getpid()}"
        cmd = _docker_cmd(bundle, results, mode, timeout_s, name, map_name, req.n)
        t0 = time.time()
        print(f"[deform] trial {t + 1}/{req.proto.trials} ({mode}) ...")
        log = results / f"trial_{t:02d}_{mode}.log"
        with open(log, "w") as lf:
            subprocess.run(cmd, check=False, stdout=lf, stderr=subprocess.STDOUT,
                           timeout=timeout_s + 120)
        print(f"[deform]   log: {log}")
        new = sorted(set(glob.glob(str(results / "zone_ep*.csv"))) - before)
        ended = ""
        try:
            m = re.search(r"Episode \d+ ended \u2014 ([^\n\[]+)", log.read_text(errors="ignore"))
            ended = m.group(1).strip() if m else ""
        except OSError:
            pass
        for r in _read_rows(new):
            if ended:                      # bridge mode logs the reason; the metrics node writes it itself
                r["terminated"] = ended
            r.update({"baseline": "deform", "sim": mode, "map": req.map_label,
                      "n_agents": req.n, "dynamics": req.dynamics if mode == "f1tenth" else "turtlebot3",
                      "target_speed": float("nan"), "seed": req.proto.seed + t, "trial": t,
                      "wall_time_s": round(time.time() - t0, 1), "bundle": str(bundle)})
            rows.append(r)
        if not new:
            print(f"[deform] trial {t + 1}: no CSV written (container failed or timed out)")
    return rows
