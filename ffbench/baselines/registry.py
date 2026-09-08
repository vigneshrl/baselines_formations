"""Baseline registry: what each baseline can do on each backend."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ffbench.eval.protocol import Protocol
from ffbench.maps.source import MapSource


@dataclass
class NativeRequest:
    src: MapSource
    map_label: str
    proto: Protocol
    n: int
    speed: float
    dynamics: str
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BaselineSpec:
    name: str
    description: str
    controller: Optional[Callable[[dict], Any]] = None     # f1tenth backend
    native: Optional[Callable[[NativeRequest], List[dict]]] = None
    external: Optional[Callable[[NativeRequest, str], List[dict]]] = None  # own runner, both modes
    formation: str = "abreast"
    control_hz: float = 100.0
    native_note: str = ""
    agent_counts: Optional[List[int]] = None


def _orca_ctrl(opts):
    from ffbench.baselines.orca_f1tenth import OrcaController
    return OrcaController(walls=bool(opts.get("orca_walls", True)))


def _orca_native(req):
    from ffbench.baselines.orca_native import run_orca_native
    return run_orca_native(req)


def _lf_ctrl(opts):
    from ffbench.baselines.leader_follower import LeaderFollowerController
    return LeaderFollowerController()


def _nmpc_ctrl(opts):
    from ffbench.baselines.nmpc import NMPCConfig, NMPCController
    cfg = NMPCConfig()
    if opts.get("controller_model"):
        cfg.model = opts["controller_model"]
    return NMPCController(cfg)


def _deform(req, mode):
    from ffbench.baselines.deform import run_deform
    return run_deform(req, mode)


def _gcbf(req, mode):
    from ffbench.baselines.external import run_gcbf
    return run_gcbf(req, mode)


def _las(req, mode):
    from ffbench.baselines.external import run_las
    return run_las(req, mode)


REGISTRY: Dict[str, BaselineSpec] = {
    "orca": BaselineSpec(
        "orca", "Optimal Reciprocal Collision Avoidance (Python-RVO2)",
        controller=_orca_ctrl, native=_orca_native, control_hz=100.0,
        native_note="RVO2 holonomic discs on the traced corridor polygon"),
    "leader_follower": BaselineSpec(
        "leader_follower", "Pure-pursuit convoy with PD gap control",
        controller=_lf_ctrl, formation="column",
        native_note="its native simulator is f1tenth_gym"),
    "nmpc": BaselineSpec(
        "nmpc", "Decentralised NMPC per agent (own lane, corridor + neighbour constraints, no funnel)",
        controller=_nmpc_ctrl, control_hz=20.0,
        native_note="its native simulator is f1tenth_gym"),
    "deform": BaselineSpec(
        "deform", "DEFORM formation planner (NeSC-IV), ROS Noetic in Docker",
        external=_deform, agent_counts=None,
        native_note="Gazebo + TurtleBot3 on a corridor rescaled to the TurtleBot radius"),
    "gcbf": BaselineSpec(
        "gcbf", "GCBF+ pretrained DubinsCar policy (MIT-REALM)",
        external=_gcbf,
        native_note="PyRoboSim room + JAX policy (gcbf_baseline.run_gcbf_eval)"),
    "las": BaselineSpec(
        "las", "Learning Adaptive Safety CBF-PPO (fixed 3 agents)",
        external=_las, agent_counts=[3],
        native_note="its own f110-multi-agent gym on the open_narrow_obs override"),
}

ALIASES = {"lf": "leader_follower", "mpc": "nmpc", "dmpc": "nmpc", "gcbfplus": "gcbf",
           "gcbf+": "gcbf"}


def get(name: str) -> BaselineSpec:
    key = ALIASES.get(name.lower(), name.lower())
    if key not in REGISTRY:
        raise KeyError(f"unknown baseline '{name}'. Known: {', '.join(REGISTRY)}")
    return REGISTRY[key]
