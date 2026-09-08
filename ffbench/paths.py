"""Repository layout, interpreters and import plumbing."""
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASELINES = ROOT / "baselines"
BASELINE_MAPS = BASELINES / "maps"
GYM_MAPS = ROOT / "f1tenth_gym" / "maps"
DEFORM_DOCKER = BASELINES / "deform_docker"
GENERATED = pathlib.Path(os.environ.get("FFBENCH_GENERATED", ROOT / "ffbench_generated"))
RESULTS = pathlib.Path(os.environ.get("FFBENCH_RESULTS", ROOT / "ffbench_results"))


def _rvo2_build() -> pathlib.Path:
    """The Python-RVO2 build directory for *this* interpreter (or the pip install)."""
    tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
    cands = sorted((BASELINES / "Python-RVO2" / "build").glob("lib.*"))
    for c in cands:
        if tag in c.name:
            return c
    return cands[0] if cands else BASELINES / "Python-RVO2" / "build" / "lib"


RVO2_BUILD = _rvo2_build()

# Interpreters for the external baselines.  Override per machine with
# FFBENCH_PY_FASTFUNNELS / FFBENCH_PY_GCBF; both default to the running one.
PY_FASTFUNNELS = pathlib.Path(os.environ.get("FFBENCH_PY_FASTFUNNELS", sys.executable))
PY_GCBF = pathlib.Path(os.environ.get("FFBENCH_PY_GCBF", sys.executable))

_DONE = False


def setup_sys_path() -> None:
    """Put the RVO2 build and ``baselines/`` ahead of the repo root.

    ``baselines/`` must precede the root because both ship an ``eval_metrics``
    module and only the baselines one is the 5-metric collector.
    """
    global _DONE
    if _DONE:
        return
    for p in (str(BASELINES), str(RVO2_BUILD)):
        while p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    _DONE = True


def stub_envs_f110() -> None:
    """Let ``baselines/orca.py`` and ``leader_follower.py`` import cheaply.

    Both import ``envs.f110_env`` at module load only to build their own gym
    adapter, and that import chain pulls in torch and the training stack.  The
    backends here drive the simulator themselves, so the adapter is stubbed
    unless the real module is already loaded.
    """
    if "envs.f110_env" in sys.modules:
        return
    pkg = sys.modules.get("envs")
    if pkg is None:
        pkg = types.ModuleType("envs")
        pkg.__path__ = [str(ROOT / "envs")]  # type: ignore[attr-defined]
        sys.modules["envs"] = pkg
    stub = types.ModuleType("envs.f110_env")
    stub.F110Config = object  # type: ignore[attr-defined]
    stub.F110EnvAdapter = object  # type: ignore[attr-defined]
    sys.modules["envs.f110_env"] = stub


def load_module_from_file(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
