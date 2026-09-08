"""ffbench -- one entry point for every FastFunnels baseline.

Each baseline can run on two backends:

* ``native``  -- the simulator the baseline was published with (RVO2 discs for
  ORCA, Gazebo + TurtleBot3 for DEFORM, the JAX/PyRoboSim world for GCBF+, the
  LAS multi-agent gym for LAS).
* ``f1tenth`` -- f1tenth_gym with the single-track (``st``) or kinematic (``ks``)
  vehicle model, the same plant FastFunnels itself is evaluated on.

Maps come from one source (``baselines/maps/<name>``) and are converted to
whatever each backend needs by :mod:`ffbench.maps`.  Every run is scored by the
same narrow-zone metrics (:mod:`ffbench.eval.metrics`).
"""
from ffbench.paths import setup_sys_path as _setup

_setup()
