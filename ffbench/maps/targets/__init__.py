from ffbench.maps.targets.f1tenth import write_f1tenth  # noqa: F401
from ffbench.maps.targets.rvo2 import rvo2_polygons, write_rvo2  # noqa: F401
from ffbench.maps.targets.ros import write_ros  # noqa: F401
from ffbench.maps.targets.gcbf import write_gcbf  # noqa: F401

TARGETS = {
    "f1tenth": write_f1tenth,
    "rvo2": write_rvo2,
    "ros": write_ros,
    "gcbf": write_gcbf,
}
