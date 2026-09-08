"""f1tenth_gym track directory: ``<name>_map.pgm/.yaml``, centreline, obs_pos."""
from __future__ import annotations

import pathlib
import shutil

import numpy as np
import yaml

from ffbench.maps.source import MapSource


def write_f1tenth(src: MapSource, out_root, force: bool = False) -> pathlib.Path:
    """Write ``out_root/<name>/`` and return it.

    An unscaled source that already has every file is returned as-is, so the
    standard maps are used in place and never duplicated.
    """
    if abs(src.scale - 1.0) < 1e-12 and not force:
        needed = [f"{src.name}_map.yaml", f"{src.name}_centerline.csv"]
        if all((src.map_dir / f).exists() for f in needed):
            return src.map_dir
    d = pathlib.Path(out_root) / src.name
    d.mkdir(parents=True, exist_ok=True)
    pgm = d / f"{src.name}_map.pgm"
    shutil.copyfile(src.image_path, pgm)
    with open(d / f"{src.name}_map.yaml", "w") as f:
        yaml.safe_dump({
            "image": pgm.name,
            "resolution": float(src.resolution),
            "origin": [float(src.origin[0]), float(src.origin[1]), 0.0],
            "negate": 0, "occupied_thresh": 0.45, "free_thresh": 0.196,
        }, f, sort_keys=False)
    rows = np.column_stack([src.centerline, src.centerline_widths])
    np.savetxt(d / f"{src.name}_centerline.csv", rows, delimiter=", ",
               header="x_m,y_m, w_tr_right_m, w_tr_left_m", comments="#", fmt="%.6f")
    with open(d / f"{src.name}_obs_pos.yaml", "w") as f:
        yaml.safe_dump(src.obs_pos_dict(), f, sort_keys=False)
    return d
