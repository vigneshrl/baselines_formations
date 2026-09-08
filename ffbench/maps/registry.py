"""Map aliases (``standard_ON`` ...) and layout splits (``eval_matched`` ...)."""
from __future__ import annotations

import pathlib
from typing import List, Tuple

from ffbench.paths import BASELINE_MAPS, ROOT

ALIASES = {
    "standard_ON": BASELINE_MAPS / "open_narrow_obs",
    "extended_ON": BASELINE_MAPS / "extended_open_narrow_obs",
    "lshape": BASELINE_MAPS / "lshape_obs",
    "zigzag": BASELINE_MAPS / "zigzag_obs",
    "slalom": BASELINE_MAPS / "slalom_obs",
}
# lower-case lookups so --map standard_on works too
_ALIAS_LC = {k.lower(): v for k, v in ALIASES.items()}

SPLITS = {
    "eval_matched": ROOT / "maps" / "eval_maps_matched",
    "eval_heldout": ROOT / "maps" / "eval_maps_heldout",
}


def resolve(spec: str) -> List[Tuple[str, pathlib.Path]]:
    """``standard_ON`` | ``eval_matched[:N]`` | ``/path/to/map_dir`` -> [(label, dir)]."""
    key, _, count = spec.partition(":")
    if key.lower() in _ALIAS_LC:
        return [(key, _ALIAS_LC[key.lower()])]
    if key in SPLITS:
        root = SPLITS[key]
        dirs = sorted(p for p in root.iterdir() if p.is_dir())
        if count:
            dirs = dirs[: int(count)]
        return [(p.name, p) for p in dirs]
    p = pathlib.Path(spec).expanduser()
    if p.is_dir():
        return [(p.name, p.resolve())]
    raise KeyError(
        f"unknown map '{spec}'. Aliases: {', '.join(ALIASES)}; splits: "
        f"{', '.join(SPLITS)}; or a map directory path.")


def describe() -> str:
    lines = ["aliases:"]
    for k, v in ALIASES.items():
        lines.append(f"  {k:14s} -> {v}")
    lines.append("splits (append :N to take the first N maps):")
    for k, v in SPLITS.items():
        lines.append(f"  {k:14s} -> {v}")
    return "\n".join(lines)
