# ffbench

One command line for every FastFunnels baseline, on the simulator it was
published with **or** on the f1tenth_gym plant, from one map source, scored by
one metrics class.

```bash
conda activate ffbench
python run_experiment.py --orca   --num_agents=4 --map=standard_ON                  # f1tenth plant, single-track model
python run_experiment.py --orca   --num_agents=4 --map=standard_ON --sim=native     # pure RVO2 discs
python run_experiment.py --nmpc   --num_agents=4 --map=standard_ON --dynamics=ks    # kinematic plant
python run_experiment.py --deform --num_agents=4 --map=standard_ON --sim=native     # Gazebo + TurtleBot3, in Docker
python run_experiment.py --gcbf   --num_agents=4 --map=standard_ON                  # GCBF+ cross-over (needs the gcbf env)
python run_experiment.py --list                                                     # baselines, maps, splits
```

Every run writes `ffbench_results/<baseline>_<map>_n<N>_<sim>.jsonl` (one row
per trial), the aggregated `.csv`, and prints the paper table
(V_bar, eSR, T_zone, flow, deformability, aSR).

---

## 1. Installation

### 1.1 Python (conda)

```bash
git clone <this repo> FastFunnels && cd FastFunnels
bash ffbench/env/setup.sh          # creates conda env "ffbench", builds Python-RVO2, runs a smoke test
conda activate ffbench
```

`setup.sh` does three things you can also do by hand:

| step | command |
|---|---|
| conda env | `conda env create -f ffbench/env/environment.yaml` (Python 3.9; `environment-full.yaml` is the exact pip freeze of the reference machine, torch and CUDA wheels included) |
| f1tenth_gym | `pip install --no-deps -e ./f1tenth_gym` (its pyproject pins gymnasium 0.29; the code runs on the pinned 1.1.1) |
| RVO2 (ORCA) | `cd baselines/Python-RVO2 && python setup.py build` (needs cmake and a C++ compiler; the env file installs both). ffbench finds `build/lib.*` for the running interpreter automatically. |

Headless machines: `export QT_QPA_PLATFORM=offscreen` (run_experiment sets it if unset).

### 1.2 GCBF+ (optional, separate env)

GCBF+ needs JAX 0.4.30 and pyrobosim, which do not coexist with the main env.

```bash
bash ffbench/env/setup.sh --gcbf                   # or: conda env create -f ffbench/env/environment-gcbf.yaml
export FFBENCH_PY_GCBF=$(conda run -n ffbench-gcbf python -c 'import sys;print(sys.executable)')
```

`--gcbf` then re-launches the GCBF+ runner under that interpreter. Without the
variable it runs under the current one and fails on the JAX import.

### 1.3 ROS / Gazebo (DEFORM) — container

Everything ROS lives in a ROS Noetic container (Gazebo 11, CasADi/IPOPT, the
DEFORM planner from the pinned `DEFORM/` submodule, the f1tenth bridge
package, DEFORM's own TurtleBot3 + RealSense simulation). Build it once, from
the repo root, after `git submodule update --init DEFORM`:

```bash
# Docker
docker build -t deform_ros:latest -f baselines/deform_docker/Dockerfile .
docker tag deform_ros:latest deform_ros:f1tenth_patched

# Apptainer (HPC nodes, no root). The Ubuntu 20.04 base needs the
# user-namespace fakeroot mode; the def disables apt's sandbox user for it.
export APPTAINER_CACHEDIR=/big/disk/apptainer_cache APPTAINER_TMPDIR=/big/disk/apptainer_tmp
apptainer build --fakeroot --ignore-fakeroot-command ffbench_generated/deform_ros.sif ffbench/ros/deform_ros.def
bash ffbench/ros/smoke_test_sif.sh          # packages, CasADi, gym import, TurtleBot3 sim
```

`run_experiment.py --deform` then drives one container per trial (Docker if
present, else the SIF; `FFBENCH_DEFORM_SIF` to point elsewhere), each on its
own ROS master port, and reads the CSV the episode writes. Without either
runtime it generates the bundle and prints the command to run elsewhere.

If your checkout of `f1tenth_gym/` contains Python cache files you cannot
read (shared machines), stage a clean copy first:
`rsync -a --exclude __pycache__ f1tenth_gym/ ffbench_generated/build_ctx/f1tenth_gym/`
and point the `%files` line of the def at it.

### 1.4 LAS (optional, separate env)

`--las` calls `baselines/las_sweep.py`, which needs the LAS authors' forks of
f1tenth_gym (`f110_gym`, pinned to gym 0.19) and omnisafe, pydantic, torch.
They do not coexist with the main env:

```bash
bash ffbench/env/setup.sh --las                    # env "ffbench-las" (Python 3.8, CPU torch)
export FFBENCH_PY_FASTFUNNELS=$(conda run -n ffbench-las python -c 'import sys;print(sys.executable)')
python run_experiment.py --las --num_agents 3 --map standard_ON --trials 20
```

The install order in `setup.sh` matters (old pip/setuptools for gym 0.19, the
forks with `--no-deps`); `environment-las.yaml` documents it.

---

## 2. What a run does

1. **Map** — `--map` resolves an alias (`standard_ON`, `extended_ON`, `lshape`,
   `zigzag`, `slalom`), a layout split (`eval_matched[:N]`, `eval_heldout[:N]`),
   or a directory in the FastFunnels map format. `MapSource` loads the PGM,
   centreline and obstacle sidecar once and traces the corridor polygon.
2. **Protocol** — N agents spawn in a rank 7 m before the pinch (the repo's
   original convention, i.e. inside the scored zone), facing down the corridor;
   the convoy baseline spawns as a column along the track. The rank is only
   shifted sideways if a wall or obstacle is within 0.5 m (`--spawn_up_m` to change). A straight approach segment
   is spliced into the centreline through the zone. The scored zone is 12 m
   either side of the pinch.
3. **Backend** — `--sim f1tenth` builds the f1tenth_gym env with the chosen
   vehicle model and steps the baseline's controller through it; `--sim native`
   hands the same spawn and reference to the baseline's own simulator.
4. **Metrics** — `baselines/eval_metrics.py::FullRunMetrics`, plus a
   `deformability_multi` column (tightest spread sampled only while ≥2 agents
   are in the zone). A trial ends when every agent has cleared the zone, the
   plant reports a collision, or `--max_steps` elapse.
5. **Report** — JSONL rows, aggregated CSV (NaN-ignoring means; `inf` survives
   only when every trial timed out), printed table.

---

## 3. Baselines

| flag | on the f1tenth plant (`--sim f1tenth`) | native (`--sim native`) |
|---|---|---|
| `--orca` | `baselines/orca.py` control law: lane following → RVO2 velocity → pure-pursuit steering. RVO2 sees the corridor walls and obstacles (`--orca_no_walls` hides them). | RVO2 integrates holonomic discs itself. |
| `--leader_follower` | `baselines/leader_follower.py`: pure-pursuit leader, PD gap followers, column spawn. | its native simulator is f1tenth_gym |
| `--nmpc` | decentralised NMPC per agent (`ffbench/baselines/nmpc.py`): own-lane reference, corridor half-plane constraints from the traced polygon, hard slacked keep-out from the other agents, kinematic or single-track prediction model (`--controller_model st`), 20 Hz. No funnel, no leader. | its native simulator is f1tenth_gym |
| `--deform` | DEFORM's planner driving the f1tenth plant through the ROS bridge (`--dynamics` is passed to the bridge). | Gazebo + TurtleBot3 on a world generated from the map, corridor rescaled by `robot_radius / 0.29` (`--robot_radius`, `--robot_model`). |
| `--gcbf` | `gcbf_baseline.run_gcbf_f110` (Dubins yaw-rate/accel → steer/speed). | `gcbf_baseline.run_gcbf_eval` (PyRoboSim room + JAX policy). |
| `--las` | `baselines/las_sweep.py`, fixed 3 agents on its own `open_narrow_obs` override; runs under `FFBENCH_PY_FASTFUNNELS` (the `ffbench-las` env). | same |

`deform`, `gcbf` and `las` are *external*: they keep their own runner and
ffbench prepares inputs and normalises their rows. GCBF+ rollouts are
deterministic per map, so `--trials` is ignored there; LAS returns the mean of
its episodes as one row.

Videos and plots: `--record` writes `<out>_<sim>_t<trial>.mp4` for the first
`--record_trials` trials (the gym renderer for f1tenth runs, a trace animation
for native runs); `--save_traces` stores positions for
`python -m ffbench.eval.plot_traces <out>` and `python -m ffbench.eval.animate_traces`.

Useful flags: `--speed` (target m/s, default 5), `--trials` (default 20),
`--seed`, `--spawn_jitter 0.1` (per-seed pose jitter; without it a
deterministic baseline's trials are identical), `--formation abreast|column`,
`--zone_half_m`, `--render`, `--out`.

---

## 4. Layout

```
run_experiment.py            CLI
ffbench/
  paths.py                   repo layout; FFBENCH_PY_GCBF / FFBENCH_PY_FASTFUNNELS / FFBENCH_GENERATED / FFBENCH_RESULTS overrides
  maps/
    source.py                MapSource: one map, corridor polygon, rescaling, both obstacle YAML schemas
    registry.py              aliases and splits
    generate.py              python -m ffbench.maps.generate --map standard_ON --target f1tenth rvo2 ros gcbf [--robot-radius 0.22]
    targets/                 f1tenth track dir · RVO2 polygons · ROS bundle (map_server, Gazebo world, DEFORM configs, launch files) · GCBF+ rectangles
  models/f1tenth.py          the plant (st | ks) and command translators (twist, holonomic velocity, Dubins → steer/speed)
  eval/                      protocol.py (spawn/reference/zone) · runner.py (trial loop) · metrics.py · report.py
  baselines/                 registry.py + one adapter per baseline (orca_f1tenth, orca_native, leader_follower, nmpc, deform, external)
  ros/ffbench_ros/           metrics node for Gazebo runs (copied into every ROS bundle)
  env/                       environment.yaml (minimal) · environment-full.yaml (exact freeze) · environment-gcbf.yaml · setup.sh
ffbench_generated/           derived artefacts, regenerated on demand, gitignored (see below)
ffbench_results/             JSONL + CSV outputs, gitignored
```

### `ffbench_generated/`

Nothing in it is hand-edited. It holds:

- `f1tenth/<map>/` — track directories for *rescaled* maps only (unscaled
  maps are used in place from `baselines/maps/`);
- `rvo2/<map>/<map>_rvo2.json` — wall and obstacle polygons for ORCA native;
- `ros/<map>_<N>agents/` — the DEFORM bundle: `map/` (map_server), `worlds/`
  (Gazebo SDF), `config/` (bridge params, formation, metrics), `launch/`,
  `track/` (f1tenth format for the bridge), `ffbench_ros/` (metrics node),
  `results/` (CSVs written by the container);
- `gcbf/<map>/` — wall rectangles;
- `external/` — raw JSONL/CSV from the GCBF+ and LAS runners before
  normalisation.

Delete the directory freely; every run recreates what it needs.

---

## 5. Adding a baseline

1. Write a controller with `reset(ctx, obs)` and `act(obs, step) -> (n, 2)`
   array of `[steering_angle, speed]` (see `ffbench/baselines/orca_f1tenth.py`
   for wrapping an existing script without running its `__init__`).
2. If it has a native simulator, write `run_<name>_native(req) -> list[row]`
   that builds the reference with `build_reference(...)` and feeds
   `ZoneMetrics` an obs dict with `poses_x, poses_y, poses_theta,
   linear_vels_x` (see `orca_native.py`).
3. Register it in `ffbench/baselines/registry.py`; the CLI flag appears
   automatically.

Command translators for other command types live in
`ffbench/models/f1tenth.py::translators`.

---

## 6. Status (verified 2026-09-07)

Every row below was produced by `run_experiment.py` on the reference machine,
standard_ON, target speed 5 m/s, single-track plant, 5 trials with
`--spawn_jitter 0.1` (merged table: `ffbench_results/verify_all.csv`,
trajectory plots: `ffbench_results/verify_*_traces.png`).

| baseline | sim | N | eSR | V_bar m/s | T_zone s | note |
|---|---|---|---|---|---|---|
| orca | f1tenth | 1 / 2 / 4 | 1.0 / 1.0 / 1.0 | 4.3 / 4.2 / 4.0 | 4.6 / 4.9 / 5.3 | rank squeezes to single file through the pinch |
| orca | native (RVO2) | 1 / 2 / 4 | 1.0 / 1.0 / 0.8 | 4.6 / 4.6 / 4.5 | 4.3 / 4.4 / 4.5 | one 4-disc trial brushes at 0.5 m |
| leader_follower | f1tenth | 1 / 2 / 4 | 1.0 / 1.0 / 1.0 | 3.2 / 3.3 / 3.3 | 6.0 / 6.5 / 7.9 | convoy spawned along the track bend |
| nmpc | f1tenth | 1 / 2 / 4 | 1.0 / 1.0 / 1.0 | 4.6 / 4.4 / 4.3 | 4.3 / 4.6 / 4.9 | kinematic prediction model, 20 Hz |
| gcbf | f1tenth | 4 | 0.0 | 0.6 | inf | 2 of 4 agents through, the rest stall (deterministic, 1 run) |
| gcbf | native (PyRoboSim) | 4 | 0.0 | 0.7 | inf | 3 of 4 agents through, matches the GCBF+ README failure mode |
| las | f1tenth (own env) | 3 | 0.2 | 0.74 | 14.3 | 20 episodes; reproduces the old `las_sweep_results.csv` exactly. Runs on its own `open_narrow_obs` override (pinch at (40.9, -41.3), 4 m zone), not the standard_ON corridor. The NPC `vgain` option has no effect on the result. |
| deform | f1tenth (ROS bridge) | 4 | 0.0 | 0.09 | inf | 3 trials in the Apptainer container: 2 end in a collision after ~15 s, 1 times out at 240 s; consistent with the old `deform_sweep_results.csv` |
| deform | native (Gazebo + TurtleBot3) | 4 | 0.0 | 0.03 | inf | 2 trials, 600 s each, real map scale: the formation creeps toward the pinch at 2-4 cm/s and never enters it; no collisions. A corridor rescaled to the TurtleBot body (0.36x) is worse: the planner keeps "reducing formation" and stands still |

Things learned while verifying, all now built into the protocol:

- A rank spawned 14 m before the pinch sits beside the wall-hugging disc at
  the funnel mouth, and shifting the rank off its lanes to avoid it makes
  every lane-following baseline steer into its neighbour. The spawn is 7 m
  before the pinch, the repo's original convention, with a sideways shift only
  when something is within 0.5 m.
- Spawn jitter is along-track and heading only; lateral jitter eats the rank gap.
- The NMPC hard keep-out must be below the rank gap (0.5 m vs 0.6 m), otherwise
  the solver starts infeasible and the cars yaw into each other on step one.
- The convoy spawns along the reference path, because a straight column
  extended upstream leaves the corridor where the track bends.
- The raw deformability metric returns NaN whenever the last agent exits
  alone; the reported column samples the tightest spread only while at least
  two agents are in the zone, floored at one car width.

All six baselines now have at least three end-to-end confirmations on the
reference machine; DEFORM's ran in the Apptainer build of the ROS container
(`ffbench/ros/deform_ros.def`), so the Docker path itself is untested here
though it uses the same bundle, bind mounts and launch files.

Known protocol differences from the old `baselines/sweep_runner.py` tables:
RVO2 sees the walls, and the original waypoints running alongside the spliced
approach are dropped (the old scoring latched onto them and reported roughly
40 % of the true zone time). Under the new protocol ORCA with 4 agents clears
standard_ON.
