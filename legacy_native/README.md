# legacy_native — the earlier hand-built scenes

Kept for reference; **not used by the benchmark** (`run_experiment.py`), which
generates its Gazebo worlds, PyRoboSim rooms and RVO2 polygons from the shared
maps in `baselines/maps/` instead.

| folder | what it is |
|---|---|
| `Gazebo_worlds/` | eight hand-authored Gazebo scenes (static boxes, moving obstacles, dense fields, `engineers_way`), see its `readme.md` |
| `models/` | the four TurtleBot3 burger models with per-robot namespaces used by those scenes |
| `Pyrobosim_2D_envs/` | the ROS1 and ROS2 PyRoboSim 2-D setups the first ORCA/DEFORM comparisons ran in |

Demo videos of these scenes: [Google Drive](https://drive.google.com/drive/folders/1KrD17Asrr-kUL6zi8UAPdhaNJMniDBb9?usp=sharing).
