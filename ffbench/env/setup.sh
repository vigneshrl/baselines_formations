#!/usr/bin/env bash
# One-shot environment setup for ffbench (Linux).  Run from the repo root:
#     bash ffbench/env/setup.sh            # creates conda env "ffbench"
#     bash ffbench/env/setup.sh --gcbf     # also creates "ffbench-gcbf" for the GCBF+ baseline
#     bash ffbench/env/setup.sh --las      # also creates "ffbench-las" for the LAS baseline
set -euo pipefail
cd "$(dirname "$0")/../.."
CONDA=${CONDA_EXE:-conda}
command -v mamba >/dev/null 2>&1 && CONDA=mamba

echo "[setup] creating conda env 'ffbench' from ffbench/env/environment.yaml"
$CONDA env create -f ffbench/env/environment.yaml -n ffbench || \
$CONDA env update -f ffbench/env/environment.yaml -n ffbench
PY=$($CONDA run -n ffbench python -c 'import sys; print(sys.executable)')

echo "[setup] installing the vendored f1tenth_gym (no dependency resolution: its pyproject pins an older gymnasium)"
"$PY" -m pip install --no-deps -e ./f1tenth_gym

echo "[setup] building Python-RVO2 (ORCA) against $PY"
( cd baselines/Python-RVO2 && "$PY" -m pip install cython && "$PY" setup.py build )
"$PY" -c "import sys, glob; sys.path.insert(0, glob.glob('baselines/Python-RVO2/build/lib.*')[0]); import rvo2; print('rvo2 ok')"

if [[ "${1:-}" == "--gcbf" ]]; then
  echo "[setup] creating conda env 'ffbench-gcbf' from ffbench/env/environment-gcbf.yaml"
  $CONDA env create -f ffbench/env/environment-gcbf.yaml -n ffbench-gcbf || \
  $CONDA env update -f ffbench/env/environment-gcbf.yaml -n ffbench-gcbf
  GPY=$($CONDA run -n ffbench-gcbf python -c 'import sys; print(sys.executable)')
  "$GPY" -m pip install --no-deps -e ./baselines/gcbfplus
  echo "[setup] add to your shell:  export FFBENCH_PY_GCBF=$GPY"
fi

if [[ "${1:-}" == "--las" ]]; then
  echo "[setup] creating conda env 'ffbench-las' (LAS baseline: forked gym + omnisafe)"
  $CONDA env create -f ffbench/env/environment-las.yaml -n ffbench-las || \
  $CONDA env update -f ffbench/env/environment-las.yaml -n ffbench-las
  LPY=$($CONDA run -n ffbench-las python -c 'import sys; print(sys.executable)')
  "$LPY" -m pip install --index-url https://download.pytorch.org/whl/cpu "torch~=2.0.1"
  "$LPY" -m pip install "pip<24.1" "setuptools==65.7.0" "wheel<0.41"
  "$LPY" -m pip install --no-deps --no-build-isolation "gym==0.19.0"
  "$LPY" -m pip install --no-deps "git+https://github.com/luigiberducci/f1tenth_gym.git@asrl-submission-refactoring#egg=f110_gym"
  "$LPY" -m pip install "git+https://github.com/luigiberducci/omnisafe.git@asrl-submission#egg=omnisafe" "pyglet<2"
  echo "[setup] add to your shell:  export FFBENCH_PY_FASTFUNNELS=$LPY   (used by --las)"
fi

echo "[setup] smoke test"
QT_QPA_PLATFORM=offscreen "$PY" run_experiment.py --orca --num_agents 2 --map standard_ON --trials 1 --out /tmp/ffbench_smoke
echo "[setup] done. Activate with: conda activate ffbench"
