#!/bin/bash
# Launches a python script inside Isaac Sim's bundled environment with the ROS 2
# bridge env vars set correctly. Same as launch_isaac_ros2.sh, but for policies
# trained with unitree_rl_lab (e.g. Unitree-G1-29dof-Velocity): it additionally
# checks that the unitree_rl_lab package is installed in the venv, since the
# policy script imports the robot config from it (to spawn the exact same URDF
# robot as in training).
#
# Usage:
#   ./launch_isaac_ros2_unitree.sh /path/to/script.py
#
# Configure via env vars (both have sensible defaults for a standard Isaac Sim 5.1 +
# env_isaaclab venv install):
#   ISAACLAB_VENV   - path to the python venv Isaac Sim was installed into (default: ~/env_isaaclab)
#   ISAAC_PYTHON    - python version used by that venv, e.g. python3.11 (default: python3.11)
set -e

unset AMENT_PREFIX_PATH
unset ROS_DISTRO
unset PYTHONPATH

ISAACLAB_VENV="${ISAACLAB_VENV:-$HOME/env_isaaclab}"
ISAAC_PYTHON="${ISAAC_PYTHON:-python3.11}"

if [ ! -f "$ISAACLAB_VENV/bin/activate" ]; then
    echo "error: no venv found at $ISAACLAB_VENV (set ISAACLAB_VENV to override)" >&2
    exit 1
fi

source "$ISAACLAB_VENV/bin/activate"

if ! python -c "import importlib.util, sys; sys.exit(importlib.util.find_spec('unitree_rl_lab') is None)"; then
    echo "error: unitree_rl_lab is not installed in $ISAACLAB_VENV." >&2
    echo "       Install it with: pip install -e <unitree_rl_lab>/source/unitree_rl_lab" >&2
    exit 1
fi

export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$ISAACLAB_VENV/lib/$ISAAC_PYTHON/site-packages/isaacsim/exts/isaacsim.ros2.bridge/humble/lib"

python "$@"
