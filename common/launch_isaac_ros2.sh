#!/bin/bash
# Launches a python script inside Isaac Sim's bundled environment with the ROS 2
# bridge env vars set correctly. Isaac Sim's Python/extensions and the system ROS 2
# install conflict on PYTHONPATH/AMENT_PREFIX_PATH, so this unsets the system ones
# first and points everything at Isaac Sim's own bundled libs.
#
# Usage:
#   ./launch_isaac_ros2.sh /path/to/script.py
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

export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$ISAACLAB_VENV/lib/$ISAAC_PYTHON/site-packages/isaacsim/exts/isaacsim.ros2.bridge/humble/lib"

python "$@"
