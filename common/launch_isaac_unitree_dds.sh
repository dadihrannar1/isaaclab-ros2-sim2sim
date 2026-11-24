#!/bin/bash
# Launches a python script inside the Isaac Sim venv with Unitree's DDS (CycloneDDS +
# unitree_sdk2_python) set up, for g1/g1_29dof_unitree_dds_bridge.py. Same idea as
# launch_isaac_ros2_unitree.sh, but for Unitree's own DDS bus instead of ROS 2.
#
# Usage:
#   ./launch_isaac_unitree_dds.sh /path/to/script.py
#
# Configure via env vars:
#   ISAACLAB_VENV    - Isaac Sim python venv (default: ~/env_isaaclab)
#   CYCLONEDDS_HOME  - CycloneDDS 0.10.x install that unitree_sdk2py was built against
#                      (default: ~/unitree_g1/cyclonedds/install)
set -e

unset AMENT_PREFIX_PATH
unset ROS_DISTRO
unset PYTHONPATH

ISAACLAB_VENV="${ISAACLAB_VENV:-$HOME/env_isaaclab}"
export CYCLONEDDS_HOME="${CYCLONEDDS_HOME:-$HOME/unitree_g1/cyclonedds/install}"

if [ ! -f "$ISAACLAB_VENV/bin/activate" ]; then
    echo "error: no venv found at $ISAACLAB_VENV (set ISAACLAB_VENV to override)" >&2
    exit 1
fi
if [ ! -d "$CYCLONEDDS_HOME/lib" ]; then
    echo "error: no CycloneDDS install at $CYCLONEDDS_HOME (set CYCLONEDDS_HOME to override)" >&2
    exit 1
fi

source "$ISAACLAB_VENV/bin/activate"
export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib:$LD_LIBRARY_PATH"

for mod in unitree_rl_lab unitree_sdk2py; do
    if ! python -c "import importlib.util, sys; sys.exit(importlib.util.find_spec('$mod') is None)"; then
        echo "error: python package '$mod' is not installed in $ISAACLAB_VENV." >&2
        exit 1
    fi
done

python "$@"
