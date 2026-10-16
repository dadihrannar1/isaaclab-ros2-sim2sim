# isaaclab-ros2-sim2sim

Deploys RL locomotion policies trained in [Isaac Lab](https://github.com/isaac-sim/IsaacLab) back into Isaac Sim through a ROS 2 bridge — a sim-to-sim validation step that exercises the same `/cmd_vel` → policy → joint torques → `/odom` loop a real robot deployment would use, without needing real hardware.

Covers two robots:
- **Unitree A1** — 12-DOF quadruped, flat-terrain velocity-tracking policy.
- **Unitree G1** — 37-DOF humanoid (legs + torso + arms + 3-finger hands), flat-terrain velocity-tracking policy.

## Repo layout

```
common/
  launch_isaac_ros2.sh   # env launcher (Isaac Sim venv + ROS2 bridge vars), robot-agnostic
  walk_forward.py        # ROS2 test client: publishes /cmd_vel, walks 1m, stops
a1/
  a1_policy_ros2.py       # A1 deployment script
g1/
  g1_policy_ros2.py       # G1 deployment script
```

## Prerequisites

- Ubuntu with ROS 2 Humble installed (`/opt/ros/humble`).
- Isaac Sim 5.1 installed into a Python venv (referred to below as `env_isaaclab`), with `isaacsim.ros2.bridge` available.
- An [Isaac Lab](https://github.com/isaac-sim/IsaacLab) checkout (used for its `isaaclab`/`isaaclab_tasks` packages and training scripts).

## Configuration

Both deployment scripts and the launcher read their paths from environment variables (all have sensible defaults except the policy path, which is required):

| Variable | Default | Meaning |
|---|---|---|
| `ISAACLAB_VENV` | `~/env_isaaclab` | Python venv Isaac Sim was installed into |
| `ISAACLAB_PATH` | `~/IsaacLab` | Isaac Lab repo checkout |
| `ISAAC_PYTHON` | `python3.11` | Python version of that venv |
| `A1_POLICY_PATH` | *(required for A1)* | Path to a trained rsl_rl checkpoint, e.g. `.../model_299.pt` |
| `G1_POLICY_PATH` | *(required for G1)* | Path to the **exported TorchScript policy** (`exported/policy.pt`), not the raw checkpoint |

Note the A1/G1 asymmetry: A1's script rebuilds the actor network by hand and loads a raw `rsl_rl` checkpoint's `actor_state_dict`. G1's script instead loads the TorchScript module that `play.py` auto-exports alongside the checkpoints (`<run_dir>/exported/policy.pt`) — this sidesteps having to hand-match hidden-layer sizes/activation for a much larger network, and is the pattern worth following for new robots.

## Training (to produce your own checkpoints)

Checkpoints aren't committed to this repo (binary, and large). Train your own:

```bash
cd $ISAACLAB_PATH && source $ISAACLAB_VENV/bin/activate

# A1 — flat terrain, ~300 iterations, a few minutes on a modern GPU
python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-Unitree-A1-v0 --headless

# G1 — flat terrain, ~1500 iterations, ~20-25 min on an RTX 4090
python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-G1-v0 --headless
# Or for custom training
python ~/isaaclab-ros2-sim2sim/rl_training/train.py --task My-Velocity-Flat-G1-v0 --headless
```

Checkpoints land in `$ISAACLAB_PATH/logs/rsl_rl/<experiment_name>/<timestamp>/`. For G1, also run `play.py` once to generate the exported TorchScript policy (it does this automatically as a side effect of loading a checkpoint for visualization):

```bash
python scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Flat-G1-Play-v0 --num_envs 4
```
which writes `<run_dir>/exported/policy.pt` (and `.onnx`).

## Running

Each robot needs **two terminals**: one running the Isaac Sim policy (blocks until closed), one sending `/cmd_vel` commands. Don't run ROS 2's `setup.bash` in the Isaac Sim terminal — the launcher deliberately unsets those vars to avoid a `PYTHONPATH`/`AMENT_PREFIX_PATH` conflict with Isaac Sim's bundled Python.

**A1:**
```bash
# Terminal 1
A1_POLICY_PATH=~/IsaacLab/logs/rsl_rl/unitree_a1_flat/<timestamp>/model_299.pt \
    common/launch_isaac_ros2.sh a1/a1_policy_ros2.py

# Terminal 2
source /opt/ros/humble/setup.bash
python3 common/walk_forward.py
```

**G1:**
```bash
# Terminal 1
G1_POLICY_PATH=~/IsaacLab/logs/rsl_rl/g1_flat/<timestamp>/exported/policy.pt \
    common/launch_isaac_ros2.sh g1/g1_policy_ros2.py

# Terminal 2
source /opt/ros/humble/setup.bash
python3 common/walk_forward.py
# Or for a random direction
source /opt/ros/humble/setup.bash
python3 common/walk_randomly.py
# Or for keyboard control
source /opt/ros/humble/setup.bash
python3 common/controller.py
```

Both publish `/joint_states` and `/odom`, and subscribe to `/cmd_vel`.

## Lessons learned (worth reading before adapting this to a new robot)

These cost real debugging time and aren't obvious from the Isaac Lab/Isaac Sim docs:

1. **Joint order is native-articulation order, not "the order you'd guess."** Isaac Lab's `JointPositionAction` with `joint_names=['.*']` and `preserve_order=False` resolves to the asset's raw native joint order (`asset.find_joints` falls back to `slice(None)`) — for G1 this is grouped by joint *type* across both legs (all hips, then all thighs/knees, then all ankles, ...), not grouped per-leg like you might assume. Build any per-joint tables (default pose, gains, limits) keyed by **joint name**, read at runtime from the live articulation, never by a hardcoded position list.

2. **`physics_dt` must equal `rendering_dt`.** If they differ, `world.step(render=True)` silently runs `int(rendering_dt/physics_dt)` physics substeps per call instead of exactly one (see `SimulationContext.step()`), which desyncs your control-rate math and breaks any "recompute torque every physics tick" loop. Set them equal, and only pass `render=True` on the last tick of each decimation window if you want to reduce render overhead.

3. **Effort control (`set_joint_efforts`) only works if the USD's native joint drive is actually disabled.** It isn't always. A1's USD ships with zero native stiffness/damping, so manually computing PD torque and calling `set_joint_efforts()` works as intended. G1's USD (`g1_minimal.usd`) ships with a **live native PD drive already baked in** (non-zero stiffness on every joint) — left alone, it fights the manually-computed torque the whole time, which looks like "the legs barely move." Fix: `robot._articulation_view.switch_control_mode("effort")` right after `initialize()`, which zeroes the native gains and gives your own control loop full authority. Verify empirically if in doubt: apply a large one-sided effort to a single named joint and confirm it actually moves by the expected amount.

4. **`last_action` in the observation is the raw, pre-scale/offset policy output**, not the processed target position (confirmed from `isaaclab/managers/action_manager.py`: `action_manager.action` stores the raw input before any `JointAction.process_actions` scaling). Store exactly what the network produced, before applying `scale`/`offset`.

## License

MIT, see [LICENSE](LICENSE).
