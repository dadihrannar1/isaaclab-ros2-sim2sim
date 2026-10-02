#!/usr/bin/env python3
"""
A1 locomotion policy with ROS 2 bridge.
Loads trained policy, subscribes to /cmd_vel, publishes /joint_states and /odom.

Run with:
    A1_POLICY_PATH=/path/to/model_XXX.pt \\
        common/launch_isaac_ros2.sh a1/a1_policy_ros2.py

Env vars (all have defaults for a standard Isaac Sim 5.1 + env_isaaclab install):
    ISAACLAB_VENV   - path to the Isaac Sim python venv (default: ~/env_isaaclab)
    ISAACLAB_PATH   - path to the IsaacLab repo checkout (default: ~/IsaacLab)
    ISAAC_PYTHON    - python version of that venv, e.g. python3.11 (default: python3.11)
    A1_POLICY_PATH  - required: path to the trained rsl_rl checkpoint (model_*.pt)
"""

import os
import sys

ISAACLAB_VENV = os.path.expanduser(os.environ.get("ISAACLAB_VENV", "~/env_isaaclab"))
ISAACLAB_PATH = os.path.expanduser(os.environ.get("ISAACLAB_PATH", "~/IsaacLab"))
ISAAC_PYTHON = os.environ.get("ISAAC_PYTHON", "python3.11")

POLICY_PATH = os.environ.get("A1_POLICY_PATH")
if not POLICY_PATH:
    raise RuntimeError(
        "Set A1_POLICY_PATH to a trained checkpoint, e.g.:\n"
        "  export A1_POLICY_PATH=~/IsaacLab/logs/rsl_rl/unitree_a1_flat/<timestamp>/model_299.pt\n"
        "See the README for the train.py command to produce one."
    )
POLICY_PATH = os.path.expanduser(POLICY_PATH)

# ROS 2 bridge env vars - must be set before SimulationApp
os.environ["ROS_DISTRO"] = "humble"
os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "") + \
    f":{ISAACLAB_VENV}/lib/{ISAAC_PYTHON}/site-packages/isaacsim/exts/isaacsim.ros2.bridge/humble/lib"
os.environ.pop("AMENT_PREFIX_PATH", None)

# Add Isaac Lab to path
sys.path.insert(0, f"{ISAACLAB_PATH}/source/isaaclab")
sys.path.insert(0, f"{ISAACLAB_PATH}/source/isaaclab_tasks")

from isaacsim import SimulationApp
app = SimulationApp({"headless": False})

import numpy as np
from scipy.spatial.transform import Rotation as R
import io
import omni
import omni.graph.core as og
import usdrt.Sdf
from omni.isaac.core import World
from omni.isaac.core.utils.extensions import enable_extension
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import UsdGeom, Gf


enable_extension("isaacsim.ros2.bridge")

import torch
import torch.nn as nn

# ── Rebuild actor network ──────────────────────────────────────────────────────
class ActorMLP(nn.Module):
    def __init__(self, obs_dim=48, act_dim=12, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden),  nn.ELU(),
            nn.Linear(hidden, hidden),  nn.ELU(),
            nn.Linear(hidden, act_dim),
        )
    def forward(self, x):
        return self.mlp(x)

ckpt = torch.load(POLICY_PATH, map_location="cpu")
policy = ActorMLP()
policy.load_state_dict(ckpt["actor_state_dict"], strict=False)
policy.eval()
print("Policy loaded. Iteration:", ckpt["iter"])
# ── Scene setup ────────────────────────────────────────────────────────────────
# physics_dt matches training (params/env.yaml: sim.dt=0.005, decimation=4 -> 50 Hz
# control). rendering_dt is set equal to physics_dt (not left at the 1/60 default):
# when they differ, world.step(render=True) internally runs int(rendering_dt/physics_dt)
# physics substeps per call (see SimulationContext.step()) instead of exactly one, which
# breaks the 1-call == 1-physics-tick assumption the main loop below relies on for
# reapplying torque every physics step. Keeping them equal makes every world.step() call
# advance exactly one deterministic 0.005s tick, whether it renders or not.
world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 200.0, rendering_dt=1.0 / 200.0)
world.scene.add_default_ground_plane()

# Must use the IsaacLab-specific USD — same as training; has correct joint order and actuator params
A1_USD = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Robots/Unitree/A1/a1.usd"
add_reference_to_stage(usd_path=A1_USD, prim_path="/World/A1")

stage = omni.usd.get_context().get_stage()
prim = stage.GetPrimAtPath("/World/A1")
xform = UsdGeom.Xformable(prim)
xform.ClearXformOpOrder()
xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.42))  # matches training init_state z=0.42

# ── ROS 2 OmniGraph bridge ─────────────────────────────────────────────────────
og.Controller.edit(
    {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
    {
        og.Controller.Keys.CREATE_NODES: [
            ("OnPlaybackTick",    "omni.graph.action.OnPlaybackTick"),
            ("ROS2Context",       "isaacsim.ros2.bridge.ROS2Context"),
            ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
            ("SubscribeTwist",    "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
            ("ComputeOdometry",   "isaacsim.core.nodes.IsaacComputeOdometry"),
            ("ReadSimTime",       "isaacsim.core.nodes.IsaacReadSimulationTime"),
            ("PublishOdometry",   "isaacsim.ros2.bridge.ROS2PublishOdometry"),
        ],
        og.Controller.Keys.CONNECT: [
            ("OnPlaybackTick.outputs:tick",        "PublishJointState.inputs:execIn"),
            ("OnPlaybackTick.outputs:tick",        "SubscribeTwist.inputs:execIn"),
            ("OnPlaybackTick.outputs:tick",        "ComputeOdometry.inputs:execIn"),
            ("ROS2Context.outputs:context",        "PublishJointState.inputs:context"),
            ("ROS2Context.outputs:context",        "SubscribeTwist.inputs:context"),
            ("ROS2Context.outputs:context",        "PublishOdometry.inputs:context"),
            ("ComputeOdometry.outputs:execOut",         "PublishOdometry.inputs:execIn"),
            ("ComputeOdometry.outputs:position",        "PublishOdometry.inputs:position"),
            ("ComputeOdometry.outputs:orientation",     "PublishOdometry.inputs:orientation"),
            ("ComputeOdometry.outputs:linearVelocity",  "PublishOdometry.inputs:linearVelocity"),
            ("ComputeOdometry.outputs:angularVelocity", "PublishOdometry.inputs:angularVelocity"),
            ("ReadSimTime.outputs:simulationTime",      "PublishOdometry.inputs:timeStamp"),
        ],
        og.Controller.Keys.SET_VALUES: [
            ("PublishJointState.inputs:targetPrim", ["/World/A1/trunk"]),
            ("PublishJointState.inputs:topicName",  "joint_states"),
            ("SubscribeTwist.inputs:topicName",     "cmd_vel"),
            ("ComputeOdometry.inputs:chassisPrim",  [usdrt.Sdf.Path("/World/A1/trunk")]),
            ("PublishOdometry.inputs:topicName",    "odom"),
        ],
    },
)

world.reset()

# ── Articulation handle ────────────────────────────────────────────────────────
from isaacsim.core.prims import SingleArticulation
robot = SingleArticulation(prim_path="/World/A1/trunk", name="a1")
robot.initialize()

# Determine the articulation's native joint order.
try:
    usd_joint_names = list(robot.joint_names)
except AttributeError:
    usd_joint_names = list(robot.dof_names)
print("USD joint order:", usd_joint_names)

# IsaacLab's JointPositionAction is configured with joint_names=['.*'], preserve_order=False
# (confirmed in params/env.yaml). When the pattern matches *all* joints and preserve_order is
# False, IsaacLab's find_joints() does NOT reorder by name — isaaclab/envs/mdp/actions/
# joint_actions.py falls back to `self._joint_ids = slice(None)`, i.e. the asset's native
# joint order. joint_pos_rel/joint_vel_rel observations use the same default full-asset
# ordering. So training order == USD native order; no permutation is needed or correct here
# (an earlier version of this script assumed a per-leg grouped order, which scrambled both
# the observations and the applied actions and caused the robot to jitter instead of stand).
JOINT_NAMES = usd_joint_names
joint_idx = list(range(12))
inv_joint_idx = list(range(12))

# Default joint positions from training init_state, matched by name (params/env.yaml):
#   .*L_hip_joint: 0.1        .*R_hip_joint: -0.1
#   F[L,R]_thigh_joint: 0.8   R[L,R]_thigh_joint: 1.0
#   .*_calf_joint: -1.5
def _default_pos(name: str) -> float:
    if name.endswith("hip_joint"):
        return 0.1 if name.startswith(("FL", "RL")) else -0.1
    if name.endswith("thigh_joint"):
        return 0.8 if name.startswith("F") else 1.0
    if name.endswith("calf_joint"):
        return -1.5
    raise ValueError(f"Unrecognized joint name: {name}")

DEFAULT_POS = torch.tensor([_default_pos(n) for n in JOINT_NAMES])

last_action = torch.zeros(12)
cmd_vel = [0.0, 0.0, 0.0]  # [vx, vy, yaw_rate]

# ── ROS 2 cmd_vel reader ───────────────────────────────────────────────────────
def get_cmd_vel():
    """Read the latest cmd_vel from the OmniGraph SubscribeTwist node."""
    try:
        linear = og.Controller.get(og.Controller.attribute("/ActionGraph/SubscribeTwist.outputs:linearVelocity"))
        angular_z = og.Controller.get(og.Controller.attribute("/ActionGraph/SubscribeTwist.outputs:angularVelocity"))[2]
        return [float(linear[0]), float(linear[1]), float(angular_z)]
    except Exception:
        return [0.0, 0.0, 0.0]

# ── PD torque control (replicates IsaacLab's DCMotor actuator model) ───────────
# IsaacLab USD uses DCMotor (software PD), so joint drive stiffness in the USD
# is 0; apply_action(joint_positions=...) does nothing. set_joint_efforts()
# only holds for a single physics frame, so torque must be recomputed from
# fresh joint state and reapplied every physics step (not just every policy
# step) — exactly like the real actuator model, which holds the *target*
# fixed across the decimation window but keeps running the PD loop every step.
KP = 25.0
KD = 0.5
EFFORT_LIMIT = 33.5

# Target joint positions (USD order), held fixed between policy updates.
target_pos_usd = DEFAULT_POS.numpy()[inv_joint_idx]

# ── Main loop ──────────────────────────────────────────────────────────────────
world.play()
print("Simulation running. Topics: /joint_states, /odom, /cmd_vel")

step = 0
POLICY_FREQ = 4  # run policy every 4 sim steps (matches decimation=4)

while app.is_running():
    # Apply PD torque every physics step, toward the current held-fixed target.
    joint_pos_raw = robot.get_joint_positions()
    joint_vel_raw = robot.get_joint_velocities()
    if joint_pos_raw is not None and joint_vel_raw is not None:
        effort = KP * (target_pos_usd - joint_pos_raw) - KD * joint_vel_raw
        effort = np.clip(effort, -EFFORT_LIMIT, EFFORT_LIMIT)
        robot.set_joint_efforts(effort)

    # Only render on the last physics tick of each decimation window (still ~50 fps,
    # matching the control rate) — physics_dt == rendering_dt so this doesn't change
    # how much sim time a step advances, only how often the viewport redraws.
    world.step(render=(step + 1) % POLICY_FREQ == 0)
    step += 1

    if step % POLICY_FREQ != 0:
        continue

    # Read cmd_vel from ROS 2
    cmd = get_cmd_vel()

    # Get robot state
    pos, orient = robot.get_world_pose()
    lin_vel_world = robot.get_linear_velocity()
    ang_vel_world = robot.get_angular_velocity()
    joint_pos_raw = robot.get_joint_positions()
    joint_vel_raw = robot.get_joint_velocities()

    if joint_pos_raw is None or joint_vel_raw is None:
        continue

    # Reorder joints from USD order to training order
    joint_pos = joint_pos_raw[joint_idx]
    joint_vel = joint_vel_raw[joint_idx]

    # Isaac Sim / Isaac Lab use (w, x, y, z) quaternion convention.
    # scipy.spatial.transform.Rotation.from_quat expects (x, y, z, w).
    quat = orient  # (w, x, y, z) from Isaac Sim
    rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # → (x, y, z, w) for scipy

    # Rotate world-frame velocities to body frame (matches training obs: root_lin_vel_b, root_ang_vel_b)
    lin_vel_body = rot.inv().apply(lin_vel_world)
    ang_vel_body = rot.inv().apply(ang_vel_world)

    # Gravity projected into body frame
    gravity_world = np.array([0.0, 0.0, -1.0])
    projected_gravity = rot.inv().apply(gravity_world)

    # Build observation vector (48 dims for flat env, no height scan)
    obs = np.concatenate([
        lin_vel_body,                                     # 3  base_lin_vel (body frame)
        ang_vel_body,                                     # 3  base_ang_vel (body frame)
        projected_gravity,                                # 3
        np.array(cmd),                                    # 3  velocity_commands
        joint_pos - DEFAULT_POS.numpy(),                  # 12 joint_pos_rel
        joint_vel,                                        # 12 joint_vel_rel
        last_action.numpy(),                              # 12 last_action
    ])

    # Run policy
    with torch.no_grad():
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
        action = policy(obs_tensor).squeeze(0)

    last_action = action.clone()

    if step % (POLICY_FREQ * 50) == 0:  # print every ~50 policy steps
        print(f"[step {step}] obs norm: {np.linalg.norm(obs):.3f} | "
              f"action: {np.round(action.numpy(), 3)}")

    # scale=0.25 confirmed in params/env.yaml: scale: 0.25
    target_pos_train = DEFAULT_POS + action * 0.25

    # Convert training order back to USD joint order; held fixed until next policy update
    target_pos_usd = target_pos_train.numpy()[inv_joint_idx]

app.close()
