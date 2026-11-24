#!/usr/bin/env python3
"""
G1 29-DoF "fake robot" in Isaac Sim for testing unitree_rl_lab's real deploy controller (g1_ctrl).

This is the Isaac Sim counterpart of unitree_mujoco: instead of running the policy itself (like
g1_29dof_policy_ros2.py does), it pretends to be the real G1 on Unitree's DDS bus:
  - publishes   rt/lowstate  (joint q/dq/tau, pelvis IMU, wireless remote) every physics step
  - subscribes  rt/lowcmd    and applies  tau = tau_ff + kp*(q_des - q) + kd*(dq_des - dq)
                             per motor every physics step, like the motor drivers do
So g1_ctrl runs completely unchanged, with the same config.yaml, policy.onnx and deploy.yaml that
go on the robot. This tests the whole deploy chain (joint order, gains, obs building, ONNX export,
FSM) - not just the policy.

Run (two terminals):
    1)  cd ~/isaaclab-ros2-sim2sim
        common/launch_isaac_unitree_dds.sh g1/g1_29dof_unitree_dds_bridge.py
    2)  cd ~/unitree_g1/unitree_rl_lab/deploy/robots/g1_29dof/build
        ./g1_ctrl --network lo

Keyboard (click the Isaac Sim viewport first so it has focus) - replaces the remote:
    1           "LT + Up"   -> FixStand        (stand up)
    2           "RB + X"    -> Velocity        (start policy)
    0           "LT + B"    -> Passive         (STOP: limp + damped)
    W / S       left stick up/down     -> forward / backward   (vx)
    A / D       left stick left/right  -> sideways              (vy)
    Q / E       right stick left/right -> turn left / right     (yaw)
    G           virtual gantry on/off  (starts ON, like hanging on the real gantry)
    UP / DOWN   raise / lower the gantry by 2 cm
Sticks give +-0.5 when held; g1_ctrl clamps them to the command ranges in deploy.yaml.

Same procedure as on the real robot: wait for limp -> 1 (stand) -> DOWN x4 (lower the gantry until
the feet are flat on the ground; FixStand only holds a pose, it can't balance) -> 2 (policy) ->
G (gantry off) -> W etc. Press 0 to stop at any time.

Env vars:
    ISAACLAB_PATH      - IsaacLab checkout (default: ~/IsaacLab)
    G1_DDS_INTERFACE   - network interface for DDS (default: lo, must match g1_ctrl --network)
    G1_DDS_AUTOTEST=1  - no keyboard needed: runs stand -> gantry off -> policy -> walk -> turn
                         by itself and prints the pelvis pose (for headless checks)
    G1_HEADLESS=1      - run without a window (only useful with G1_DDS_AUTOTEST=1)
"""

import os
import sys
import struct
import threading
import time

ISAACLAB_PATH = os.path.expanduser(os.environ.get("ISAACLAB_PATH", "~/IsaacLab"))
DDS_INTERFACE = os.environ.get("G1_DDS_INTERFACE", "lo")
AUTOTEST = os.environ.get("G1_DDS_AUTOTEST", "0") == "1"
HEADLESS = os.environ.get("G1_HEADLESS", "0") == "1"

sys.path.insert(0, f"{ISAACLAB_PATH}/source/isaaclab")
sys.path.insert(0, f"{ISAACLAB_PATH}/source/isaaclab_tasks")

from isaacsim import SimulationApp
# Isaac Lab's app config, same reason as in g1_29dof_policy_ros2.py (URDF importer version).
app = SimulationApp({"headless": HEADLESS}, experience=f"{ISAACLAB_PATH}/apps/isaaclab.python.kit")

import numpy as np
from scipy.spatial.transform import Rotation as R
import carb
import omni
import omni.appwindow
from isaacsim.core.api import World
from pxr import Usd, UsdPhysics

# Unitree's Python SDK (unitree_sdk2_python) - talks the same DDS as the robot and g1_ctrl.
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.utils.crc import CRC

# ── Scene setup (identical to g1_29dof_policy_ros2.py) ─────────────────────────────
PHYSICS_DT = 1.0 / 200.0
world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT, rendering_dt=PHYSICS_DT)
world.scene.add_default_ground_plane(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG

G1_PRIM = "/World/G1"
SPAWN_Z = 0.8
spawn_cfg = UNITREE_G1_29DOF_CFG.spawn
spawn_cfg.func(G1_PRIM, spawn_cfg, translation=(0.0, 0.0, SPAWN_Z))

stage = omni.usd.get_context().get_stage()
G1_ROOT_PRIM = None
for p in Usd.PrimRange(stage.GetPrimAtPath(G1_PRIM)):
    if p.HasAPI(UsdPhysics.ArticulationRootAPI):
        G1_ROOT_PRIM = str(p.GetPath())
        break
if G1_ROOT_PRIM is None:
    raise RuntimeError(f"No ArticulationRootAPI found under {G1_PRIM} - URDF import failed?")
print("Articulation root:", G1_ROOT_PRIM)  # the pelvis link = where the G1's IMU sits

world.reset()

from isaacsim.core.prims import SingleArticulation, RigidPrim
robot = SingleArticulation(prim_path=G1_ROOT_PRIM, name="g1")
robot.initialize()
# Motor commands go into PhysX's implicit joint drives (stiffness=kp, damping=kd, target=q/dq,
# plus tau as a feed-forward joint force) - the same ImplicitActuator model as in training.
# An explicit PD torque loop at 200 Hz is unstable with g1_ctrl's FixStand arm gains (kd=10 on
# light links), the real motors run their PD loop at kHz rates.
# Rigid-body view of the pelvis, only used to apply the virtual gantry force.
pelvis_body = RigidPrim(G1_ROOT_PRIM, name="g1_pelvis_body")
pelvis_body.initialize()

try:
    sim_joint_names = list(robot.joint_names)
except AttributeError:
    sim_joint_names = list(robot.dof_names)

# ── Unitree SDK motor order (= motor_state/motor_cmd index on the real robot) ─────────
# Derived from params/deploy.yaml joint_ids_map (see g1_29dof_policy_ros2.py).
SDK_JOINT_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
NUM_JOINTS = len(SDK_JOINT_ORDER)
diff = sorted(set(sim_joint_names) ^ set(SDK_JOINT_ORDER))
if diff:
    raise RuntimeError(f"Sim joints and SDK_JOINT_ORDER differ (G1 variant mismatch?): {diff}")
SIM_TO_SDK = np.array([sim_joint_names.index(n) for n in SDK_JOINT_ORDER])  # sim_vec[SIM_TO_SDK] -> sdk order
SDK_TO_SIM = np.array([SDK_JOINT_ORDER.index(n) for n in sim_joint_names])  # sdk_vec[SDK_TO_SIM] -> sim order

# Motor torque limits (same actuator groups as in g1_29dof_policy_ros2.py / unitree.py), armature 0.01.
EFFORT_LIMIT = {
    "hip_pitch": 88.0, "hip_yaw": 88.0, "waist_yaw": 88.0,
    "hip_roll": 139.0, "knee": 139.0,
    "ankle_pitch": 25.0, "ankle_roll": 25.0, "waist_roll": 25.0, "waist_pitch": 25.0,
    "shoulder_pitch": 25.0, "shoulder_roll": 25.0, "shoulder_yaw": 25.0, "elbow": 25.0, "wrist_roll": 25.0,
    "wrist_pitch": 5.0, "wrist_yaw": 5.0,
}


def effort_limit(name):
    for key, val in EFFORT_LIMIT.items():
        if key in name:
            return val
    raise KeyError(name)


EFFORT_LIMIT_SIM = np.array([effort_limit(n) for n in sim_joint_names])
robot._articulation_view.set_armatures(np.full((1, NUM_JOINTS), 0.01))
robot._articulation_view.set_max_efforts(EFFORT_LIMIT_SIM[None, :])

# Start like the real robot after power-on: hanging on the gantry with all joints at 0.
robot.set_joint_positions(np.zeros(NUM_JOINTS))
robot.set_joint_velocities(np.zeros(NUM_JOINTS))

# ── DDS ──────────────────────────────────────────────────────────────────────────────
ChannelFactoryInitialize(0, DDS_INTERFACE)  # domain 0, like g1_ctrl
state_pub = ChannelPublisher("rt/lowstate", LowState_)
state_pub.Init()
crc = CRC()

_cmd_lock = threading.Lock()
_latest_cmd = None
_latest_cmd_time = 0.0


def on_lowcmd(msg: LowCmd_):
    global _latest_cmd, _latest_cmd_time
    with _cmd_lock:
        _latest_cmd = msg
        _latest_cmd_time = time.monotonic()


cmd_sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
cmd_sub.Init(on_lowcmd, 10)

state_msg = unitree_hg_msg_dds__LowState_()
state_msg.mode_machine = 0  # 0 = simulation; g1_ctrl accepts it instead of the real robot type (5)

# ── Virtual remote (keyboard) ──────────────────────────────────────────────────────
# wireless_remote layout (unitree_joystick.hpp, REMOTE_DATA_RX, little-endian, 40 bytes):
#   uint8 head[2] | uint16 buttons | float lx | float rx | float ry | float L2 | float ly | pad
BTN_BITS = {"R1": 0, "L1": 1, "Start": 2, "Select": 3, "R2": 4, "L2": 5, "f1": 6, "f2": 7,
            "A": 8, "B": 9, "X": 10, "Y": 11, "up": 12, "right": 13, "down": 14, "left": 15}
# Button combos, as (start, end) seconds relative to the key press. g1_ctrl's LT (= L2) is a
# smoothed axis that needs ~0.5 s of holding before it counts as pressed, so the second button
# comes 0.8 s after LT (on the real remote you also hold LT first).
MACROS = {
    "stand": [("L2", 0.0, 1.3), ("up", 0.8, 1.3)],   # LT + up     -> FixStand
    "start": [("R1", 0.0, 0.6), ("X", 0.3, 0.6)],    # RB + X      -> Velocity
    "stop":  [("L2", 0.0, 1.3), ("B", 0.8, 1.3)],    # LT + B      -> Passive
}
STICK = 0.5
held_keys = set()
active_macro = None  # (name, start_wallclock)
gantry_on = True
gantry_z = SPAWN_Z + 0.06  # feet a few cm above the ground once standing


def run_macro(name):
    global active_macro
    active_macro = (name, time.monotonic())
    print(f"[remote] {name}: " + " + ".join(b for b, _, _ in MACROS[name]))


def set_gantry(on):
    global gantry_on, gantry_xy
    gantry_on = on
    if on:
        gantry_xy = robot.get_world_pose()[0][:2].copy()
    print(f"[gantry] {'ON' if on else 'OFF'} (height {gantry_z:.2f} m)")


def on_keyboard(event, *args, **kwargs):
    global gantry_z
    K = carb.input.KeyboardInput
    key = event.input
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        held_keys.add(key)
        if key == K.KEY_1:
            run_macro("stand")
        elif key == K.KEY_2:
            run_macro("start")
        elif key == K.KEY_0:
            run_macro("stop")
        elif key == K.G:
            set_gantry(not gantry_on)
        elif key == K.UP:
            gantry_z += 0.02
            print(f"[gantry] height {gantry_z:.2f} m")
        elif key == K.DOWN:
            gantry_z -= 0.02
            print(f"[gantry] height {gantry_z:.2f} m")
    elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
        held_keys.discard(key)
    return True


if not HEADLESS:
    _input = carb.input.acquire_input_interface()
    _kb_sub = _input.subscribe_to_keyboard_events(
        omni.appwindow.get_default_app_window().get_keyboard(), on_keyboard)

# Scripted remote for G1_DDS_AUTOTEST=1: (sim time [s], action)
AUTOTEST_SEQUENCE = [
    (3.0, "stand"), (6.5, ("gantry_z", 0.78)), (8.0, "start"), (10.0, "gantry_off"),
    (12.0, ("ly", 0.5)), (22.0, ("ly", 0.0)), (22.0, ("rx", -0.5)), (28.0, ("rx", 0.0)),
    (31.0, "done"),
]
auto_sticks = {"lx": 0.0, "ly": 0.0, "rx": 0.0}


def remote_bytes():
    btn = 0
    if active_macro is not None:
        name, t0 = active_macro
        dt = time.monotonic() - t0
        for b, start, end in MACROS[name]:
            if start <= dt < end:
                btn |= 1 << BTN_BITS[b]
    K = carb.input.KeyboardInput
    ly = STICK * ((K.W in held_keys) - (K.S in held_keys)) + auto_sticks["ly"]
    lx = STICK * ((K.D in held_keys) - (K.A in held_keys)) + auto_sticks["lx"]  # vy = -lx
    rx = STICK * ((K.E in held_keys) - (K.Q in held_keys)) + auto_sticks["rx"]  # yaw = -rx
    l2 = 1.0 if btn & (1 << BTN_BITS["L2"]) else 0.0
    # head bytes non-zero so g1_ctrl never sees an all-zero (= disconnected) remote
    data = struct.pack("<2BH5f", 0x55, 0x51, btn, lx, rx, 0.0, l2, ly)
    return list(data) + [0] * (40 - len(data))


# ── Virtual gantry (like unitree_mujoco's elastic band) ─────────────────────────────
# Spring-damper on the pelvis position plus an upright-keeping torque, applied every step
# while on. Real gantry: a rope on the torso; this is close enough for standing up.
GANTRY_KP_Z, GANTRY_KD_Z = 10000.0, 800.0
GANTRY_KP_XY, GANTRY_KD_XY = 2000.0, 400.0
GANTRY_KP_ROT, GANTRY_KD_ROT = 400.0, 40.0
gantry_xy = np.array([0.0, 0.0])


def gantry_wrench(pos, rot, lin_vel, ang_vel):
    target = np.array([gantry_xy[0], gantry_xy[1], gantry_z])
    err = target - pos
    force = np.array([
        GANTRY_KP_XY * err[0] - GANTRY_KD_XY * lin_vel[0],
        GANTRY_KP_XY * err[1] - GANTRY_KD_XY * lin_vel[1],
        max(0.0, GANTRY_KP_Z * err[2] - GANTRY_KD_Z * lin_vel[2]),  # a rope only pulls up
    ])
    # keep the pelvis upright and stop it from spinning (a real rope also damps twisting)
    up_body = rot.apply([0.0, 0.0, 1.0])
    tilt_axis = np.cross(up_body, [0.0, 0.0, 1.0])
    torque = GANTRY_KP_ROT * tilt_axis - GANTRY_KD_ROT * np.asarray(ang_vel)
    return force, torque


# ── Main loop ─────────────────────────────────────────────────────────────────────────
world.play()
set_gantry(True)
print(f"Fake G1 on DDS (domain 0, interface '{DDS_INTERFACE}'). Now start:  ./g1_ctrl --network {DDS_INTERFACE}")
print("Keys: 1 stand | DOWN x4 feet on ground | 2 start policy | G gantry off | WASD move | QE turn | 0 STOP")

step = 0
tick_ms = 0
prev_lin_vel_w = np.zeros(3)
wall_start = time.monotonic()
cmd_connected = False
auto_idx = 0
last_gains = None

while app.is_running():
    sim_time = step * PHYSICS_DT

    # ---- read state ----
    q_sim = robot.get_joint_positions()
    dq_sim = robot.get_joint_velocities()
    pos, quat_wxyz = robot.get_world_pose()
    lin_vel_w = robot.get_linear_velocity()
    ang_vel_w = robot.get_angular_velocity()
    if q_sim is None or dq_sim is None:
        world.step(render=False)
        step += 1
        continue
    rot = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])

    # ---- motor drivers: apply the latest rt/lowcmd ----
    with _cmd_lock:
        cmd = _latest_cmd
        cmd_age = time.monotonic() - _latest_cmd_time
    kp_sdk = np.zeros(NUM_JOINTS)  # no (or stale) command -> motors off, like the robot
    kd_sdk = np.zeros(NUM_JOINTS)
    q_des_sdk = np.zeros(NUM_JOINTS)
    dq_des_sdk = np.zeros(NUM_JOINTS)
    tau_ff_sdk = np.zeros(NUM_JOINTS)
    if cmd is not None and cmd_age < 0.1:
        if not cmd_connected:
            print("[dds] receiving rt/lowcmd from g1_ctrl")
            cmd_connected = True
        for i in range(NUM_JOINTS):
            m = cmd.motor_cmd[i]
            if m.mode == 0:  # motor disabled
                continue
            kp_sdk[i], kd_sdk[i] = m.kp, m.kd
            q_des_sdk[i], dq_des_sdk[i], tau_ff_sdk[i] = m.q, m.dq, m.tau
    elif cmd_connected and cmd is not None and cmd_age > 0.5:
        print("[dds] rt/lowcmd stopped - motors off")
        cmd_connected = False
    gains = (kp_sdk[SDK_TO_SIM], kd_sdk[SDK_TO_SIM])
    if last_gains is None or not (np.array_equal(gains[0], last_gains[0]) and np.array_equal(gains[1], last_gains[1])):
        robot._articulation_view.set_gains(kps=gains[0][None, :], kds=gains[1][None, :])
        last_gains = gains
    robot._articulation_view.set_joint_position_targets(q_des_sdk[SDK_TO_SIM][None, :])
    robot._articulation_view.set_joint_velocity_targets(dq_des_sdk[SDK_TO_SIM][None, :])
    robot.set_joint_efforts(tau_ff_sdk[SDK_TO_SIM])
    # torque estimate for tau_est in lowstate (what the drive applies, ignoring the implicit part)
    tau_sim = np.clip((tau_ff_sdk + kp_sdk * (q_des_sdk - q_sim[SIM_TO_SDK])
                       + kd_sdk * (dq_des_sdk - dq_sim[SIM_TO_SDK]))[SDK_TO_SIM],
                      -EFFORT_LIMIT_SIM, EFFORT_LIMIT_SIM)

    # ---- virtual gantry ----
    if gantry_on:
        f, t = gantry_wrench(np.asarray(pos), rot, np.asarray(lin_vel_w), np.asarray(ang_vel_w))
        pelvis_body.apply_forces_and_torques_at_pos(forces=f[None, :], torques=t[None, :], is_global=True)

    # ---- publish rt/lowstate (what the robot's motors + pelvis IMU report) ----
    q_sdk, dq_sdk, tau_sdk_state = q_sim[SIM_TO_SDK], dq_sim[SIM_TO_SDK], tau_sim[SIM_TO_SDK]
    for i in range(NUM_JOINTS):
        ms = state_msg.motor_state[i]
        ms.q = float(q_sdk[i])
        ms.dq = float(dq_sdk[i])
        ms.tau_est = float(tau_sdk_state[i])
    imu = state_msg.imu_state
    imu.quaternion = [float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])]
    imu.gyroscope = [float(v) for v in rot.inv().apply(ang_vel_w)]  # body frame, rad/s
    acc_w = (np.asarray(lin_vel_w) - prev_lin_vel_w) / PHYSICS_DT + np.array([0.0, 0.0, 9.81])
    imu.accelerometer = [float(v) for v in rot.inv().apply(acc_w)]
    imu.rpy = [float(v) for v in rot.as_euler("xyz")]
    prev_lin_vel_w = np.asarray(lin_vel_w).copy()
    state_msg.wireless_remote = remote_bytes()
    tick_ms += int(PHYSICS_DT * 1000)
    state_msg.tick = tick_ms
    state_msg.crc = crc.Crc(state_msg)
    state_pub.Write(state_msg)

    # ---- autotest script ----
    if AUTOTEST and auto_idx < len(AUTOTEST_SEQUENCE) and sim_time >= AUTOTEST_SEQUENCE[auto_idx][0]:
        action = AUTOTEST_SEQUENCE[auto_idx][1]
        auto_idx += 1
        if action == "gantry_off":
            set_gantry(False)
        elif action == "done":
            print("[autotest] done")
            break
        elif isinstance(action, tuple) and action[0] == "gantry_z":
            gantry_z = action[1]
            print(f"[gantry] height {gantry_z:.2f} m (feet on the ground)")
        elif isinstance(action, tuple):
            auto_sticks[action[0]] = action[1]
            print(f"[autotest] {action[0]} = {action[1]}")
        else:
            run_macro(action)
    if AUTOTEST and step % 200 == 0:
        print(f"[autotest t={sim_time:5.1f}s] pelvis xyz=({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:.2f}) "
              f"yaw={np.degrees(rot.as_euler('xyz')[2]):+.0f} deg  lowcmd={'yes' if cmd_connected else 'no'}  "
              f"max|q-q_cmd|={np.max(np.abs(q_des_sdk - q_sim[SIM_TO_SDK])) if cmd_connected else 0:.3f} rad")

    # ---- step physics, render at ~50 fps, keep real time (g1_ctrl runs on the wall clock) ----
    world.step(render=(step % 4 == 3))
    step += 1
    ahead = step * PHYSICS_DT - (time.monotonic() - wall_start)
    if ahead > 0:
        time.sleep(ahead)
    elif ahead < -0.5:  # fell behind (e.g. window dragged) - don't try to catch up
        wall_start = time.monotonic() - step * PHYSICS_DT

app.close()
