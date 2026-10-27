"""Custom reward functions.

Every reward function follows the same contract:
  - first argument is the env, the rest are whatever you put in RewTerm(params={...})
  - returns a tensor of shape (num_envs,), one value per robot
  - return the raw quantity; the sign and scale come from the RewTerm weight
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_too_close(
    env: ManagerBasedRLEnv, min_dist: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """How far (in m) the two feet are closer than min_dist in the horizontal plane. 0 when far enough apart.

    Use with a negative weight to penalize crossing/clipping legs.
    """
    asset = env.scene[asset_cfg.name]
    # asset_cfg.body_ids is filled in by the reward manager from the body_names you pass in the config
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]  # (num_envs, 2 feet, xy)
    dist = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=1)  # (num_envs,)
    return torch.clamp(min_dist - dist, min=0.0)

def stationary_feet_airtime(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, force_threshold: float = 1.0,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """penalize the robot for lifting its feet off the ground when it is standing still"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # (num_envs, 2 feet, xyz)
    in_air = torch.norm(forces, dim=-1) < force_threshold  # (num_envs, 2 feet)
    command = env.command_manager.get_command(command_name)  # (num_envs, 3)
    standing = torch.norm(command, dim=1) < command_threshold  # (num_envs,)
    return in_air.sum(dim=1).float() * standing

def stand_still_joint_deviation(
    env: ManagerBasedRLEnv, command_name: str, command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Sum of |joint pos - default joint pos| (rad) over the selected joints, only when the command is ~zero.

    Use with a negative weight: gives the robot a target pose to settle into when told to stand still,
    instead of only punishing foot lift-off. Uses the full (vx, vy, wz) command, so turning in place is not penalized.
    """
    asset = env.scene[asset_cfg.name]
    deviation = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    command = env.command_manager.get_command(command_name)  # (num_envs, 3)
    standing = torch.norm(command, dim=1) < command_threshold  # (num_envs,)
    return torch.sum(torch.abs(deviation), dim=1) * standing