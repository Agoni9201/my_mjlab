from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.manipulation.mdp.commands import (
  LiftingCommand,
  MultiCubeLiftingCommand,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def staged_position_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  reaching_std: float,
  bringing_std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Curriculum reward that gates lifting bonus on reaching progress.

  Returns reaching * (1 + bringing), where both terms are Gaussian kernels
  over position error. Ensures learning signal for approach before lift.
  """
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  command = cast(LiftingCommand, env.command_manager.get_term(command_name))
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  obj_pos_w = obj.data.root_link_pos_w
  reach_error = torch.sum(torch.square(ee_pos_w - obj_pos_w), dim=-1)
  reaching = torch.exp(-reach_error / reaching_std**2)
  position_error = torch.sum(torch.square(command.target_pos - obj_pos_w), dim=-1)
  bringing = torch.exp(-position_error / bringing_std**2)
  return reaching * (1.0 + bringing)


def bring_object_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  std: float,
) -> torch.Tensor:
  obj: Entity = env.scene[object_name]
  command = cast(LiftingCommand, env.command_manager.get_term(command_name))
  obj_pos_w = torch.nan_to_num(obj.data.root_link_pos_w)
  position_error = torch.sum(
    torch.square(command.target_pos - obj.data.root_link_pos_w), dim=-1
  )
  return torch.exp(-position_error / std**2)


def multi_cube_staged_position_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  reaching_std: float,
  bringing_std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Staged reward for the target cube selected by MultiCubeLiftingCommand."""
  robot: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, MultiCubeLiftingCommand):
    raise TypeError(
      f"Command '{command_name}' must be a MultiCubeLiftingCommand, got {type(command)}"
    )
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  obj_pos_w = command.target_object_pos()
  reach_error = torch.sum(torch.square(ee_pos_w - obj_pos_w), dim=-1)
  reaching = torch.exp(-reach_error / reaching_std**2)
  position_error = torch.sum(torch.square(command.target_pos - obj_pos_w), dim=-1)
  bringing = torch.exp(-position_error / bringing_std**2)
  return reaching * (1.0 + bringing)


def multi_cube_bring_object_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """Gaussian reward for bringing the selected target cube to goal."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, MultiCubeLiftingCommand):
    raise TypeError(
      f"Command '{command_name}' must be a MultiCubeLiftingCommand, got {type(command)}"
    )
  obj_pos_w = command.target_object_pos()
  position_error = torch.sum(torch.square(command.target_pos - obj_pos_w), dim=-1)
  return torch.exp(-position_error / std**2)


def joint_velocity_hinge_penalty(
  env: ManagerBasedRlEnv,
  max_vel: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Quadratic hinge penalty on joint velocities exceeding a symmetric limit.

  Penalizes only the amount by which |v| exceeds max_vel. Returns a negative
  penalty, shaped as the negative squared L2 norm of the excess velocities.
  """
  robot: Entity = env.scene[asset_cfg.name]
  joint_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
  excess = (joint_vel.abs() - max_vel).clamp_min(0.0)
  return (excess**2).sum(dim=-1)

def joint_velocity_hinge_penalty_clipped(
  env: ManagerBasedRlEnv,
  max_vel: float,
  clip_excess: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Quadratic hinge penalty with clipped excess to avoid rare reward explosions."""
  robot: Entity = env.scene[asset_cfg.name]
  joint_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
  excess = (joint_vel.abs() - max_vel).clamp_min(0.0).clamp_max(clip_excess)
  return (excess**2).sum(dim=-1)


def contact_force_reward(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_scale: float = 5.0,
  min_force: float = 0.25,
) -> torch.Tensor:
  """Dense reward for establishing contact with the manipulation object."""
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data

  if data.force is not None:
    force_mag = torch.norm(torch.nan_to_num(data.force), dim=-1).amax(dim=-1)
    active_force = (force_mag - min_force).clamp_min(0.0)
    return torch.tanh(active_force / force_scale)

  if data.found is None:
    raise RuntimeError(
      f"Contact sensor '{sensor_name}' must expose either `force` or `found`."
    )
  return (torch.nan_to_num(data.found).amax(dim=-1) > 0).float()


def lift_height_progress_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  start_height: float,
) -> torch.Tensor:
  """Reward normalized upward progress from the table surface toward target z."""
  obj: Entity = env.scene[object_name]
  command = cast(LiftingCommand, env.command_manager.get_term(command_name))
  obj_z = torch.nan_to_num(obj.data.root_link_pos_w[:, 2])
  target_z = command.target_pos[:, 2]
  denom = (target_z - start_height).clamp_min(1.0e-4)
  progress = ((obj_z - start_height) / denom).clamp(0.0, 1.0)
  return torch.sqrt(progress)

# 新增
# def acdc_pose_quality_reward(
#   env: ManagerBasedRlEnv,
#   pose_std: float,
#   vel_std: float,
#   asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
# ) -> torch.Tensor:
#   """ACDC-style quality term: prefer default-like and smooth joint postures.

#   This is a lightweight proxy for anthropomorphic quality when replay-level
#   contrastive control is unavailable in the current training stack.
#   """
#   robot: Entity = env.scene[asset_cfg.name]
#   joint_ids = asset_cfg.joint_ids
#   joint_pos = robot.data.joint_pos[:, joint_ids]
#   default_joint_pos = robot.data.default_joint_pos[:, joint_ids]
#   joint_vel = robot.data.joint_vel[:, joint_ids]

#   pose_error = torch.square(joint_pos - default_joint_pos)
#   vel_error = torch.square(joint_vel)

#   pose_score = torch.exp(-torch.mean(pose_error / (pose_std**2), dim=-1))
#   vel_score = torch.exp(-torch.mean(vel_error / (vel_std**2), dim=-1))
#   return pose_score * vel_score