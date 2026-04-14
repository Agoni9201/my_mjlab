from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)  # [B]
  assert data.found is not None
  return torch.any(data.found, dim=-1)


def object_below_height(
  env: ManagerBasedRlEnv,
  object_name: str,
  min_height: float,
) -> torch.Tensor:
  obj: Entity = env.scene[object_name]
  return obj.data.root_link_pos_w[:, 2] < min_height


def invalid_object_state(
  env: ManagerBasedRlEnv,
  object_name: str,
) -> torch.Tensor:
  obj: Entity = env.scene[object_name]
  if obj.is_fixed_base:
    pose = obj.data.root_link_pose_w
    vel = obj.data.root_link_vel_w
  else:
    pose = obj.data.data.qpos[:, obj.data.indexing.free_joint_q_adr]
    vel = obj.data.data.qvel[:, obj.data.indexing.free_joint_v_adr]
  state = torch.cat([pose, vel], dim=-1)
  return ~torch.isfinite(state).all(dim=-1)


def joint_velocity_exceeded(
  env: ManagerBasedRlEnv,
  max_abs_vel: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  return torch.any(joint_vel.abs() > max_abs_vel, dim=-1)
