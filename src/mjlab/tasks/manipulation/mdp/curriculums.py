from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import torch

from mjlab.tasks.manipulation.mdp.commands import LiftingCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = ["LiftingCommandCurriculumStage", "lifting_command_curriculum"]


class _LiftingCommandCurriculumStageOptional(TypedDict, total=False):
  success_threshold: float
  target_position_range: dict[str, tuple[float, float]]
  object_pose_range: dict[str, tuple[float, float]]


class LiftingCommandCurriculumStage(_LiftingCommandCurriculumStageOptional):
  step: int


def _apply_range_updates(
  range_cfg: object,
  updates: dict[str, tuple[float, float]],
  stage_step: int,
  field_name: str,
) -> None:
  valid_keys = {"x", "y", "z", "yaw"}
  unknown = updates.keys() - valid_keys
  if unknown:
    raise KeyError(
      f"lifting_command_curriculum: stage at step {stage_step} sets unknown "
      f"{field_name} key(s) {unknown}."
    )
  for key, value in updates.items():
    setattr(range_cfg, key, value)


def lifting_command_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  stages: list[LiftingCommandCurriculumStage],
) -> dict[str, torch.Tensor]:
  """Apply staged curriculum updates to a lifting command configuration."""
  del env_ids  # Unused.
  command_cfg = env.command_manager.get_term_cfg(command_name)
  if not isinstance(command_cfg, LiftingCommandCfg):
    raise TypeError(
      f"lifting_command_curriculum expects '{command_name}' to use "
      f"LiftingCommandCfg, got {type(command_cfg).__name__}."
    )

  for stage in stages:
    if env.common_step_counter >= stage["step"]:
      if "success_threshold" in stage:
        command_cfg.success_threshold = stage["success_threshold"]
      if "target_position_range" in stage:
        _apply_range_updates(
          command_cfg.target_position_range,
          stage["target_position_range"],
          stage["step"],
          "target_position_range",
        )
      if "object_pose_range" in stage:
        object_pose_range = command_cfg.object_pose_range
        if object_pose_range is None:
          raise RuntimeError(
            "lifting_command_curriculum cannot update object_pose_range when "
            "the command has object_pose_range=None."
          )
        _apply_range_updates(
          object_pose_range,
          stage["object_pose_range"],
          stage["step"],
          "object_pose_range",
        )

  return {
    "success_threshold": torch.tensor(command_cfg.success_threshold),
    "target_z_min": torch.tensor(command_cfg.target_position_range.z[0]),
    "target_z_max": torch.tensor(command_cfg.target_position_range.z[1]),
    "object_x_span": torch.tensor(
      command_cfg.object_pose_range.x[1] - command_cfg.object_pose_range.x[0]
    ),
    "object_y_span": torch.tensor(
      command_cfg.object_pose_range.y[1] - command_cfg.object_pose_range.y[0]
    ),
  }
