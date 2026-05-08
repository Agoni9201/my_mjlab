from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import torch

from mjlab.tasks.manipulation.mdp.commands import LiftingCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = [
  "LiftingCommandCurriculumStage",
  # "AdaptiveLambdaCurriculumCfg",
  "lifting_command_curriculum",
  # "adaptive_lambda_quality_curriculum",
]

class _LiftingCommandCurriculumStageOptional(TypedDict, total=False):
  success_threshold: float
  target_position_range: dict[str, tuple[float, float]]
  object_pose_range: dict[str, tuple[float, float]]


class LiftingCommandCurriculumStage(_LiftingCommandCurriculumStageOptional):
  step: int

# # 新增
# class _AdaptiveLambdaCurriculumOptional(TypedDict, total=False):
#   low_success_threshold: float
#   high_success_threshold: float
#   smoothing_factor: float
#   min_scale: float
#   max_scale: float

# class AdaptiveLambdaCurriculumCfg(_AdaptiveLambdaCurriculumOptional):
#   quality_reward_name: str
#   command_name: str

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


# def adaptive_lambda_quality_curriculum(
#   env: ManagerBasedRlEnv,
#   env_ids: torch.Tensor,
#   quality_reward_name: str,
#   command_name: str,
#   low_success_threshold: float = 0.30,
#   high_success_threshold: float = 0.60,
#   smoothing_factor: float = 0.70,
#   min_scale: float = 0.50,
#   max_scale: float = 2.00,
# ) -> dict[str, torch.Tensor]:
#   """ACDC-like adaptive scheduler for a posture quality reward.

#   The reward weight is scaled according to success rate and smoothed over time:
#   low success -> lower quality weight (prioritize task completion),
#   high success -> higher quality weight (prioritize anthropomorphic quality).
#   """
#   del env_ids  # Unused.

#   reward_term_cfg = env.reward_manager.get_term_cfg(quality_reward_name)
#   command_term = env.command_manager.get_term(command_name)
#   if not hasattr(command_term, "metrics"):
#     raise TypeError(
#       f"adaptive_lambda_quality_curriculum expects command '{command_name}' "
#       "to expose a metrics dict."
#     )

#   command_metrics = command_term.metrics
#   if "episode_success" in command_metrics:
#     success = torch.nan_to_num(command_metrics["episode_success"].float())
#   elif "at_goal" in command_metrics:
#     success = torch.nan_to_num(command_metrics["at_goal"].float())
#   else:
#     raise KeyError(
#       f"Command '{command_name}' has no 'episode_success' or 'at_goal' metric "
#       "for adaptive lambda scheduling."
#     )
#   success_rate = success.mean().item()

#   state_key = f"_acdc_lambda_state__{quality_reward_name}"
#   if not hasattr(env, state_key):
#     setattr(
#       env,
#       state_key,
#       {
#         "current_scale": 1.0,
#         "base_weight": float(reward_term_cfg.weight),
#       },
#     )

#   state = getattr(env, state_key)
#   if success_rate < low_success_threshold:
#     target_scale = min_scale
#   elif success_rate <= high_success_threshold:
#     target_scale = 1.0
#   else:
#     target_scale = max_scale

#   prev_scale = float(state["current_scale"])
#   current_scale = smoothing_factor * prev_scale + (1.0 - smoothing_factor) * target_scale
#   current_scale = max(min_scale, min(max_scale, current_scale))
#   state["current_scale"] = current_scale

#   reward_term_cfg.weight = float(state["base_weight"]) * current_scale
#   return {
#     "success_rate": torch.tensor(success_rate),
#     "target_scale": torch.tensor(target_scale),
#     "current_scale": torch.tensor(current_scale),
#     "weight": torch.tensor(reward_term_cfg.weight),
#   }
