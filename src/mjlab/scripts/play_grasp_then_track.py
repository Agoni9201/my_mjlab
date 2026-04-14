"""Sequential WA1_D11 demo: grasp cylinder, transition to motion frame 0, then track.

This script keeps the existing trained grasp and tracking policies separate and
orchestrates them in one playback scene:

1. Run the grasp policy in the WA1_D11 manipulation scene.
2. Once the cylinder is stably lifted, attach it to the right grasp site.
3. Interpolate the robot joints to the first frame of the tracking motion.
4. Hand control to the tracking policy until the motion is finished.
"""

from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import mujoco 
import torch
import tyro
from tensordict import TensorDict

from mjlab.asset_zoo.robots import (
  WA1_D11_ACTION_SCALE,
  WA1_D11_MANIPULATION_ACTION_SCALE,
  WA1_FRONT_TABLE_POS,
  WA1_FRONT_TABLE_TOP_SIZE,
  WA1_HAND_CYLINDER_QUAT,
  WA1_HAND_CYLINDER_SIZE,
  WA1_LEFT_GRASP_SITE_NAME,
  WA1_RIGHT_GRASP_SITE_NAME,
)
from mjlab.asset_zoo.robots.wa1_d11.wa1_d11_constants import (
  WA1_LEFT_HAND_CYLINDER_GRASP_JOINT_POS,
  WA1_RIGHT_HAND_CYLINDER_GRASP_JOINT_POS,
  WA1_LEFT_ARM_JOINT_NAMES,
  WA1_RIGHT_ARM_JOINT_NAMES,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.managers.observation_manager import ObservationManager, ObservationTermCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.manipulation.config.wa1_d11.env_cfgs import (
  apply_wa1_d11_grasp_cylinder_stage,
  wa1_d11_grasp_cylinder_env_cfg,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.config.wa1_d11.env_cfgs import wa1_d11_tracking_env_cfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

_TRACK_TASK_ID = "Mjlab-Tracking-FixedBase-WA1-D11"
_GRASP_TASK_ID = "Mjlab-Grasp-Cylinder-WA1-D11"
_TABLE_TOP_SURFACE_Z = WA1_FRONT_TABLE_POS[2] + WA1_FRONT_TABLE_TOP_SIZE[2]
_CYLINDER_CENTER_Z = _TABLE_TOP_SURFACE_Z + WA1_HAND_CYLINDER_SIZE[1]
_LEFT_HAND_CYLINDER_LOCAL_POS = (0.0, -0.03, -0.165)
_LEFT_HAND_CYLINDER_LOCAL_QUAT = WA1_HAND_CYLINDER_QUAT
_LEFT_HAND_BODY_NAME = "WRIST_FLANGE_L"

_DEFAULT_GRASP_CHECKPOINT = (
  "/home/robot706/yx/mjlab_v3/logs/rsl_rl/wa1_d11_grasp_cylinder/"
  "2026-04-08_12-28-02_test_arm_v2/model_4999.pt"
)
_DEFAULT_TRACKING_CHECKPOINT = (
  "/home/robot706/yx/mjlab_v3/logs/rsl_rl/wa1_d11_tracking/"
  "2026-04-08_16-00-26_tracking_v2/model_6000.pt"
)
_DEFAULT_MOTION_FILE = "/home/robot706/yx/mjlab_v3/artifacts/motions:v30/motion.npz"

# 给 body 加一个 cylinder geom，尺寸沿用机器人手里圆柱尺寸
def _get_left_hand_visual_cylinder_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="left_hand_visual_cylinder")
  body.add_freejoint(name="left_hand_visual_cylinder_joint")
  geom = body.add_geom(
    name="left_hand_visual_cylinder_geom",
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    size=WA1_HAND_CYLINDER_SIZE,
    mass=1e-6,
    rgba=(0.95, 0.55, 0.15, 0.7),
  )
  geom.quat[:] = (1.0, 0.0, 0.0, 0.0)
  geom.contype = 0
  geom.conaffinity = 0
  return spec


def _stage_from_checkpoint(path: Path) -> Literal["easy", "mid", "final"]:
  stem = path.stem
  if not stem.startswith("model_"):
    return "final"
  try:
    iteration = int(stem.split("_", maxsplit=1)[1])
  except ValueError:
    return "final"
  if iteration < 400:
    return "easy"
  if iteration < 1200:
    return "mid"
  return "final"


def _resolve_grasp_stage(
  requested: Literal["default", "match_checkpoint", "easy", "mid", "final"],
  checkpoint_file: Path,
) -> Literal["easy", "mid", "final"]:
  if requested == "default":
    requested = "match_checkpoint"
  if requested == "match_checkpoint":
    return _stage_from_checkpoint(checkpoint_file)
  return requested


def _select_viewer_backend(
  viewer: Literal["auto", "native", "viser", "none"],
) -> Literal["native", "viser", "none"]:
  if viewer != "auto":
    return viewer
  has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
  return "native" if has_display else "viser"


def _make_scale_tensor(
  names: list[str],
  scale_map: dict[str, float],
  device: str,
) -> torch.Tensor:
  return torch.tensor([scale_map[name] for name in names], device=device)

def _collect_action_ids_by_name(
  action_names: list[str],
  selected_names: set[str],
  device: str,
) -> torch.Tensor:
  ids = [idx for idx, name in enumerate(action_names) if name in selected_names]
  return torch.tensor(ids, dtype=torch.long, device=device)


def _grasp_last_action(
  env: ManagerBasedRlEnv,
  track_to_grasp_ratio: torch.Tensor,
) -> torch.Tensor:
  action = env.action_manager.action
  converted = torch.zeros_like(action)
  nonzero = torch.abs(track_to_grasp_ratio) > 1e-8
  converted[:, nonzero] = action[:, nonzero] * track_to_grasp_ratio[nonzero]
  return converted


def _load_inference_policy(
  task_id: str,
  checkpoint_file: Path,
  device: str,
  motion_file: str | None = None,
):
  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  if task_id == _TRACK_TASK_ID:
    assert motion_file is not None
    env_cfg.commands["motion"].motion_file = motion_file

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint_file),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  policy = runner.get_inference_policy(device=device)
  wrapped.close()
  return policy, agent_cfg


@dataclass(frozen=True)
class GraspThenTrackPlayConfig:
  grasp_checkpoint_file: str = _DEFAULT_GRASP_CHECKPOINT
  tracking_checkpoint_file: str = _DEFAULT_TRACKING_CHECKPOINT
  motion_file: str = _DEFAULT_MOTION_FILE
  device: str | None = None
  viewer: Literal["auto", "native", "viser", "none"] = "auto"
  grasp_stage: Literal["default", "match_checkpoint", "easy", "mid", "final"] = (
    "default"
  )
  grasp_success_hold_steps: int = 20 # 抓取成功连续帧数阈值
  grasp_phase_speed: float = 0.2
  transition_phase_speed: float = 2.0
  tracking_phase_speed: float = 2.0
  transition_duration_s: float = 2.0
  track_start_delay_s: float = 0.3  # 从过渡动作结束到跟踪动作开始的延迟时间
  track_blend_duration_s: float = 0.4 # 从抓取动作过渡到跟踪动作的平滑过渡时间
  final_hold_duration_s: float = 1.5  # 进入跟踪动作后保持最后一帧动作的时间
  attach_object_during_tracking: bool = True
  show_left_hand_cylinder: bool = False
  max_steps: int | None = None

# 保存所有依赖和参数  状态机
class SequentialGraspTrackPolicy:
  def __init__(
    self,
    env: ManagerBasedRlEnv,
    grasp_policy,
    tracking_policy,
    grasp_obs_manager: ObservationManager,
    tracking_obs_manager: ObservationManager,
    grasp_to_track_ratio: torch.Tensor,
    track_action_scale: torch.Tensor,
    grasp_success_hold_steps: int,
    grasp_phase_speed: float,
    transition_phase_speed: float,
    tracking_phase_speed: float,
    transition_steps: int,
    track_start_delay_steps: int,
    track_blend_steps: int,
    final_hold_steps: int,
    attach_object_during_tracking: bool,
    show_left_hand_cylinder: bool,
  ) -> None:
    self.env = env
    self.grasp_policy = grasp_policy
    self.tracking_policy = tracking_policy
    self.grasp_obs_manager = grasp_obs_manager
    self.tracking_obs_manager = tracking_obs_manager
    self.grasp_to_track_ratio = grasp_to_track_ratio
    self.track_action_scale = track_action_scale
    self.grasp_success_hold_steps = grasp_success_hold_steps
    self.grasp_phase_speed = grasp_phase_speed
    self.transition_phase_speed = transition_phase_speed
    self.tracking_phase_speed = tracking_phase_speed
    self.transition_steps = max(transition_steps, 1)
    self.track_start_delay_steps = max(track_start_delay_steps, 0)
    self.track_blend_steps = max(track_blend_steps, 0)
    self.final_hold_steps = max(final_hold_steps, 1)
    self.attach_object_during_tracking = attach_object_during_tracking
    self.show_left_hand_cylinder = show_left_hand_cylinder

    self.robot = env.scene["robot"]
    self.cylinder_right = env.scene["cylinder_right"]
    self.cylinder_left = env.scene["cylinder_left"]
    self.left_hand_cylinder = (
      env.scene["left_hand_cylinder"]
      if "left_hand_cylinder" in env.scene.entities
      else None
    )
    self.lift_command_right = env.command_manager.get_term("lift_height_right")
    self.lift_command_left = env.command_manager.get_term("lift_height_left")
    self.motion_command = env.command_manager.get_term("motion")
    self.joint_ids = self.motion_command.joint_indexes
    self.right_grasp_site_id = self.robot.find_sites(
      (WA1_RIGHT_GRASP_SITE_NAME,), preserve_order=True
    )[0][0]
    self.left_grasp_site_id = self.robot.find_sites(
      (WA1_LEFT_GRASP_SITE_NAME,), preserve_order=True
    )[0][0]
    self.left_hand_body_id = self.robot.find_bodies(
      (_LEFT_HAND_BODY_NAME,), preserve_order=True
    )[0][0]

    joint_pos_action = env.action_manager.get_term("joint_pos")
    assert isinstance(joint_pos_action, JointPositionAction)
    self.action_term = joint_pos_action
    left_side_joint_names = set(WA1_LEFT_ARM_JOINT_NAMES) | set(
      WA1_LEFT_HAND_CYLINDER_GRASP_JOINT_POS.keys()
    )
    right_side_joint_names = set(WA1_RIGHT_ARM_JOINT_NAMES) | set(
      WA1_RIGHT_HAND_CYLINDER_GRASP_JOINT_POS.keys()
    )
    self.left_side_action_ids = _collect_action_ids_by_name(
      joint_pos_action.target_names, left_side_joint_names, env.device
    )
    self.right_side_action_ids = _collect_action_ids_by_name(
      joint_pos_action.target_names, right_side_joint_names, env.device
    )
    self.default_joint_pos = self.robot.data.default_joint_pos[:, self.joint_ids].clone()
    self.zero_root_velocity = torch.zeros(env.num_envs, 6, device=env.device)
    self._left_side_hold_joint_target = self.default_joint_pos.clone()
    self._right_side_hold_joint_target = self.default_joint_pos.clone()

    self.right_object_pos_offset = torch.zeros(env.num_envs, 3, device=env.device)
    self.right_object_quat_offset = torch.zeros(env.num_envs, 4, device=env.device)
    self.right_object_quat_offset[:, 0] = 1.0
    self.left_object_pos_offset = torch.zeros(env.num_envs, 3, device=env.device)
    self.left_object_quat_offset = torch.zeros(env.num_envs, 4, device=env.device)
    self.left_object_quat_offset[:, 0] = 1.0
    self.first_track_joint = self.motion_command.motion.joint_pos[0].unsqueeze(0).clone()
    self.last_track_joint = self.motion_command.motion.joint_pos[-1].unsqueeze(0).clone()
    # 状态机
    self.phase: Literal[
      "grasp", "transition", "pre_track_hold", "tracking", "hold"
    ] = "grasp"
    self._grasp_success_counter = 0
    self._transition_step = 0
    self._pre_track_hold_step = 0
    self._tracking_blend_step = 0
    self._hold_step = 0
    self._transition_start_joint = self.first_track_joint.clone()
    self._tracking_needs_frame0 = False

  def reset(self) -> None:
    env_ids = torch.arange(self.env.num_envs, device=self.env.device)
    self.phase = "grasp"
    self._grasp_success_counter = 0
    self._transition_step = 0
    self._pre_track_hold_step = 0
    self._tracking_blend_step = 0
    self._hold_step = 0
    self.motion_command.cfg.debug_vis = False
    self.lift_command_right.cfg.debug_vis = True
    self.lift_command_left.cfg.debug_vis = True
    self.motion_command.time_steps.zero_()
    self.motion_command.time_left.fill_(1e9)
    self.lift_command_right.time_left.fill_(1e9)
    self.lift_command_left.time_left.fill_(1e9)
    self._tracking_needs_frame0 = False

    self.env.action_manager.reset(env_ids=env_ids)
    self.grasp_obs_manager.reset(env_ids=env_ids)
    self.tracking_obs_manager.reset(env_ids=env_ids)

    # env.reset() samples the object/target with the lift command, then the injected
    # motion command overwrites the robot pose. Restore the robot to its default pose
    # so the grasp policy starts from the same state family it was trained on.
    self.robot.write_joint_state_to_sim(
      self.robot.data.default_joint_pos.clone(),
      self.robot.data.default_joint_vel.clone(),
      env_ids=env_ids,
    )
    self.robot.reset(env_ids=env_ids)
    current_joint = self.robot.data.joint_pos[:, self.joint_ids].clone()
    self._left_side_hold_joint_target = current_joint.clone()
    self._right_side_hold_joint_target = current_joint.clone()
    if self.show_left_hand_cylinder:
      self._update_left_hand_cylinder_visual()
    self._sync_scene()

  def __call__(self, _obs) -> torch.Tensor:
    del _obs
    if self.show_left_hand_cylinder:
      self._update_left_hand_cylinder_visual()
    if self.phase in {"transition", "pre_track_hold", "tracking", "hold"} and self.attach_object_during_tracking:
      self._attach_object_to_hand()

    if self.phase == "grasp":
      return self._grasp_action()
    if self.phase == "transition":
      return self._transition_action()
    if self.phase == "pre_track_hold":
      return self._pre_track_hold_action()
    if self.phase == "tracking":
      return self._tracking_action()
    return self._hold_phase_action()

  def _grasp_action(self) -> torch.Tensor:
    obs_tensor = self.grasp_obs_manager.compute_group("actor", update_history=True)
    obs = TensorDict({"actor": obs_tensor}, batch_size=[self.env.num_envs])
    grasp_action = self.grasp_policy(obs)
    track_action = grasp_action * self.grasp_to_track_ratio

    if self._grasp_is_stable():
      self._grasp_success_counter += 1
    else:
      self._grasp_success_counter = 0

    if self._grasp_success_counter >= self.grasp_success_hold_steps:
      self._begin_transition()

    return track_action

  def _transition_action(self) -> torch.Tensor:
    alpha = float(self._transition_step + 1) / float(self.transition_steps)
    target_joint = torch.lerp(
      self._transition_start_joint, self.first_track_joint, alpha
    )
    action = self._raw_action_for_joint_target(target_joint)
    self._transition_step += 1
    if self._transition_step >= self.transition_steps:
      self._begin_pre_track_hold()
    return action

  def _pre_track_hold_action(self) -> torch.Tensor:
    if self._pre_track_hold_step >= self.track_start_delay_steps:
      self._begin_tracking()
      return self._tracking_action()
    self._pre_track_hold_step += 1
    return self._raw_action_for_joint_target(self.first_track_joint)

  def _tracking_action(self) -> torch.Tensor:
    motion_total = self.motion_command.motion.time_step_total
    if self._tracking_needs_frame0:
      self.motion_command.time_steps.zero_()
      self._tracking_needs_frame0 = False
    if int(self.motion_command.time_steps.max().item()) >= motion_total - 1:
      self.phase = "hold"
      self._hold_step = 0
      return self._hold_last_track_action()

    obs_tensor = self.tracking_obs_manager.compute_group("actor", update_history=True)
    obs = TensorDict({"actor": obs_tensor}, batch_size=[self.env.num_envs])
    policy_action = self.tracking_policy(obs)
    if self._tracking_blend_step < self.track_blend_steps:
      hold_action = self._raw_action_for_joint_target(self.first_track_joint)
      blend_alpha = float(self._tracking_blend_step + 1) / float(
        max(self.track_blend_steps, 1)
      )
      # Smoothstep avoids an abrupt change in the first tracking frames.
      blend_alpha = blend_alpha * blend_alpha * (3.0 - 2.0 * blend_alpha)
      self._tracking_blend_step += 1
      return torch.lerp(hold_action, policy_action, blend_alpha)
    return policy_action

  def _hold_last_track_action(self) -> torch.Tensor:
    motion_total = self.motion_command.motion.time_step_total
    self.motion_command.time_steps.fill_(max(motion_total - 2, 0))
    return self._raw_action_for_joint_target(self.last_track_joint)

  def _hold_phase_action(self) -> torch.Tensor:
    if self._hold_step >= self.final_hold_steps:
      self._reset_episode()
      return self._grasp_action()
    self._hold_step += 1
    return self._hold_last_track_action()

  def _grasp_is_stable(self) -> bool:
    success_right = bool(torch.all(self.lift_command_right.compute_success()).item())
    success_left = bool(torch.all(self.lift_command_left.compute_success()).item())
    height_right = float(self.cylinder_right.data.root_link_pos_w[0, 2].item())
    height_left = float(self.cylinder_left.data.root_link_pos_w[0, 2].item())
    return (
      success_right
      and success_left
      and height_right >= (_CYLINDER_CENTER_Z + 0.10)
      and height_left >= (_CYLINDER_CENTER_Z + 0.10)
    )

  def _begin_transition(self) -> None:
    self.phase = "transition"
    self._transition_step = 0
    self._transition_start_joint = self.robot.data.joint_pos[:, self.joint_ids].clone()
    self.lift_command_right.cfg.debug_vis = False
    self.lift_command_left.cfg.debug_vis = False
    self.lift_command_right.time_left.fill_(1e9)
    self.lift_command_left.time_left.fill_(1e9)
    self.motion_command.cfg.debug_vis = False
    self.motion_command.time_steps.zero_()
    self.motion_command.time_left.fill_(1e9)
    if self.attach_object_during_tracking:
      self._capture_object_attach_offset()

  def _begin_pre_track_hold(self) -> None:
    self.phase = "pre_track_hold"
    self._pre_track_hold_step = 0
    self.motion_command.time_steps.zero_()
    self.motion_command.time_left.fill_(1e9)
    self.motion_command.cfg.debug_vis = False

  def _begin_tracking(self) -> None:
    env_ids = torch.arange(self.env.num_envs, device=self.env.device)
    self.phase = "tracking"
    self.motion_command.time_steps.zero_()
    self.motion_command.time_left.fill_(1e9)
    self.motion_command.cfg.debug_vis = True
    self.env.action_manager.reset(env_ids=env_ids)
    self.tracking_obs_manager.reset(env_ids=env_ids)
    self._tracking_needs_frame0 = True
    self._tracking_blend_step = 0

  def _reset_episode(self) -> None:
    self.env.reset()
    self.reset()

  def _raw_action_for_joint_target(self, joint_target: torch.Tensor) -> torch.Tensor:
    raw_action = torch.zeros_like(joint_target)
    nonzero = torch.abs(self.track_action_scale) > 1e-8
    raw_action[:, nonzero] = (
      joint_target[:, nonzero] - self.default_joint_pos[:, nonzero]
    ) / self.track_action_scale[nonzero]
    return raw_action

  def _capture_object_attach_offset(self) -> None:
    right_site_pos = self.robot.data.site_pos_w[:, self.right_grasp_site_id]
    right_site_quat = self.robot.data.site_quat_w[:, self.right_grasp_site_id]
    right_obj_pos = self.cylinder_right.data.root_link_pos_w
    right_obj_quat = self.cylinder_right.data.root_link_quat_w
    self.right_object_pos_offset = quat_apply(
      quat_inv(right_site_quat), right_obj_pos - right_site_pos
    )
    self.right_object_quat_offset = quat_mul(quat_inv(right_site_quat), right_obj_quat)

    left_site_pos = self.robot.data.site_pos_w[:, self.left_grasp_site_id]
    left_site_quat = self.robot.data.site_quat_w[:, self.left_grasp_site_id]
    left_obj_pos = self.cylinder_left.data.root_link_pos_w
    left_obj_quat = self.cylinder_left.data.root_link_quat_w
    self.left_object_pos_offset = quat_apply(
      quat_inv(left_site_quat), left_obj_pos - left_site_pos
    )
    self.left_object_quat_offset = quat_mul(quat_inv(left_site_quat), left_obj_quat)

  def _attach_object_to_hand(self) -> None:
    right_site_pos = self.robot.data.site_pos_w[:, self.right_grasp_site_id]
    right_site_quat = self.robot.data.site_quat_w[:, self.right_grasp_site_id]
    right_obj_pos = right_site_pos + quat_apply(right_site_quat, self.right_object_pos_offset)
    right_obj_quat = quat_mul(right_site_quat, self.right_object_quat_offset)
    right_obj_pose = torch.cat([right_obj_pos, right_obj_quat], dim=-1)
    self.cylinder_right.write_root_link_pose_to_sim(right_obj_pose)
    self.cylinder_right.write_root_link_velocity_to_sim(self.zero_root_velocity)

    left_site_pos = self.robot.data.site_pos_w[:, self.left_grasp_site_id]
    left_site_quat = self.robot.data.site_quat_w[:, self.left_grasp_site_id]
    left_obj_pos = left_site_pos + quat_apply(left_site_quat, self.left_object_pos_offset)
    left_obj_quat = quat_mul(left_site_quat, self.left_object_quat_offset)
    left_obj_pose = torch.cat([left_obj_pos, left_obj_quat], dim=-1)
    self.cylinder_left.write_root_link_pose_to_sim(left_obj_pose)
    self.cylinder_left.write_root_link_velocity_to_sim(self.zero_root_velocity)

  def _update_left_hand_cylinder_visual(self) -> None:
    if self.left_hand_cylinder is None:
      return
    body_pos = self.robot.data.body_link_pos_w[:, self.left_hand_body_id]
    body_quat = self.robot.data.body_link_quat_w[:, self.left_hand_body_id]
    local_pos = torch.tensor(
      _LEFT_HAND_CYLINDER_LOCAL_POS, device=self.env.device, dtype=torch.float32
    ).unsqueeze(0)
    local_quat = torch.tensor(
      _LEFT_HAND_CYLINDER_LOCAL_QUAT, device=self.env.device, dtype=torch.float32
    ).unsqueeze(0)
    cyl_pos = body_pos + quat_apply(body_quat, local_pos)
    cyl_quat = quat_mul(body_quat, local_quat)
    cyl_pose = torch.cat([cyl_pos, cyl_quat], dim=-1)
    self.left_hand_cylinder.write_root_link_pose_to_sim(cyl_pose)
    self.left_hand_cylinder.write_root_link_velocity_to_sim(self.zero_root_velocity)

  def _sync_scene(self) -> None:
    self.env.scene.write_data_to_sim()
    self.env.sim.forward()
    self.env.sim.sense()
    self.env.obs_buf = self.env.observation_manager.compute(update_history=True)

  def get_speed_multiplier(self) -> float:
    if self.phase == "transition":
      return self.transition_phase_speed
    if self.phase == "grasp":
      return self.grasp_phase_speed
    if self.phase in {"tracking", "pre_track_hold", "hold"}:
      return self.tracking_phase_speed
    return 1.0


class _PhaseSpeedMixin:
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._phase_speed_cache: float | None = None

  def tick(self) -> bool:
    get_speed = getattr(self.policy, "get_speed_multiplier", None)
    if callable(get_speed):
      desired = float(get_speed())
      if (
        self._phase_speed_cache is None
        or abs(desired - self._phase_speed_cache) > 1e-6
      ):
        self._phase_speed_cache = desired
        self._time_multiplier = desired
    return super().tick()


class PhaseAwareNativeViewer(_PhaseSpeedMixin, NativeMujocoViewer):
  pass


class PhaseAwareViserViewer(_PhaseSpeedMixin, ViserPlayViewer):
  pass


def _build_combo_env(
  motion_file: str,
  device: str,
  grasp_stage: Literal["easy", "mid", "final"],
  clip_actions: float | None,
  show_left_hand_cylinder: bool,
):
  combo_cfg = wa1_d11_grasp_cylinder_env_cfg(play=True)
  if show_left_hand_cylinder:
    combo_cfg.scene.entities["left_hand_cylinder"] = EntityCfg(
      init_state=EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, _CYLINDER_CENTER_Z),
      ),
      spec_fn=_get_left_hand_visual_cylinder_spec,
    )
  combo_cfg.actions["joint_pos"].scale = WA1_D11_ACTION_SCALE
  combo_cfg.terminations = {}
  combo_cfg.curriculum = {}
  combo_cfg.commands["lift_height_right"].resampling_time_range = (1e9, 1e9)
  combo_cfg.commands["lift_height_left"].resampling_time_range = (1e9, 1e9)
  combo_cfg.episode_length_s = int(1e9)
  apply_wa1_d11_grasp_cylinder_stage(
    combo_cfg, grasp_stage, command_name="lift_height_right"
  )
  apply_wa1_d11_grasp_cylinder_stage(
    combo_cfg, grasp_stage, command_name="lift_height_left"
  )

  env = ManagerBasedRlEnv(cfg=combo_cfg, device=device)
  wrapped_env = RslRlVecEnvWrapper(env, clip_actions=clip_actions)

  motion_cfg = wa1_d11_tracking_env_cfg(play=True).commands["motion"]
  motion_cfg.motion_file = motion_file
  motion_term = motion_cfg.build(env)
  motion_term.cfg.debug_vis = False
  motion_term.time_left.fill_(1e9)
  env.command_manager.cfg["motion"] = motion_cfg
  env.command_manager._terms["motion"] = motion_term

  joint_pos_action = env.action_manager.get_term("joint_pos")
  assert isinstance(joint_pos_action, JointPositionAction)
  target_names = joint_pos_action.target_names

  grasp_actor_cfg = copy.deepcopy(wa1_d11_grasp_cylinder_env_cfg(play=True).observations["actor"])
  track_scale = _make_scale_tensor(
    target_names, WA1_D11_ACTION_SCALE, device
  )
  grasp_scale = _make_scale_tensor(
    target_names, WA1_D11_MANIPULATION_ACTION_SCALE, device
  )

  grasp_to_track_ratio = torch.zeros_like(track_scale)
  nonzero_track = torch.abs(track_scale) > 1e-8
  grasp_to_track_ratio[nonzero_track] = (
    grasp_scale[nonzero_track] / track_scale[nonzero_track]
  )
  track_to_grasp_ratio = torch.zeros_like(track_scale)
  nonzero_grasp = torch.abs(grasp_scale) > 1e-8
  track_to_grasp_ratio[nonzero_grasp] = (
    track_scale[nonzero_grasp] / grasp_scale[nonzero_grasp]
  )

  grasp_actor_cfg.terms["actions"] = ObservationTermCfg(
    func=_grasp_last_action,
    params={"track_to_grasp_ratio": track_to_grasp_ratio},
  )
  grasp_obs_manager = ObservationManager({"actor": grasp_actor_cfg}, env)

  tracking_actor_cfg = copy.deepcopy(wa1_d11_tracking_env_cfg(play=True).observations["actor"])
  tracking_obs_manager = ObservationManager({"actor": tracking_actor_cfg}, env)

  return (
    wrapped_env,
    env,
    grasp_obs_manager,
    tracking_obs_manager,
    grasp_to_track_ratio,
    track_scale,
  )


def run_play(cfg: GraspThenTrackPlayConfig) -> None:
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  grasp_checkpoint = Path(cfg.grasp_checkpoint_file).expanduser().resolve()
  tracking_checkpoint = Path(cfg.tracking_checkpoint_file).expanduser().resolve()
  motion_file = str(Path(cfg.motion_file).expanduser().resolve())

  if not grasp_checkpoint.exists():
    raise FileNotFoundError(f"Grasp checkpoint not found: {grasp_checkpoint}")
  if not tracking_checkpoint.exists():
    raise FileNotFoundError(f"Tracking checkpoint not found: {tracking_checkpoint}")
  if not Path(motion_file).exists():
    raise FileNotFoundError(f"Motion file not found: {motion_file}")

  import mjlab.tasks  # noqa: F401

  grasp_policy, grasp_agent_cfg = _load_inference_policy(
    _GRASP_TASK_ID, grasp_checkpoint, device
  )
  tracking_policy, tracking_agent_cfg = _load_inference_policy(
    _TRACK_TASK_ID, tracking_checkpoint, device, motion_file=motion_file
  )

  grasp_stage = _resolve_grasp_stage(cfg.grasp_stage, grasp_checkpoint)
  print(f"[INFO]: Using grasp play stage: {grasp_stage}")

  clip_actions = tracking_agent_cfg.clip_actions
  if grasp_agent_cfg.clip_actions is not None:
    if clip_actions is None:
      clip_actions = grasp_agent_cfg.clip_actions
    else:
      clip_actions = max(float(clip_actions), float(grasp_agent_cfg.clip_actions))

  (
    wrapped_env,
    raw_env,
    grasp_obs_manager,
    tracking_obs_manager,
    grasp_to_track_ratio,
    track_action_scale,
  ) = _build_combo_env(
    motion_file,
    device,
    grasp_stage,
    clip_actions,
    cfg.show_left_hand_cylinder,
  )

  policy = SequentialGraspTrackPolicy(
    env=raw_env,
    grasp_policy=grasp_policy,
    tracking_policy=tracking_policy,
    grasp_obs_manager=grasp_obs_manager,
    tracking_obs_manager=tracking_obs_manager,
    grasp_to_track_ratio=grasp_to_track_ratio,
    track_action_scale=track_action_scale,
    grasp_success_hold_steps=cfg.grasp_success_hold_steps,
    grasp_phase_speed=cfg.grasp_phase_speed,
    transition_phase_speed=cfg.transition_phase_speed,
    tracking_phase_speed=cfg.tracking_phase_speed,
    transition_steps=int(round(cfg.transition_duration_s / raw_env.step_dt)),
    track_start_delay_steps=int(round(cfg.track_start_delay_s / raw_env.step_dt)),
    track_blend_steps=int(round(cfg.track_blend_duration_s / raw_env.step_dt)),
    final_hold_steps=int(round(cfg.final_hold_duration_s / raw_env.step_dt)),
    attach_object_during_tracking=cfg.attach_object_during_tracking,
    show_left_hand_cylinder=cfg.show_left_hand_cylinder,
  )
  policy.reset()

  viewer_backend = _select_viewer_backend(cfg.viewer)
  if viewer_backend == "native":
    PhaseAwareNativeViewer(wrapped_env, policy).run(num_steps=cfg.max_steps)
  elif viewer_backend == "viser":
    PhaseAwareViserViewer(wrapped_env, policy).run(num_steps=cfg.max_steps)
  else:
    total_steps = cfg.max_steps or 200
    for _ in range(total_steps):
      obs = wrapped_env.get_observations()
      actions = policy(obs)
      wrapped_env.step(actions)
    wrapped_env.close()


def main() -> None:
  config = tyro.cli(GraspThenTrackPlayConfig)
  run_play(config)


if __name__ == "__main__":
  main()
