from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  quat_from_euler_xyz,
  sample_uniform,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class LiftingCommand(CommandTerm):
  cfg: LiftingCommandCfg

  def __init__(self, cfg: LiftingCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.object: Entity = env.scene[cfg.entity_name] # 目标物体实体
    self.target_pos = torch.zeros(self.num_envs, 3, device=self.device) # 训练时外部拿到的命令就是target_pos，智能体需要根据这个目标位置来学习如何移动物体到目标位置。
    self.episode_success = torch.zeros(self.num_envs, device=self.device) # 各自训练指标metrics，比如说位置误差、是否达到目标等，episode_success是一个指标，表示每个环境是否成功完成任务，一旦成功就保持为1。
    
    # # 保存“课程给的提升目标 z”，门控时不要丢掉，缓存“抬升目标 z”和抓取锁存状态
    # self._lift_target_pos = torch.zeros(self.num_envs, 3, device=self.device)
    # self._pregrasp_target = torch.zeros(self.num_envs, 3, device=self.device)
    # self._grasp_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    # self._grasp_true_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

    self.metrics["object_height"] = torch.zeros(self.num_envs, device=self.device) # 物体高度，作为训练指标之一
    self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device) # 位置误差，表示物体当前位置与目标位置之间的距离，也是训练指标之一
    self.metrics["at_goal"] = torch.zeros(self.num_envs, device=self.device) # 是否达到目标位置，通常是一个二值指标，当位置误差小于某个阈值时为1，否则为0
    self.metrics["episode_success"] = torch.zeros(self.num_envs, device=self.device) # 一旦成功就锁存为1
    # self.metrics["grasped"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.target_pos

  def _update_metrics(self) -> None:
    object_pos_w = self.object.data.root_link_pos_w
    object_height = object_pos_w[:, 2]
    position_error = torch.norm(self.target_pos - object_pos_w, dim=-1)
    at_goal = (position_error < self.cfg.success_threshold).float()

    # Latch episode_success to 1 once goal is reached.
    self.episode_success = torch.maximum(self.episode_success, at_goal)

    self.metrics["object_height"] = object_height
    self.metrics["position_error"] = position_error
    self.metrics["at_goal"] = at_goal
    self.metrics["episode_success"] = self.episode_success

  def compute_success(self) -> torch.Tensor:
    position_error = self.metrics["position_error"]
    return position_error < self.cfg.success_threshold # 成功的条件就是位置误差小于某个阈值，这个阈值在cfg中定义为success_threshold，智能体需要学习如何将物体移动到目标位置，使得位置误差足够小，从而完成任务。
  # 只在重采样时触发负责采样目标点、重置物体的目标位置和物体位置。compute_success只计算当前的成功状态，不修改环境状态。
  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)

    # Reset episode success for resampled envs.
    self.episode_success[env_ids] = 0.0
    # self._grasp_latched[env_ids] = False
    # self._grasp_true_count[env_ids] = 0

    # Set target position based on difficulty mode.如果是fixed模式，目标位置固定在(0.4, 0.0, 0.3)；如果是dynamic模式，目标位置在指定范围内随机采样。目标位置是相对于环境原点的绝对坐标。
    if self.cfg.difficulty == "fixed":
      target_pos = torch.tensor(
        [0.4, 0.0, 0.3], device=self.device, dtype=torch.float32
      ).expand(n, 3)
      self.target_pos[env_ids] = target_pos + self._env.scene.env_origins[env_ids]
    else:
      assert self.cfg.difficulty == "dynamic"
      r = self.cfg.target_position_range
      lower = torch.tensor([r.x[0], r.y[0], r.z[0]], device=self.device)
      upper = torch.tensor([r.x[1], r.y[1], r.z[1]], device=self.device)
      target_pos = sample_uniform(lower, upper, (n, 3), device=self.device)
      self.target_pos[env_ids] = target_pos + self._env.scene.env_origins[env_ids]
    # # 记录采样出来的位置作为提升目标位置
    # self._lift_target_pos[env_ids] = self.target_pos[env_ids]
    # obj_pos_for_pregrasp = None
    # # Reset object to new position.
    if self.cfg.object_pose_range is not None:
      r = self.cfg.object_pose_range
      lower = torch.tensor([r.x[0], r.y[0], r.z[0]], device=self.device)
      upper = torch.tensor([r.x[1], r.y[1], r.z[1]], device=self.device)
      pos = sample_uniform(lower, upper, (n, 3), device=self.device)
      pos = pos + self._env.scene.env_origins[env_ids]

      # Sample orientation (yaw only, keep upright).
      yaw = sample_uniform(r.yaw[0], r.yaw[1], (n,), device=self.device)
      quat = quat_from_euler_xyz(
        torch.zeros(n, device=self.device),  # roll
        torch.zeros(n, device=self.device),  # pitch
        yaw,
      )
      pose = torch.cat([pos, quat], dim=-1)

      velocity = torch.zeros(n, 6, device=self.device)

      self.object.write_root_link_pose_to_sim(pose, env_ids=env_ids)
      self.object.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)
      # this is deterministic for this resample
    #   obj_pos_for_pregrasp = pos
    # else:
    #   # no object reset this round -> use current object pose
    #   obj_pos_for_pregrasp = torch.nan_to_num(self.object.data.root_link_pos_w[env_ids])

    # # pregrasp target: only computed once per resample
    # pre = obj_pos_for_pregrasp.clone()
    # pre[:, 0] = obj_pos_for_pregrasp[:, 0] + self.cfg.pregrasp_offset_x
    # pre[:, 1] = obj_pos_for_pregrasp[:, 1] + self.cfg.pregrasp_offset_y
    # pre[:, 2] = obj_pos_for_pregrasp[:, 2] + self.cfg.pregrasp_offset_z
    # self._pregrasp_target[env_ids] = pre# 如果抓住了，目标z就是提升目标z；如果没抓住，目标z就是物体当前z加上一个偏移，这样智能体就需要先学会如何抓住物体，然后才能学会提升物体。

  def _update_command(self) -> None:
    pass
    # if self.cfg.grasp_sensor_name is None: # 如果没有配置抓取传感器就不做门控
    #   return
    # # 从场景中取接触传感器
    # sensor = self._env.scene[self.cfg.grasp_sensor_name]
    # assert isinstance(sensor, ContactSensor)
    # data = sensor.data
    # # 计算是否抓住了物体，如果传感器有力的输出，就根据力的大小判断是否抓住；如果没有力的输出但有接触检测输出，就根据是否检测到接触来判断；如果两者都没有，就认为没有抓住。
    # if data.force is not None:
    #   force_mag = torch.norm(torch.nan_to_num(data.force), dim=-1).amax(dim=-1)
    #   grasped_now = force_mag > self.cfg.grasp_min_force
    # elif data.found is not None:
    #   grasped_now = (torch.nan_to_num(data.found).amax(dim=-1) > 0)
    # else:
    #   grasped_now = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    #   # 多帧确认：连续 N 帧为 True 才认为抓取成功
    # confirm_frames = max(1, int(self.cfg.grasp_confirm_frames))
    # self._grasp_true_count = torch.where(
    #   grasped_now,
    #   torch.clamp(self._grasp_true_count + 1, max=confirm_frames),# 限幅，超过confirm_frames就保持在confirm_frames，这样后续判断是否抓住只需要看_grasp_true_count是否大于等于confirm_frames即可
    #   torch.zeros_like(self._grasp_true_count),
    # )
    # grasped_confirmed = self._grasp_true_count >= confirm_frames

    # # 是否锁住状态，如果cfg.latch_grasp为True，一旦抓住就锁存，直到重采样才解锁；如果为False，则不锁存，实时根据当前是否抓住来决定。
    #   # 锁存逻辑：一旦确认抓住，可保持到重采样
    # if self.cfg.latch_grasp:
    #   self._grasp_latched = self._grasp_latched | grasped_confirmed
    #   grasped = self._grasp_latched
    # else:
    #   grasped = grasped_confirmed
    
    #   # 关键：失败时不重算 pregrasp，直接使用缓存的 _pregrasp_target 重试抓取
    # self.target_pos[:] = torch.where(
    #   grasped.unsqueeze(-1),
    #   self._lift_target_pos,
    #   self._pregrasp_target,
    # )
    # # 指标记录（可选但建议保留）
    # self.metrics["grasped"] = grasped.float()
    
    # # 没抓住时候的z目标
    # obj_pos = torch.nan_to_num(self.object.data.root_link_pos_w[env_ids])# 物体当前的位置
    # pre = obj_pos.clone()
    # pre[:, 0] = obj_pos[:, 0] + self.cfg.pregrasp_offset_x
    # pre[:, 1] = obj_pos[:, 1] + self.cfg.pregrasp_offset_y
    # pre[:, 2] = obj_pos[:, 2] + self.cfg.pregrasp_offset_z
    # self._pregrasp_target[env_ids] = pre# 如果抓住了，目标z就是提升目标z；如果没抓住，目标z就是物体当前z加上一个偏移，这样智能体就需要先学会如何抓住物体，然后才能学会提升物体。

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    for batch in env_indices:
      target_pos = self.target_pos[batch].cpu().numpy()
      visualizer.add_sphere(
        center=target_pos,
        radius=0.03,
        color=self.cfg.viz.target_color,
        label=f"target_position_{batch}",
      )


@dataclass(kw_only=True)
class LiftingCommandCfg(CommandTermCfg):
  entity_name: str
  success_threshold: float = 0.05
  difficulty: Literal["fixed", "dynamic"] = "fixed"
  # # 新增：抓取门控参数
  # grasp_confirm_frames: int = 5
  # grasp_sensor_name: str | None = None # 对应ContactSensor的名称，如果不为None，则表示使用抓取门控机制
  # grasp_min_force: float = 0.3 # 多大力算抓住
  # pregrasp_offset_x: float = -0.08
  # pregrasp_offset_y: float = 0.0
  # pregrasp_offset_z: float = 0.03 # 没抓住时候目标z相对物体z的偏移
  # latch_grasp: bool = True # 一旦抓住就锁存，直到重采样才解锁

  @dataclass
  class TargetPositionRangeCfg:
    """Configuration for target position sampling in dynamic mode."""

    x: tuple[float, float] = (0.3, 0.5)
    y: tuple[float, float] = (-0.2, 0.2)
    z: tuple[float, float] = (0.2, 0.4)

  # Only used in dynamic mode.
  target_position_range: TargetPositionRangeCfg = field(
    default_factory=TargetPositionRangeCfg
  )

  @dataclass
  class ObjectPoseRangeCfg:
    """Configuration for object pose sampling when resampling commands."""

    x: tuple[float, float] = (0.3, 0.35)
    y: tuple[float, float] = (-0.1, 0.1)
    z: tuple[float, float] = (0.02, 0.05)
    yaw: tuple[float, float] = (-math.pi, math.pi)

  object_pose_range: ObjectPoseRangeCfg | None = field(
    default_factory=ObjectPoseRangeCfg
  )

  @dataclass
  class VizCfg:
    target_color: tuple[float, float, float, float] = (1.0, 0.5, 0.0, 0.3)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> LiftingCommand:
    return LiftingCommand(self, env)
