from __future__ import annotations

import copy
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import wandb
from rsl_rl.utils import check_nan
from tensordict import TensorDict

from mjlab.asset_zoo.robots import (
  WA1_D11_ACTION_SCALE,
  WA1_D11_MANIPULATION_ACTION_SCALE,
  WA1_LEFT_GRASP_SITE_NAME,
  WA1_RIGHT_GRASP_SITE_NAME,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.observation_manager import ObservationManager
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


class ManipulationOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    super().__init__(env, train_cfg, log_dir, device)
    self._teacher_policy = None
    self._teacher_obs_manager: ObservationManager | None = None
    self._teacher_to_student_ratio: torch.Tensor | None = None
    self._teacher_side_specs: list[dict[str, Any]] = []
    self._teacher_guidance_enabled = False
    self._init_teacher_guidance()

  @staticmethod
  def _make_scale_tensor(
    names: list[str],
    scale_map: dict[str, float],
    device: str,
  ) -> torch.Tensor:
    return torch.tensor([scale_map[name] for name in names], device=device)

  def _load_teacher_policy(
    self,
    task_id: str,
    checkpoint_file: Path,
    motion_file: str | None,
  ):
    env_cfg = load_env_cfg(task_id, play=True)
    if motion_file and "motion" in env_cfg.commands:
      env_cfg.commands["motion"].motion_file = motion_file

    agent_cfg = load_rl_cfg(task_id)
    teacher_env = ManagerBasedRlEnv(cfg=env_cfg, device=self.device)
    wrapped = RslRlVecEnvWrapper(teacher_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device=self.device)
    runner.load(
      str(checkpoint_file),
      load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False},
      strict=True,
      map_location=self.device,
    )
    policy = runner.get_inference_policy(device=self.device)
    wrapped.close()
    return policy

  def _init_teacher_guidance(self) -> None:
    teacher_ckpt = self.cfg.get("teacher_checkpoint_file")
    base_weight = float(self.cfg.get("teacher_guidance_weight", 0.0))
    if not teacher_ckpt or base_weight <= 0.0:
      return

    checkpoint_file = Path(str(teacher_ckpt)).expanduser().resolve()
    if not checkpoint_file.exists():
      print(f"[WARN] Teacher checkpoint not found, disabling guidance: {checkpoint_file}")
      return

    teacher_task_id = self.cfg.get("teacher_task_id") or "Mjlab-Tracking-FixedBase-WA1-D11"
    teacher_motion_file = self.cfg.get("teacher_motion_file")

    try:
      teacher_env_cfg = load_env_cfg(teacher_task_id, play=True)
      if teacher_motion_file and "motion" in teacher_env_cfg.commands:
        teacher_env_cfg.commands["motion"].motion_file = str(
          Path(str(teacher_motion_file)).expanduser().resolve()
        )

      if "motion" in teacher_env_cfg.commands:
        motion_cfg = copy.deepcopy(teacher_env_cfg.commands["motion"])
        motion_term = motion_cfg.build(self.env.unwrapped)
        motion_term.cfg.debug_vis = False
        motion_term.time_left.fill_(1e9)
        self.env.unwrapped.command_manager.cfg["motion"] = motion_cfg
        self.env.unwrapped.command_manager._terms["motion"] = motion_term

      tracking_actor_cfg = copy.deepcopy(teacher_env_cfg.observations["actor"])
      self._teacher_obs_manager = ObservationManager(
        {"actor": tracking_actor_cfg}, self.env.unwrapped
      )
      self._teacher_policy = self._load_teacher_policy(
        str(teacher_task_id), checkpoint_file, teacher_motion_file
      )

      joint_pos_action = self.env.unwrapped.action_manager.get_term("joint_pos")
      target_names = joint_pos_action.target_names
      track_scale = self._make_scale_tensor(
        target_names, WA1_D11_ACTION_SCALE, self.device
      )
      manipulation_scale = self._make_scale_tensor(
        target_names, WA1_D11_MANIPULATION_ACTION_SCALE, self.device
      )
      ratio = torch.zeros_like(track_scale)
      nonzero = torch.abs(manipulation_scale) > 1e-8
      ratio[nonzero] = track_scale[nonzero] / manipulation_scale[nonzero]
      self._teacher_to_student_ratio = ratio

      robot = self.env.unwrapped.scene["robot"]
      side_defs = (
        ("cylinder_right", WA1_RIGHT_GRASP_SITE_NAME, "right_grasp_contact"),
        ("cylinder_left", WA1_LEFT_GRASP_SITE_NAME, "left_grasp_contact"),
      )
      for object_name, site_name, sensor_name in side_defs:
        if object_name not in self.env.unwrapped.scene.entities:
          continue
        if sensor_name not in self.env.unwrapped.scene.sensors:
          continue
        site_id = robot.find_sites((site_name,), preserve_order=True)[0][0]
        start_height = float(
          self.env.unwrapped.scene[object_name].data.root_link_pos_w[:, 2].mean().item()
        )
        self._teacher_side_specs.append(
          {
            "object_name": object_name,
            "site_id": site_id,
            "sensor_name": sensor_name,
            "start_height": start_height,
          }
        )

      if not self._teacher_side_specs:
        print("[WARN] No valid object/site/sensor pair found for teacher guidance.")
        self._teacher_obs_manager = None
        self._teacher_policy = None
        self._teacher_to_student_ratio = None
        return

      self._teacher_guidance_enabled = True
      print(f"[INFO] Teacher guidance enabled from checkpoint: {checkpoint_file}")
    except Exception as exc:
      self._teacher_obs_manager = None
      self._teacher_policy = None
      self._teacher_to_student_ratio = None
      self._teacher_side_specs = []
      self._teacher_guidance_enabled = False
      print(f"[WARN] Failed to initialize teacher guidance: {exc}")

  def _teacher_anneal_weight(self, iteration: int) -> float:
    base_weight = float(self.cfg.get("teacher_guidance_weight", 0.0))
    start_iter = int(self.cfg.get("teacher_anneal_start_iter", 0))
    end_iter = int(self.cfg.get("teacher_anneal_end_iter", start_iter))
    if iteration <= start_iter:
      return base_weight
    if end_iter <= start_iter:
      return 0.0
    if iteration >= end_iter:
      return 0.0
    alpha = float(iteration - start_iter) / float(end_iter - start_iter)
    return base_weight * (1.0 - alpha)

  def _compute_teacher_phase_gate(self) -> torch.Tensor:
    assert self._teacher_side_specs
    robot = self.env.unwrapped.scene["robot"]
    dist_far = float(self.cfg.get("teacher_dist_far", 0.20))
    dist_near = float(self.cfg.get("teacher_dist_near", 0.08))
    release_height = float(self.cfg.get("teacher_release_height", 0.04))
    post_grasp_scale = float(self.cfg.get("teacher_post_grasp_scale", 0.03))
    contact_force_threshold = float(self.cfg.get("teacher_contact_force_threshold", 0.35))

    gates: list[torch.Tensor] = []
    denom = max(dist_far - dist_near, 1e-6)
    for spec in self._teacher_side_specs:
      obj = self.env.unwrapped.scene[spec["object_name"]]
      sensor = self.env.unwrapped.scene[spec["sensor_name"]]
      ee_pos = robot.data.site_pos_w[:, spec["site_id"]]
      obj_pos = obj.data.root_link_pos_w
      distance = torch.norm(ee_pos - obj_pos, dim=-1)
      dist_gate = ((distance - dist_near) / denom).clamp(0.0, 1.0)

      data = sensor.data
      if data.force is not None:
        force_mag = torch.norm(torch.nan_to_num(data.force), dim=-1).amax(dim=-1)
        grasped = force_mag > contact_force_threshold
      elif data.found is not None:
        grasped = torch.nan_to_num(data.found).amax(dim=-1) > 0
      else:
        grasped = torch.zeros_like(dist_gate, dtype=torch.bool)

      obj_z = torch.nan_to_num(obj.data.root_link_pos_w[:, 2])
      lifted = obj_z > (float(spec["start_height"]) + release_height)
      phase_scale = torch.where(
        grasped | lifted,
        torch.full_like(dist_gate, post_grasp_scale),
        torch.ones_like(dist_gate),
      )
      gates.append(dist_gate * phase_scale)

    return torch.stack(gates, dim=0).mean(dim=0)

  def _teacher_guidance_reward(
    self,
    student_actions: torch.Tensor,
    iteration: int,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not self._teacher_guidance_enabled:
      zeros = torch.zeros(student_actions.shape[0], device=student_actions.device)
      return zeros, zeros, zeros
    if self._teacher_policy is None or self._teacher_obs_manager is None:
      zeros = torch.zeros(student_actions.shape[0], device=student_actions.device)
      return zeros, zeros, zeros
    if self._teacher_to_student_ratio is None:
      zeros = torch.zeros(student_actions.shape[0], device=student_actions.device)
      return zeros, zeros, zeros

    weight = self._teacher_anneal_weight(iteration)
    if weight <= 0.0:
      zeros = torch.zeros(student_actions.shape[0], device=student_actions.device)
      return zeros, zeros, zeros

    with torch.inference_mode():
      obs_tensor = self._teacher_obs_manager.compute_group("actor", update_history=True)
      teacher_obs = TensorDict({"actor": obs_tensor}, batch_size=[self.env.num_envs])
      teacher_action = self._teacher_policy(teacher_obs) * self._teacher_to_student_ratio

    guidance_std = float(self.cfg.get("teacher_guidance_std", 0.35))
    action_error = torch.mean(torch.square(student_actions - teacher_action), dim=-1)
    match_reward = torch.exp(-action_error / (guidance_std**2))
    phase_gate = self._compute_teacher_phase_gate()
    reward = weight * phase_gate * match_reward
    return reward, phase_gate, action_error

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )

    obs = self.env.get_observations().to(self.device)
    self.alg.train_mode()

    if self.is_distributed:
      print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
      self.alg.broadcast_parameters()

    self.logger.init_logging_writer()

    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    for it in range(start_it, total_it):
      start = time.time()
      teacher_reward_log = 0.0
      teacher_gate_log = 0.0
      teacher_error_log = 0.0

      with torch.inference_mode():
        for _ in range(self.cfg["num_steps_per_env"]):
          actions = self.alg.act(obs)
          executed_actions = actions
          clip_actions = self.env.clip_actions
          if clip_actions is not None:
            executed_actions = torch.clamp(actions, -clip_actions, clip_actions)

          teacher_reward, teacher_gate, teacher_error = self._teacher_guidance_reward(
            executed_actions,
            iteration=it,
          )

          obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
          if self.cfg.get("check_for_nan", True):
            check_nan(obs, rewards, dones)

          rewards = rewards + teacher_reward.to(rewards.device)

          obs, rewards, dones = (
            obs.to(self.device),
            rewards.to(self.device),
            dones.to(self.device),
          )
          self.alg.process_env_step(obs, rewards, dones, extras)
          intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
          self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

          teacher_reward_log += float(teacher_reward.mean().item())
          teacher_gate_log += float(teacher_gate.mean().item())
          teacher_error_log += float(teacher_error.mean().item())

        stop = time.time()
        collect_time = stop - start
        start = stop
        self.alg.compute_returns(obs)

      loss_dict = self.alg.update()

      step_count = max(self.cfg["num_steps_per_env"], 1)
      loss_dict["teacher_reward"] = teacher_reward_log / step_count
      loss_dict["teacher_gate"] = teacher_gate_log / step_count
      loss_dict["teacher_action_error"] = teacher_error_log / step_count
      loss_dict["teacher_weight"] = self._teacher_anneal_weight(it)

      stop = time.time()
      learn_time = stop - start
      self.current_learning_iteration = it

      self.logger.log(
        it=it,
        start_it=start_it,
        total_it=total_it,
        collect_time=collect_time,
        learn_time=learn_time,
        loss_dict=loss_dict,
        learning_rate=self.alg.learning_rate,
        action_std=self.alg.get_policy().output_std,
        rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
      )

      if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
        self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore[arg-type]

    if self.logger.writer is not None:
      self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore[arg-type]
      self.logger.stop_logging_writer()

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_dir, filename, onnx_path = self._get_export_paths(path)
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      run_name: str = (
        wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
      )  # type: ignore[assignment]
      metadata = get_base_metadata(self.env.unwrapped, run_name)
      attach_metadata_to_onnx(str(onnx_path), metadata)
      if self.logger.logger_type in ["wandb"] and self.cfg["upload_model"]:
        wandb.save(
          str(onnx_path),
          base_path=str(policy_dir),
        )
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")
