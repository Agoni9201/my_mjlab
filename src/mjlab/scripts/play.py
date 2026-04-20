"""Script to play RL agent with RSL-RL."""

import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  registry_name: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the W&B run to load (e.g. 'model_4000.pt')."""
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  play_stage: Literal["default", "match_checkpoint", "easy", "mid", "final"] = "default"
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""
  print_grasp_contact_force: bool = False
  """Print left/right grasp contact force during play for diagnostics."""
  contact_print_interval: int = 10
  """Number of env steps between force printouts."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


class _ContactForceDebugPolicy:
  """Policy wrapper that periodically prints grasp contact forces."""

  def __init__(
    self,
    base_policy,
    env: ManagerBasedRlEnv,
    interval: int,
  ) -> None:
    self._base_policy = base_policy
    self._env = env
    self._interval = max(1, int(interval))
    self._step = 0
    self._init_right_z: float | None = None
    self._init_left_z: float | None = None

  def __call__(self, obs) -> torch.Tensor:
    action = self._base_policy(obs)
    if self._step % self._interval == 0:
      self._print_contact_forces()
    self._step += 1
    return action

  def _print_contact_forces(self) -> None:
    right_force, right_found = self._extract_force_and_found("right_grasp_contact")
    left_force, left_found = self._extract_force_and_found("left_grasp_contact")
    right_z = self._extract_object_height("cylinder_right")
    left_z = self._extract_object_height("cylinder_left")
    right_status = self._diagnose_hand(
      hand="right",
      force=right_force,
      found=right_found,
      object_z=right_z,
    )
    left_status = self._diagnose_hand(
      hand="left",
      force=left_force,
      found=left_found,
      object_z=left_z,
    )
    print(
      "[CONTACT] "
      f"step={self._step:06d} "
      f"right_force={right_force:7.3f}N right_found={right_found:4.1f} "
      f"left_force={left_force:7.3f}N left_found={left_found:4.1f} "
      f"right_z={right_z:6.3f} left_z={left_z:6.3f} "
      f"right_state={right_status} left_state={left_status}"
    )

  def _extract_force_and_found(self, sensor_name: str) -> tuple[float, float]:
    if sensor_name not in self._env.scene.sensors:
      return float("nan"), float("nan")

    sensor = self._env.scene[sensor_name]
    data = sensor.data
    force = getattr(data, "force", None)
    found = getattr(data, "found", None)

    force_mag = float("nan")
    if force is not None:
      # Use env 0 and take max magnitude over slots/geoms as a compact signal.
      force_env0 = force[0]
      if force_env0.ndim == 1:
        force_mag = float(torch.linalg.vector_norm(force_env0).item())
      else:
        force_mag = float(torch.linalg.vector_norm(force_env0, dim=-1).max().item())

    found_val = float("nan")
    if found is not None:
      found_env0 = found[0]
      if found_env0.ndim == 0:
        found_val = float(found_env0.item())
      else:
        found_val = float(found_env0.max().item())

    return force_mag, found_val

  def _extract_object_height(self, object_name: str) -> float:
    if object_name not in self._env.scene.entities:
      return float("nan")
    obj = self._env.scene[object_name]
    return float(obj.data.root_link_pos_w[0, 2].item())

  def _diagnose_hand(
    self,
    hand: Literal["right", "left"],
    force: float,
    found: float,
    object_z: float,
  ) -> str:
    # Thresholds tuned for quick online diagnostics in play mode.
    found_th = 0.5
    weak_force_th = 2.0
    lift_delta_th = 0.03

    init_z = self._init_right_z if hand == "right" else self._init_left_z
    if init_z is None and object_z == object_z:  # NaN-safe check.
      if hand == "right":
        self._init_right_z = object_z
      else:
        self._init_left_z = object_z
      init_z = object_z

    lift_delta = 0.0
    if init_z is not None and object_z == object_z:
      lift_delta = object_z - init_z

    has_contact = (found == found and found >= found_th) or (
      force == force and force >= weak_force_th
    )

    if not has_contact:
      return "碰不到"
    if force == force and force < weak_force_th:
      return "夹不住"
    if lift_delta < lift_delta_th:
      return "抬不动"
    return "抬起来"


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
      # Check if the registry name includes alias, if not, append ":latest".
      registry_name = cfg.registry_name
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
    else:
      if cfg.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {cfg.motion_file}")
        motion_cmd.motion_file = cfg.motion_file
      else:
        import wandb

        api = wandb.Api()
        if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
          raise ValueError(
            "Tracking tasks require `motion_file` when using `checkpoint_file`, "
            "or provide `wandb_run_path` so the motion artifact can be resolved."
          )
        if cfg.wandb_run_path is not None:
          wandb_run = api.run(str(cfg.wandb_run_path))
          art = next(
            (a for a in wandb_run.used_artifacts() if a.type == "motions"), None
          )
          if art is None:
            raise RuntimeError("No motion artifact found in the run.")
          motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")

  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if task_id == "Mjlab-Grasp-Cylinder-WA1-D11":
    from mjlab.tasks.manipulation.config.wa1_d11.env_cfgs import (
      apply_wa1_d11_grasp_cylinder_stage,
    )

    def _stage_from_checkpoint(path: Path) -> Literal["easy", "mid", "final"]:
      match = re.search(r"model_(\d+)\.pt$", path.name)
      if match is None:
        return "final"
      iteration = int(match.group(1))
      if iteration < 400:
        return "easy"
      if iteration < 1200:
        return "mid"
      return "final"

    requested_stage = cfg.play_stage
    if requested_stage == "default" and resume_path is not None:
      requested_stage = "match_checkpoint"
    if requested_stage == "match_checkpoint":
      if resume_path is None:
        requested_stage = "final"
      else:
        requested_stage = _stage_from_checkpoint(resume_path)
    if requested_stage != "default":
      assert requested_stage in {"easy", "mid", "final"}
      apply_wa1_d11_grasp_cylinder_stage(env_cfg, requested_stage)
      print(f"[INFO]: Using play stage: {requested_stage}")

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

  if cfg.print_grasp_contact_force:
    scene_sensors = env.unwrapped.scene.sensors
    required = {"right_grasp_contact", "left_grasp_contact"}
    if required.issubset(scene_sensors.keys()):
      policy = _ContactForceDebugPolicy(
        base_policy=policy,
        env=env.unwrapped,
        interval=cfg.contact_print_interval,
      )
      print(
        "[INFO]: Grasp contact force debug enabled "
        f"(interval={max(1, cfg.contact_print_interval)} steps)"
      )
    else:
      print(
        "[WARN]: Grasp contact force debug requested, but required sensors "
        f"are missing. Available sensors: {list(scene_sensors.keys())}"
      )

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
