from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np
import torch
import tyro
from tqdm import tqdm

import mjlab
from mjlab.entity import Entity, EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_conjugate,
  quat_mul,
  quat_slerp,
)
from mjlab.utils.os import update_assets
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig


DEFAULT_WA1_D11_XML = (
  "/home/robot706/yx/mjlab_v3/src/mjlab/asset_zoo/robots/wa1_d11/xml/WA1_D11.xml"
)
DEFAULT_WA1_D11_ASSET_DIR = (
  "/home/robot706/yx/mjlab_v3/src/mjlab/asset_zoo/robots/wa1_d11/xml/meshes"
)


def _build_robot_spec_fn(
  robot_xml: str,
  robot_asset_dir: str | None,
):
  robot_xml_path = Path(robot_xml).expanduser().resolve()
  asset_dir_override = (
    Path(robot_asset_dir).expanduser().resolve()
    if robot_asset_dir is not None
    else None
  )

  def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(robot_xml_path))
    assets: dict[str, bytes] = {}
    meshdir = spec.meshdir

    if asset_dir_override is not None:
      asset_root = asset_dir_override
    elif meshdir:
      meshdir_path = Path(meshdir)
      if not meshdir_path.is_absolute():
        meshdir_path = robot_xml_path.parent / meshdir_path
      asset_root = meshdir_path
    else:
      asset_root = robot_xml_path.parent

    if asset_root.exists():
      update_assets(assets, asset_root, meshdir, recursive=True)

    spec.assets = assets
    return spec

  return get_spec


def _build_scene(
  device: str,
  robot_xml: str | None,
  robot_asset_dir: str | None,
) -> Scene:
  if robot_xml is None:
    return Scene(unitree_g1_flat_tracking_env_cfg().scene, device=device)

  scene_cfg = SceneCfg(
    entities={
      "robot": EntityCfg(
        spec_fn=_build_robot_spec_fn(robot_xml, robot_asset_dir),
      )
    }
  )
  return Scene(scene_cfg, device=device)


def _default_viewer_distance(mj_model: mujoco.MjModel) -> float:
  # Scale the camera distance with the model extent so larger robots don't fill
  # the whole preview frame.
  return max(3.5, float(mj_model.stat.extent) * 1.8)


class MotionLoader:
  def __init__(
    self,
    motion_file: str,
    input_fps: int,
    output_fps: int,
    device: torch.device | str,
    motion_mode: Literal["floating_base", "fixed_base"],
    selected_joint_count: int | None = None,
    line_range: tuple[int, int] | None = None,
  ):
    self.motion_file = motion_file
    self.input_fps = input_fps
    self.output_fps = output_fps
    self.input_dt = 1.0 / self.input_fps
    self.output_dt = 1.0 / self.output_fps
    self.current_idx = 0
    self.device = device
    self.motion_mode = motion_mode
    self.selected_joint_count = selected_joint_count
    self.line_range = line_range
    self._load_motion()
    self._interpolate_motion()
    self._compute_velocities()

  def _load_motion(self):
    """Loads the motion from the csv file."""
    if self.line_range is None:
      motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
    else:
      motion = torch.from_numpy(
        np.loadtxt(
          self.motion_file,
          delimiter=",",
          skiprows=self.line_range[0] - 1,
          max_rows=self.line_range[1] - self.line_range[0] + 1,
        )
      )
    if motion.ndim == 1:
      motion = motion.unsqueeze(0)
    motion = motion.to(torch.float32).to(self.device)
    if self.motion_mode == "floating_base":
      self.motion_base_poss_input = motion[:, :3]
      self.motion_base_rots_input = motion[:, 3:7]
      self.motion_base_rots_input = self.motion_base_rots_input[
        :, [3, 0, 1, 2]
      ]  # convert to wxyz
      self.motion_dof_poss_input = motion[:, 7:]
    else:
      self.motion_base_poss_input = None
      self.motion_base_rots_input = None
      if self.selected_joint_count is None:
        self.motion_dof_poss_input = motion
      elif motion.shape[1] == self.selected_joint_count:
        self.motion_dof_poss_input = motion
      elif motion.shape[1] == self.selected_joint_count + 7:
        # Allow fixed-base playback from floating-base CSV files by dropping
        # [base_pos(3), base_quat_xyzw(4)] columns.
        self.motion_dof_poss_input = motion[:, 7:]
      else:
        raise ValueError(
          "Invalid CSV column count for fixed_base mode: "
          f"columns={motion.shape[1]}, "
          f"expected_joints={self.selected_joint_count}, "
          f"or expected_joints_plus_base={self.selected_joint_count + 7}."
        )

    self.input_frames = motion.shape[0]
    self.duration = (self.input_frames - 1) * self.input_dt

  def _interpolate_motion(self):
    """Interpolates the motion to the output fps."""
    times = torch.arange(
      0, self.duration, self.output_dt, device=self.device, dtype=torch.float32
    )
    self.output_frames = times.shape[0]
    index_0, index_1, blend = self._compute_frame_blend(times)
    if self.motion_mode == "floating_base":
      assert self.motion_base_poss_input is not None
      assert self.motion_base_rots_input is not None
      self.motion_base_poss = self._lerp(
        self.motion_base_poss_input[index_0],
        self.motion_base_poss_input[index_1],
        blend.unsqueeze(1),
      )
      self.motion_base_rots = self._slerp(
        self.motion_base_rots_input[index_0],
        self.motion_base_rots_input[index_1],
        blend,
      )
    else:
      self.motion_base_poss = None
      self.motion_base_rots = None
    self.motion_dof_poss = self._lerp(
      self.motion_dof_poss_input[index_0],
      self.motion_dof_poss_input[index_1],
      blend.unsqueeze(1),
    )
    print(
      f"Motion interpolated, input frames: {self.input_frames}, "
      f"input fps: {self.input_fps}, "
      f"output frames: {self.output_frames}, "
      f"output fps: {self.output_fps}"
    )

  def _lerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    """Linear interpolation between two tensors."""
    return a * (1 - blend) + b * blend

  def _slerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    """Spherical linear interpolation between two quaternions."""
    slerped_quats = torch.zeros_like(a)
    for i in range(a.shape[0]):
      slerped_quats[i] = quat_slerp(a[i], b[i], float(blend[i]))
    return slerped_quats

  def _compute_frame_blend(
    self, times: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes the frame blend for the motion."""
    phase = times / self.duration
    index_0 = (phase * (self.input_frames - 1)).floor().long()
    index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
    blend = phase * (self.input_frames - 1) - index_0
    return index_0, index_1, blend

  def _compute_velocities(self):
    """Computes the velocities of the motion."""
    self.motion_dof_vels = torch.gradient(
      self.motion_dof_poss, spacing=self.output_dt, dim=0
    )[0]
    if self.motion_mode == "floating_base":
      assert self.motion_base_poss is not None
      assert self.motion_base_rots is not None
      self.motion_base_lin_vels = torch.gradient(
        self.motion_base_poss, spacing=self.output_dt, dim=0
      )[0]
      self.motion_base_ang_vels = self._so3_derivative(
        self.motion_base_rots, self.output_dt
      )
    else:
      self.motion_base_lin_vels = None
      self.motion_base_ang_vels = None

  def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
    """Computes the derivative of a sequence of SO3 rotations.

    Args:
      rotations: shape (B, 4).
      dt: time step.
    Returns:
      shape (B, 3).
    """
    q_prev, q_next = rotations[:-2], rotations[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

    omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
    omega = torch.cat(
      [omega[:1], omega, omega[-1:]], dim=0
    )  # repeat first and last sample
    return omega

  def get_next_state(
    self,
  ) -> tuple[
    tuple[
      torch.Tensor | None,
      torch.Tensor | None,
      torch.Tensor | None,
      torch.Tensor | None,
      torch.Tensor,
      torch.Tensor,
    ],
    bool,
  ]:
    """Gets the next state of the motion."""
    state = (
      None
      if self.motion_base_poss is None
      else self.motion_base_poss[self.current_idx : self.current_idx + 1],
      None
      if self.motion_base_rots is None
      else self.motion_base_rots[self.current_idx : self.current_idx + 1],
      None
      if self.motion_base_lin_vels is None
      else self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
      None
      if self.motion_base_ang_vels is None
      else self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
      self.motion_dof_poss[self.current_idx : self.current_idx + 1],
      self.motion_dof_vels[self.current_idx : self.current_idx + 1],
    )
    self.current_idx += 1
    reset_flag = False
    if self.current_idx >= self.output_frames:
      self.current_idx = 0
      reset_flag = True
    return state, reset_flag


def run_sim(
  sim: Simulation,
  scene: Scene,
  joint_names: tuple[str, ...] | None,
  input_file: str,
  input_fps: float,
  output_fps: float,
  output_file: str,
  output_video: str | None,
  output_name: str | None,
  render: bool,
  line_range: tuple[int, int] | None,
  motion_mode: Literal["floating_base", "fixed_base"],
  upload_to_wandb: bool,
  renderer: OffscreenRenderer | None = None,
):
  robot: Entity = scene["robot"]
  if joint_names is None:
    joint_names = tuple(robot.joint_names)
  robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

  motion = MotionLoader(
    motion_file=input_file,
    input_fps=input_fps,
    output_fps=output_fps,
    device=sim.device,
    motion_mode=motion_mode,
    selected_joint_count=len(robot_joint_indexes),
    line_range=line_range,
  )

  if motion.motion_dof_poss_input.shape[1] != len(robot_joint_indexes):
    raise ValueError(
      "CSV joint dimension does not match selected robot joints: "
      f"csv={motion.motion_dof_poss_input.shape[1]}, "
      f"selected_joints={len(robot_joint_indexes)}."
    )

  log: dict[str, Any] = {
    "fps": [output_fps],
    "joint_names": np.asarray(
      [robot.joint_names[idx] for idx in robot_joint_indexes], dtype=str
    ),
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
  }
  file_saved = False

  frames = []
  scene.reset()

  print(f"\nStarting simulation with {motion.output_frames} frames...")
  if render:
    print("Rendering enabled - generating video frames...")

  # Create progress bar
  pbar = tqdm(
    total=motion.output_frames,
    desc="Processing frames",
    unit="frame",
    ncols=100,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
  )

  frame_count = 0
  while not file_saved:
    (
      (
        motion_base_pos,
        motion_base_rot,
        motion_base_lin_vel,
        motion_base_ang_vel,
        motion_dof_pos,
        motion_dof_vel,
      ),
      reset_flag,
    ) = motion.get_next_state()

    if motion_mode == "floating_base":
      assert motion_base_pos is not None
      assert motion_base_rot is not None
      assert motion_base_lin_vel is not None
      assert motion_base_ang_vel is not None
      root_states = robot.data.default_root_state.clone()
      root_states[:, 0:3] = motion_base_pos
      root_states[:, :2] += scene.env_origins[:, :2]
      root_states[:, 3:7] = motion_base_rot
      root_states[:, 7:10] = motion_base_lin_vel
      root_states[:, 10:] = motion_base_ang_vel
      robot.write_root_state_to_sim(root_states)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, robot_joint_indexes] = motion_dof_pos
    joint_vel[:, robot_joint_indexes] = motion_dof_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)
    if render and renderer is not None:
      renderer.update(sim.data)
      frames.append(renderer.render())

    if not file_saved:
      log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
      log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
      log["body_pos_w"].append(robot.data.body_link_pos_w[0, :].cpu().numpy().copy())
      log["body_quat_w"].append(robot.data.body_link_quat_w[0, :].cpu().numpy().copy())
      log["body_lin_vel_w"].append(
        robot.data.body_link_lin_vel_w[0, :].cpu().numpy().copy()
      )
      log["body_ang_vel_w"].append(
        robot.data.body_link_ang_vel_w[0, :].cpu().numpy().copy()
      )

      if motion_mode == "floating_base":
        assert motion_base_lin_vel is not None
        assert motion_base_ang_vel is not None
        torch.testing.assert_close(
          robot.data.body_link_lin_vel_w[0, 0], motion_base_lin_vel[0]
        )
        torch.testing.assert_close(
          robot.data.body_link_ang_vel_w[0, 0], motion_base_ang_vel[0]
        )

      frame_count += 1
      pbar.update(1)

      if frame_count % 100 == 0:  # Update every 100 frames to avoid spam
        elapsed_time = frame_count / output_fps
        pbar.set_description(f"Processing frames (t={elapsed_time:.1f}s)")

      if reset_flag and not file_saved:
        file_saved = True
        pbar.close()

        print("\nStacking arrays and saving data...")
        for k in (
          "joint_pos",
          "joint_vel",
          "body_pos_w",
          "body_quat_w",
          "body_lin_vel_w",
          "body_ang_vel_w",
        ):
          log[k] = np.stack(log[k], axis=0)

        output_path = Path(output_file).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving to {output_path}...")
        np.savez(output_path, **log)

        if upload_to_wandb:
          print("Uploading to Weights & Biases...")
          import wandb

          assert output_name is not None
          run = wandb.init(project="csv_to_npz", name=output_name)
          print(f"[INFO]: Logging motion to wandb: {output_name}")
          registry = "motions"
          logged_artifact = run.log_artifact(
            artifact_or_path=str(output_path),
            name=output_name,
            type=registry,
          )
          run.link_artifact(
            artifact=logged_artifact,
            target_path=f"wandb-registry-{registry}/{output_name}",
          )
          print(f"[INFO]: Motion saved to wandb registry: {registry}/{output_name}")

        if render:
          import mediapy as media

          video_path = (
            Path(output_video).expanduser().resolve()
            if output_video is not None
            else output_path.with_suffix(".mp4")
          )
          video_path.parent.mkdir(parents=True, exist_ok=True)
          print("Creating video...")
          media.write_video(str(video_path), frames, fps=output_fps)
          print(f"Saved preview video to {video_path}")

          if upload_to_wandb:
            import wandb

            print("Logging video to wandb...")
            wandb.log({"motion_video": wandb.Video(str(video_path), format="mp4")})

        if upload_to_wandb:
          import wandb

          wandb.finish()


def main(
  input_file: str,
  output_name: str | None = "motions",
  output_file: str = "/tmp/motion.npz",
  output_video: str | None = None,
  input_fps: float = 30.0,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  render: bool = True,
  line_range: tuple[int, int] | None = None,
  robot_xml: str | None = DEFAULT_WA1_D11_XML,
  robot_asset_dir: str | None = DEFAULT_WA1_D11_ASSET_DIR,
  joint_names: tuple[str, ...] | None = None,
  motion_mode: Literal["floating_base", "fixed_base"] = "fixed_base",
  upload_to_wandb: bool = True,
  viewer_distance: float | None = None,
  viewer_elevation: float = -18.0,
  viewer_azimuth: float = 180.0,
  viewer_height: int = 480,
  viewer_width: int = 640,
):
  """Replay motion from CSV file and output to npz file.

  Args:
    input_file: Path to the input CSV file.
    output_name: WandB collection name. Required only when upload_to_wandb=True.
    output_file: Path to the output npz file.
    output_video: Path to the preview video. Defaults to <output_file stem>.mp4.
    input_fps: Frame rate of the CSV file.
    output_fps: Desired output frame rate.
    device: Device to use.
    render: Whether to render the simulation and save a video.
    line_range: Range of lines to process from the CSV file.
    robot_xml: MuJoCo XML path. Defaults to the WA1_D11 XML used for your
      fixed-base conversion workflow.
    robot_asset_dir: Asset directory override. Defaults to the WA1_D11 mesh
      directory.
    joint_names: Joint names to read from the CSV, in order. Defaults to all robot
      joints in natural MuJoCo order.
    motion_mode: Whether the CSV stores [base pos, base quat_xyzw, joints] or the
      full fixed-base qpos vector. Defaults to fixed_base.
    upload_to_wandb: Whether to upload the generated NPZ to a WandB registry.
    viewer_distance: Camera distance for the preview renderer. Defaults to an
      extent-scaled distance based on the robot model.
    viewer_elevation: Camera elevation in degrees for the preview renderer.
    viewer_azimuth: Camera azimuth in degrees for the preview renderer.
    viewer_height: Preview video height in pixels.
    viewer_width: Preview video width in pixels.
  """
  if device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARNING]: CUDA is not available. Falling back to CPU. This may be slow.")
    device = "cpu"
  if upload_to_wandb and output_name is None:
    raise ValueError("output_name is required when upload_to_wandb=True.")

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps

  scene = _build_scene(
    device=device,
    robot_xml=robot_xml,
    robot_asset_dir=robot_asset_dir,
  )
  model = scene.compile()

  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)

  scene.initialize(sim.mj_model, sim.model, sim.data)

  renderer = None
  if render:
    resolved_viewer_distance = (
      viewer_distance
      if viewer_distance is not None
      else _default_viewer_distance(sim.mj_model)
    )
    viewer_cfg = ViewerConfig(
      height=viewer_height,
      width=viewer_width,
      origin_type=ViewerConfig.OriginType.ASSET_ROOT,
      entity_name="robot",
      distance=resolved_viewer_distance,
      elevation=viewer_elevation,
      azimuth=viewer_azimuth,
    )
    print(
      "Preview camera: "
      f"distance={resolved_viewer_distance:.2f}, "
      f"elevation={viewer_elevation:.1f}, "
      f"azimuth={viewer_azimuth:.1f}"
    )
    renderer = OffscreenRenderer(
      model=sim.mj_model,
      cfg=viewer_cfg,
      scene=scene,
    )
    renderer.initialize()

  run_sim(
    sim=sim,
    scene=scene,
    joint_names=joint_names
    if joint_names is not None or robot_xml is not None
    else (
      "left_hip_pitch_joint",
      "left_hip_roll_joint",
      "left_hip_yaw_joint",
      "left_knee_joint",
      "left_ankle_pitch_joint",
      "left_ankle_roll_joint",
      "right_hip_pitch_joint",
      "right_hip_roll_joint",
      "right_hip_yaw_joint",
      "right_knee_joint",
      "right_ankle_pitch_joint",
      "right_ankle_roll_joint",
      "waist_yaw_joint",
      "waist_roll_joint",
      "waist_pitch_joint",
      "left_shoulder_pitch_joint",
      "left_shoulder_roll_joint",
      "left_shoulder_yaw_joint",
      "left_elbow_joint",
      "left_wrist_roll_joint",
      "left_wrist_pitch_joint",
      "left_wrist_yaw_joint",
      "right_shoulder_pitch_joint",
      "right_shoulder_roll_joint",
      "right_shoulder_yaw_joint",
      "right_elbow_joint",
      "right_wrist_roll_joint",
      "right_wrist_pitch_joint",
      "right_wrist_yaw_joint",
    ),
    input_fps=input_fps,
    input_file=input_file,
    output_fps=output_fps,
    output_name=output_name,
    output_file=output_file,
    output_video=output_video,
    render=render,
    line_range=line_range,
    motion_mode=motion_mode,
    upload_to_wandb=upload_to_wandb,
    renderer=renderer,
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)