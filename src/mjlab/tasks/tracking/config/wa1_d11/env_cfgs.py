"""WA1_D11 fixed-base tracking environment configuration."""

from mjlab.asset_zoo.robots import (
  WA1_D11_ACTION_SCALE,
  WA1_END_EFFECTOR_BODY_NAMES,
  WA1_TRACK_BODY_NAMES,
  WA1_TRACK_JOINT_NAMES,
  get_wa1_d11_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

_TRACKED_JOINT_CFG = SceneEntityCfg("robot", joint_names=WA1_TRACK_JOINT_NAMES)


def wa1_d11_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_wa1_d11_robot_cfg()}
  assert cfg.scene.entities["robot"].articulation is not None
  cfg.scene.entities["robot"].articulation.soft_joint_pos_limit_factor = 1.0
  cfg.scene.terrain = None
  cfg.scene.num_envs = 1 if play else 1024
  cfg.scene.sensors = ()

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = WA1_TRACK_JOINT_NAMES
  joint_pos_action.scale = WA1_D11_ACTION_SCALE

  for obs_group_name in ("actor", "critic"):
    obs_group = cfg.observations[obs_group_name]
    obs_group.terms.pop("base_lin_vel", None)
    obs_group.terms.pop("base_ang_vel", None)
    obs_group.terms["joint_pos"].params["asset_cfg"] = _TRACKED_JOINT_CFG
    obs_group.terms["joint_vel"].params["asset_cfg"] = _TRACKED_JOINT_CFG
  cfg.observations["actor"].terms["motion_anchor_pos_b"].noise = Unoise(
    n_min=-0.03, n_max=0.03
  )
  cfg.observations["actor"].terms["motion_anchor_ori_b"].noise = Unoise(
    n_min=-0.02, n_max=0.02
  )
  cfg.observations["actor"].terms["joint_pos"].noise = Unoise(
    n_min=-0.003, n_max=0.003
  )
  cfg.observations["actor"].terms["joint_vel"].noise = Unoise(
    n_min=-0.08, n_max=0.08
  )

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "BODY"
  motion_cmd.body_names = WA1_TRACK_BODY_NAMES
  motion_cmd.joint_names = WA1_TRACK_JOINT_NAMES
  motion_cmd.pose_range = {}
  motion_cmd.velocity_range = {}
  motion_cmd.joint_position_range = (-0.01, 0.01)

  cfg.events.pop("push_robot", None)
  cfg.events.pop("base_com", None)
  cfg.events.pop("foot_friction", None)
  cfg.events["encoder_bias"].params["asset_cfg"] = _TRACKED_JOINT_CFG
  cfg.events["encoder_bias"].params["bias_range"] = (-0.001, 0.001)

  cfg.rewards.pop("self_collisions", None)
  cfg.rewards["joint_limit"].params["asset_cfg"] = _TRACKED_JOINT_CFG
  cfg.rewards["motion_global_root_pos"].weight = 1.5
  cfg.rewards["motion_global_root_pos"].params["std"] = 0.18
  cfg.rewards["motion_global_root_ori"].weight = 1.0
  cfg.rewards["motion_global_root_ori"].params["std"] = 0.25
  cfg.rewards["motion_body_pos"].weight = 2.5
  cfg.rewards["motion_body_pos"].params["std"] = 0.18
  cfg.rewards["motion_body_ori"].weight = 2.0
  cfg.rewards["motion_body_ori"].params["std"] = 0.25
  cfg.rewards["motion_body_lin_vel"].weight = 0.75
  cfg.rewards["motion_body_lin_vel"].params["std"] = 0.6
  cfg.rewards["motion_body_ang_vel"].weight = 0.75
  cfg.rewards["motion_body_ang_vel"].params["std"] = 1.5
  cfg.rewards["motion_joint_pos"] = RewardTermCfg(
    func=mdp.motion_joint_position_error_exp,
    weight=2.0,
    params={"command_name": "motion", "std": 0.2},
  )
  cfg.rewards["motion_joint_vel"] = RewardTermCfg(
    func=mdp.motion_joint_velocity_error_exp,
    weight=0.5,
    params={"command_name": "motion", "std": 1.2},
  )
  cfg.rewards["action_rate_l2"].weight = -0.005

  cfg.terminations["anchor_pos"].params["threshold"] = 0.30
  cfg.terminations["anchor_ori"].params["threshold"] = 1.0
  cfg.terminations["ee_body_pos"].params["body_names"] = WA1_END_EFFECTOR_BODY_NAMES
  cfg.terminations["ee_body_pos"].params["threshold"] = 0.35

  cfg.viewer.body_name = "CHEST"
  cfg.viewer.distance = 2.6
  cfg.viewer.azimuth = 180.0
  cfg.viewer.elevation = -18.0

  cfg.sim.nconmax = 120
  cfg.sim.njmax = 1200
  cfg.episode_length_s = 8.0

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("encoder_bias", None)
    cfg.terminations = {}
    motion_cmd.joint_position_range = (0.0, 0.0)
    motion_cmd.sampling_mode = "start"

  return cfg
