from __future__ import annotations

from typing import Literal, cast

import mujoco

from mjlab.tasks.manipulation.mdp import LiftingCommandCfg
from mjlab.asset_zoo.robots import (
  WA1_D11_MANIPULATION_ACTION_SCALE,
  WA1_FRONT_TABLE_POS,
  WA1_FRONT_TABLE_TOP_SIZE,
  WA1_HAND_CYLINDER_SIZE,
  WA1_LEFT_ARM_JOINT_NAMES,
  WA1_RIGHT_GRASP_SITE_NAME,
  WA1_RIGHT_ARM_JOINT_NAMES,
  WA1_LEFT_GRASP_SITE_NAME,
  WA1_TRACK_JOINT_NAMES,
  WA1_TORSO_JOINT_NAMES,
  get_wa1_d11_manipulation_robot_cfg,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import ObservationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

_TABLE_TOP_SURFACE_Z = WA1_FRONT_TABLE_POS[2] + WA1_FRONT_TABLE_TOP_SIZE[2]
_CYLINDER_HALF_LENGTH = WA1_HAND_CYLINDER_SIZE[1]
_CYLINDER_MASS = 0.08
_CYLINDER_CENTER_Z = _TABLE_TOP_SURFACE_Z + _CYLINDER_HALF_LENGTH
_OBJECT_X_RANGE = (0.64, 0.74)
_OBJECT_Y_RIGHT_RANGE = (-0.14, -0.04)
_OBJECT_Y_LEFT_RANGE = (0.04, 0.14)

_TARGET_X_RANGE = (0.60, 0.70)
_TARGET_Y_RIGHT_RANGE = (-0.12, -0.04)
_TARGET_Y_LEFT_RANGE = (0.04, 0.12)

_OBJECT_X_RANGE_EASY = (0.66, 0.72)
_OBJECT_X_RANGE_MID = (0.65, 0.73)
_OBJECT_Y_RIGHT_RANGE_EASY = (-0.10, -0.06)
_OBJECT_Y_LEFT_RANGE_EASY = (0.06, 0.10)
_OBJECT_Y_RIGHT_RANGE_MID = (-0.12, -0.05)
_OBJECT_Y_LEFT_RANGE_MID = (0.05, 0.12)

_TARGET_X_RANGE_EASY = (0.61, 0.67)
_TARGET_X_RANGE_MID = (0.60, 0.69)
_TARGET_Y_RIGHT_RANGE_EASY = (-0.09, -0.06)
_TARGET_Y_LEFT_RANGE_EASY = (0.06, 0.09)
_TARGET_Y_RIGHT_RANGE_MID = (-0.10, -0.05)
_TARGET_Y_LEFT_RANGE_MID = (0.05, 0.10)

_TARGET_Z_RANGE_EASY = (_CYLINDER_CENTER_Z + 0.08, _CYLINDER_CENTER_Z + 0.11)
_TARGET_Z_RANGE_MID = (_CYLINDER_CENTER_Z + 0.15, _CYLINDER_CENTER_Z + 0.18)
_TARGET_Z_RANGE = (_CYLINDER_CENTER_Z + 0.17, _CYLINDER_CENTER_Z + 0.20)


def _controlled_joint_cfg() -> SceneEntityCfg:
  return SceneEntityCfg("robot", joint_names=WA1_TRACK_JOINT_NAMES)


def _grasp_site_right_cfg() -> SceneEntityCfg:
  return SceneEntityCfg("robot", site_names=(WA1_RIGHT_GRASP_SITE_NAME,))

def _grasp_site_left_cfg() -> SceneEntityCfg:
  return SceneEntityCfg("robot", site_names=(WA1_LEFT_GRASP_SITE_NAME,))

def _torso_joint_cfg() -> SceneEntityCfg:
  return SceneEntityCfg("robot", joint_names=WA1_TORSO_JOINT_NAMES)

def _right_arm_joint_cfg() -> SceneEntityCfg:
  return SceneEntityCfg("robot", joint_names=WA1_RIGHT_ARM_JOINT_NAMES)


def _left_arm_joint_cfg() -> SceneEntityCfg:
  return SceneEntityCfg("robot", joint_names=WA1_LEFT_ARM_JOINT_NAMES)


def apply_wa1_d11_grasp_cylinder_stage(
  cfg: ManagerBasedRlEnvCfg,
  stage: Literal["easy", "mid", "final"],
  command_name: str = "lift_height_right",
) -> None:
  lift_cmd = cast(LiftingCommandCfg, cfg.commands[command_name])

  if command_name == "lift_height_right":
    y_target = {
      "easy": _TARGET_Y_RIGHT_RANGE_EASY,
      "mid": _TARGET_Y_RIGHT_RANGE_MID,
      "final": _TARGET_Y_RIGHT_RANGE,
    }
    y_object = {
      "easy": _OBJECT_Y_RIGHT_RANGE_EASY,
      "mid": _OBJECT_Y_RIGHT_RANGE_MID,
      "final": _OBJECT_Y_RIGHT_RANGE,
    }
  elif command_name == "lift_height_left":
    y_target = {
      "easy": _TARGET_Y_LEFT_RANGE_EASY,
      "mid": _TARGET_Y_LEFT_RANGE_MID,
      "final": _TARGET_Y_LEFT_RANGE,
    }
    y_object = {
      "easy": _OBJECT_Y_LEFT_RANGE_EASY,
      "mid": _OBJECT_Y_LEFT_RANGE_MID,
      "final": _OBJECT_Y_LEFT_RANGE,
    }
  else:
    raise ValueError(f"Unsupported command_name: {command_name}")
  
  if stage == "easy":
    lift_cmd.success_threshold = 0.10
    lift_cmd.target_position_range.x = _TARGET_X_RANGE_EASY
    lift_cmd.target_position_range.y = y_target["easy"]
    lift_cmd.target_position_range.z = _TARGET_Z_RANGE_EASY
    lift_cmd.object_pose_range.x = _OBJECT_X_RANGE_EASY
    lift_cmd.object_pose_range.y = y_object["easy"]
    lift_cmd.object_pose_range.yaw = (-0.15, 0.15)
  elif stage == "mid":
    lift_cmd.success_threshold = 0.05
    lift_cmd.target_position_range.x = _TARGET_X_RANGE_MID
    lift_cmd.target_position_range.y = y_target["mid"]
    lift_cmd.target_position_range.z = _TARGET_Z_RANGE_MID
    lift_cmd.object_pose_range.x = _OBJECT_X_RANGE_MID
    lift_cmd.object_pose_range.y = y_object["mid"]
    lift_cmd.object_pose_range.yaw = (-0.25, 0.25)
  else:
    lift_cmd.success_threshold = 0.04
    lift_cmd.target_position_range.x = _TARGET_X_RANGE
    lift_cmd.target_position_range.y = y_target["final"]  
    lift_cmd.target_position_range.z = _TARGET_Z_RANGE
    lift_cmd.object_pose_range.x = _OBJECT_X_RANGE
    lift_cmd.object_pose_range.y = y_object["final"]
    lift_cmd.object_pose_range.yaw = (-0.4, 0.4)
  lift_cmd.object_pose_range.z = (_CYLINDER_CENTER_Z, _CYLINDER_CENTER_Z)


def get_table_cylinder_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="table_cylinder")
  joint = body.add_freejoint(name="table_cylinder_joint")
  joint.damping = 0.5
  joint.armature = 0.001
  geom = body.add_geom(
    name="table_cylinder_geom",
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    size=WA1_HAND_CYLINDER_SIZE,
    mass=_CYLINDER_MASS,
    rgba=(0.88, 0.24, 0.18, 1.0),
  )
  geom.quat[:] = (1.0, 0.0, 0.0, 0.0)
  geom.friction[:] = (0.6, 0.01, 0.001)
  geom.condim = 4
  return spec


def get_table_support_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="table_support")
  geom = body.add_geom(
    name="table_support_top",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=WA1_FRONT_TABLE_TOP_SIZE,
    rgba=(0.0, 0.0, 0.0, 0.0),
  )
  geom.friction[:] = (1.0, 0.02, 0.002)
  geom.condim = 4
  return spec


def wa1_d11_grasp_cylinder_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_lift_cube_env_cfg()

  cfg.scene.entities = {
    "robot": get_wa1_d11_manipulation_robot_cfg(),
    "cylinder_right": EntityCfg(spec_fn=get_table_cylinder_spec),
    "cylinder_left": EntityCfg(spec_fn=get_table_cylinder_spec),
    "table_support": EntityCfg(
      init_state=EntityCfg.InitialStateCfg(pos=WA1_FRONT_TABLE_POS),
      spec_fn=get_table_support_spec,
    ),
  }
  assert cfg.scene.entities["robot"].articulation is not None
  cfg.scene.entities["robot"].articulation.soft_joint_pos_limit_factor = 1.0
  cfg.scene.terrain = None
  cfg.scene.num_envs = 1 if play else 1024

  right_ee_table_collision_cfg = ContactSensorCfg(
    name="right_ee_table_collision",
    primary=ContactMatch(mode="subtree", pattern="WRIST_FLANGE_R", entity="robot"),
    secondary=ContactMatch(mode="geom", pattern="table_support_top", entity="table_support"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  right_grasp_contact_cfg = ContactSensorCfg(
    name="right_grasp_contact",
    primary=ContactMatch(mode="subtree", pattern="WRIST_FLANGE_R", entity="robot"),
    secondary=ContactMatch(mode="geom", pattern="table_cylinder_geom", entity="cylinder_right"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
  )
  left_ee_table_collision_cfg = ContactSensorCfg(
    name="left_ee_table_collision",
    primary=ContactMatch(mode="subtree", pattern="WRIST_FLANGE_L", entity="robot"),
    secondary=ContactMatch(mode="geom", pattern="table_support_top", entity="table_support"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  left_grasp_contact_cfg = ContactSensorCfg(
    name="left_grasp_contact",
    primary=ContactMatch(mode="subtree", pattern="WRIST_FLANGE_L", entity="robot"),
    secondary=ContactMatch(mode="geom", pattern="table_cylinder_geom", entity="cylinder_left"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
  )

  cfg.scene.sensors = (
    right_ee_table_collision_cfg,
    right_grasp_contact_cfg,
    left_ee_table_collision_cfg,
    left_grasp_contact_cfg,
  )


  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = WA1_TRACK_JOINT_NAMES
  joint_pos_action.scale = WA1_D11_MANIPULATION_ACTION_SCALE

  cfg.commands["lift_height_right"].entity_name = "cylinder_right"
  cfg.commands["lift_height_left"].entity_name = "cylinder_left"
  cfg.commands["lift_height_right"].difficulty = "dynamic"
  cfg.commands["lift_height_left"].difficulty = "dynamic"
  cfg.commands["lift_height_right"].resampling_time_range = (20.0, 20.0)
  cfg.commands["lift_height_left"].resampling_time_range = (20.0, 20.0)
  apply_wa1_d11_grasp_cylinder_stage(cfg, "easy", command_name="lift_height_right")
  apply_wa1_d11_grasp_cylinder_stage(cfg, "easy", command_name="lift_height_left")

  for obs_group_name in ("actor", "critic"):
    obs_group = cfg.observations[obs_group_name]
    obs_group.terms["joint_pos"].params["asset_cfg"] = _controlled_joint_cfg()
    obs_group.terms["joint_vel"].params["asset_cfg"] = _controlled_joint_cfg()
    
    obs_group.terms["ee_to_cube_right"].params["object_name"] = "cylinder_right"
    obs_group.terms["ee_to_cube_right"].params["asset_cfg"] = _grasp_site_right_cfg()
    obs_group.terms["cube_to_goal_right"].params["object_name"] = "cylinder_right"
    
    obs_group.terms["ee_to_cube_left"].params["object_name"] = "cylinder_left"
    obs_group.terms["ee_to_cube_left"].params["asset_cfg"] = _grasp_site_left_cfg()
    obs_group.terms["cube_to_goal_left"].params["object_name"] = "cylinder_left"

  cfg.observations["actor"].terms["joint_pos"].noise = Unoise(
    n_min=-0.003, n_max=0.003
  )
  cfg.observations["actor"].terms["joint_vel"].noise = Unoise(
    n_min=-0.08, n_max=0.08
  )

  cfg.observations["actor"].terms["ee_to_cube_right"].noise = Unoise(
    n_min=-0.005, n_max=0.005
  )
  cfg.observations["actor"].terms["cube_to_goal_right"].noise = Unoise(
    n_min=-0.005, n_max=0.005
  )
  cfg.observations["actor"].terms["ee_to_cube_left"].noise = Unoise(
    n_min=-0.005, n_max=0.005
  )
  cfg.observations["actor"].terms["cube_to_goal_left"].noise = Unoise(
    n_min=-0.005, n_max=0.005
  )
  cfg.observations["actor"].terms["ee_velocity_right"] = ObservationTermCfg(
    func=manipulation_mdp.ee_velocity,
    params={"asset_cfg": _grasp_site_right_cfg()},
    noise=Unoise(n_min=-0.02, n_max=0.02),
  )
  cfg.observations["actor"].terms["ee_velocity_left"] = ObservationTermCfg(
    func=manipulation_mdp.ee_velocity,
    params={"asset_cfg": _grasp_site_left_cfg()},
    noise=Unoise(n_min=-0.02, n_max=0.02),
  )
  cfg.observations["critic"].terms["ee_velocity_right"] = ObservationTermCfg(
    func=manipulation_mdp.ee_velocity,
    params={"asset_cfg": _grasp_site_right_cfg()},
  )
  cfg.observations["critic"].terms["ee_velocity_left"] = ObservationTermCfg(
    func=manipulation_mdp.ee_velocity,
    params={"asset_cfg": _grasp_site_left_cfg()},
  )

  cfg.events["reset_robot_joints"].params["asset_cfg"] = _controlled_joint_cfg()
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.02, 0.02)
  cfg.events["reset_robot_joints"].params["velocity_range"] = (0.0, 0.0)
  cfg.events["encoder_bias"] = EventTermCfg(
    func=dr.encoder_bias,
    mode="reset",
    params={
      "bias_range": (-0.002, 0.002),
      "asset_cfg": _controlled_joint_cfg(),
    },
  )
  cfg.events.pop("fingertip_friction_slide", None)
  cfg.events.pop("fingertip_friction_spin", None)
  cfg.events.pop("fingertip_friction_roll", None)

  cfg.rewards["lift_right"].params["object_name"] = "cylinder_right"
  cfg.rewards["lift_right"].params["asset_cfg"] = _grasp_site_right_cfg()
  cfg.rewards["lift_right"].params["reaching_std"] = 0.14
  cfg.rewards["lift_right"].params["bringing_std"] = 0.24
  cfg.rewards["lift_right"].weight = 1.5
  cfg.rewards["lift_precise_right"].params["object_name"] = "cylinder_right"
  cfg.rewards["lift_precise_right"].params["std"] = 0.12
  cfg.rewards["lift_precise_right"].weight = 0.75
  cfg.rewards["lift_clearance_right"] = RewardTermCfg(
    func=manipulation_mdp.lift_height_progress_reward,
    weight=2.5,
    params={
      "command_name": "lift_height_right",
      "object_name": "cylinder_right",
      "start_height": _CYLINDER_CENTER_Z,
    },
  )

  cfg.rewards["lift_left"].params["object_name"] = "cylinder_left"
  cfg.rewards["lift_left"].params["asset_cfg"] = _grasp_site_left_cfg()
  cfg.rewards["lift_left"].params["reaching_std"] = 0.14
  cfg.rewards["lift_left"].params["bringing_std"] = 0.24
  cfg.rewards["lift_left"].weight = 1.5
  cfg.rewards["lift_precise_left"].params["object_name"] = "cylinder_left"
  cfg.rewards["lift_precise_left"].params["std"] = 0.12
  cfg.rewards["lift_precise_left"].weight = 0.75
  cfg.rewards["lift_clearance_left"] = RewardTermCfg(
    func=manipulation_mdp.lift_height_progress_reward,
    weight=2.5,
    params={
      "command_name": "lift_height_left",
      "object_name": "cylinder_left",
      "start_height": _CYLINDER_CENTER_Z,
    },
  )

  cfg.rewards["right_grasp_contact"] = RewardTermCfg(
    func=manipulation_mdp.contact_force_reward,
    weight=1.5,
    params={
      "sensor_name": "right_grasp_contact",
      "force_scale": 8.0,
      "min_force": 0.3,
    },
  )
  cfg.rewards["left_grasp_contact"] = RewardTermCfg(
    func=manipulation_mdp.contact_force_reward,
    weight=1.5,
    params={
      "sensor_name": "left_grasp_contact",
      "force_scale": 8.0,
      "min_force": 0.3,
    },
  )

  cfg.rewards["joint_pos_limits"].params["asset_cfg"] = _controlled_joint_cfg()
  cfg.rewards["joint_vel_hinge"].func = manipulation_mdp.joint_velocity_hinge_penalty_clipped
  cfg.rewards["joint_vel_hinge"].params["asset_cfg"] = _controlled_joint_cfg()
  cfg.rewards["joint_vel_hinge"].params["clip_excess"] = 2.0
  cfg.rewards["joint_vel_hinge"].params["max_vel"] = 0.35
  cfg.rewards["joint_vel_hinge"].weight = -0.01
  cfg.rewards["torso_posture"] = RewardTermCfg(
    func=manipulation_mdp.posture,
    weight=0.25,
    params={
      "asset_cfg": _torso_joint_cfg(),
      "std": {"Waist_Z": 0.10, "Waist_Y": 0.10},
    },
  )
  cfg.rewards["action_rate_l2"].weight = -0.002

  cfg.terminations["right_ee_table_collision"] = TerminationTermCfg(
    func=manipulation_mdp.illegal_contact,
    params={"sensor_name": "right_ee_table_collision", "force_threshold": 15.0},
  )
  cfg.terminations["left_ee_table_collision"] = TerminationTermCfg(
    func=manipulation_mdp.illegal_contact,
    params={"sensor_name": "left_ee_table_collision", "force_threshold": 15.0},
  )
  cfg.terminations.pop("right_ee_ground_collision", None)
  cfg.terminations.pop("left_ee_ground_collision", None)

  cfg.terminations["object_drop_right"] = TerminationTermCfg(
    func=manipulation_mdp.object_below_height,
    params={
      "object_name": "cylinder_right",
      "min_height": _TABLE_TOP_SURFACE_Z - 0.10,
    },
  )
  cfg.terminations["object_drop_left"] = TerminationTermCfg(
    func=manipulation_mdp.object_below_height,
    params={
      "object_name": "cylinder_left",
      "min_height": _TABLE_TOP_SURFACE_Z - 0.10,
    },
  )
  cfg.terminations["invalid_object_state_right"] = TerminationTermCfg(
    func=manipulation_mdp.invalid_object_state,
    params={"object_name": "cylinder_right"},
  )
  cfg.terminations["invalid_object_state_left"] = TerminationTermCfg(
    func=manipulation_mdp.invalid_object_state,
    params={"object_name": "cylinder_left"},
  )
  cfg.terminations["joint_velocity_exceeded"] = TerminationTermCfg(
    func=manipulation_mdp.joint_velocity_exceeded,
    params={
      "max_abs_vel": 8.0,
      "asset_cfg": _controlled_joint_cfg(),
    },
  )
  cfg.curriculum = {
    "lift_reward_schedule_right": CurriculumTermCfg(
      func=manipulation_mdp.reward_curriculum,
      params={
        "reward_name": "lift_right",
        "stages": [
          {"step": 0, "weight": 1.5, "params": {"reaching_std": 0.14, "bringing_std": 0.24}},
          {"step": 400 * 24, "weight": 2.5, "params": {"reaching_std": 0.12, "bringing_std": 0.18}},
          {"step": 1200 * 24, "weight": 3.0, "params": {"reaching_std": 0.10, "bringing_std": 0.14}},
        ],
      },
    ),
    "lift_reward_schedule_left": CurriculumTermCfg(
      func=manipulation_mdp.reward_curriculum,
      params={
        "reward_name": "lift_left",
        "stages": [
          {"step": 0, "weight": 1.5, "params": {"reaching_std": 0.14, "bringing_std": 0.24}},
          {"step": 400 * 24, "weight": 2.5, "params": {"reaching_std": 0.12, "bringing_std": 0.18}},
          {"step": 1200 * 24, "weight": 3.0, "params": {"reaching_std": 0.10, "bringing_std": 0.14}},
        ],
      },
    ),
    "lift_precise_schedule_right": CurriculumTermCfg(
      func=manipulation_mdp.reward_curriculum,
      params={
        "reward_name": "lift_precise_right",
        "stages": [
          {"step": 0, "weight": 0.75, "params": {"std": 0.12}},
          {"step": 400 * 24, "weight": 3.0, "params": {"std": 0.08}},
          {"step": 1200 * 24, "weight": 5.0, "params": {"std": 0.06}},
        ],
      },
    ),
    "lift_precise_schedule_left": CurriculumTermCfg(
      func=manipulation_mdp.reward_curriculum,
      params={
        "reward_name": "lift_precise_left",
        "stages": [
          {"step": 0, "weight": 0.75, "params": {"std": 0.12}},
          {"step": 400 * 24, "weight": 3.0, "params": {"std": 0.08}},
          {"step": 1200 * 24, "weight": 5.0, "params": {"std": 0.06}},
        ],
      },
    ),
    "lift_clearance_schedule_right": CurriculumTermCfg(
      func=manipulation_mdp.reward_curriculum,
      params={
        "reward_name": "lift_clearance_right",
        "stages": [
          {"step": 0, "weight": 2.5},
          {"step": 400 * 24, "weight": 3.5},
          {"step": 1200 * 24, "weight": 4.0},
        ],
      },
    ),
    "lift_clearance_schedule_left": CurriculumTermCfg(
      func=manipulation_mdp.reward_curriculum,
      params={
        "reward_name": "lift_clearance_left",
        "stages": [
          {"step": 0, "weight": 2.5},
          {"step": 400 * 24, "weight": 3.5},
          {"step": 1200 * 24, "weight": 4.0},
        ],
      },
    ),
    "right_grasp_contact_schedule": CurriculumTermCfg(
      func=manipulation_mdp.reward_curriculum,
      params={
        "reward_name": "right_grasp_contact",
        "stages": [
          {"step": 0, "weight": 1.5},
          {"step": 400 * 24, "weight": 0.8},
          {"step": 1200 * 24, "weight": 0.35},
        ],
      },
    ),
    "left_grasp_contact_schedule": CurriculumTermCfg(
      func=manipulation_mdp.reward_curriculum,
      params={
        "reward_name": "left_grasp_contact",
        "stages": [
          {"step": 0, "weight": 1.5},
          {"step": 400 * 24, "weight": 0.8},
          {"step": 1200 * 24, "weight": 0.35},
        ],
      },
    ),
    "joint_vel_hinge_schedule": CurriculumTermCfg(
      func=manipulation_mdp.reward_curriculum,
      params={
        "reward_name": "joint_vel_hinge",
        "stages": [
          {"step": 0, "weight": -0.01, "params": {"max_vel": 0.40, "clip_excess": 2.0}},
          {"step": 1200 * 24, "weight": -0.03, "params": {"max_vel": 0.35, "clip_excess": 2.0}},
        ],
      },
    ),
    "torso_posture_schedule": CurriculumTermCfg(
      func=manipulation_mdp.reward_curriculum,
      params={
        "reward_name": "torso_posture",
        "stages": [
          {"step": 0, "weight": 0.25},
          {"step": 400 * 24, "weight": 0.45},
          {"step": 1200 * 24, "weight": 0.70},
        ],
      },
    ),
    "lift_command_schedule_right": CurriculumTermCfg(
      func=manipulation_mdp.lifting_command_curriculum,
      params={
        "command_name": "lift_height_right",
        "stages": [
          {
            "step": 0,
            "success_threshold": 0.10,
            "target_position_range": {
              "x": _TARGET_X_RANGE_EASY,
              "y": _TARGET_Y_RIGHT_RANGE_EASY,
              "z": _TARGET_Z_RANGE_EASY,
            },
            "object_pose_range": {
              "x": _OBJECT_X_RANGE_EASY,
              "y": _OBJECT_Y_RIGHT_RANGE_EASY,
              "yaw": (-0.15, 0.15),
            },
          },
          {
            "step": 800 * 24,
            "success_threshold": 0.05,
            "target_position_range": {
              "x": _TARGET_X_RANGE_MID,
              "y": _TARGET_Y_RIGHT_RANGE_MID,
              "z": _TARGET_Z_RANGE_MID,
            },
            "object_pose_range": {
              "x": _OBJECT_X_RANGE_MID,
              "y": _OBJECT_Y_RIGHT_RANGE_MID,
              "yaw": (-0.25, 0.25),
            },
          },
          {
            "step": 2200 * 24,
            "success_threshold": 0.04,
            "target_position_range": {
              "x": _TARGET_X_RANGE,
              "y": _TARGET_Y_RIGHT_RANGE,
              "z": _TARGET_Z_RANGE,
            },
            "object_pose_range": {
              "x": _OBJECT_X_RANGE,
              "y": _OBJECT_Y_RIGHT_RANGE,
              "yaw": (-0.4, 0.4),
            },
          },
        ],
      },
    ),
    "lift_command_schedule_left": CurriculumTermCfg(
      func=manipulation_mdp.lifting_command_curriculum,
      params={
        "command_name": "lift_height_left",
        "stages": [
          {
            "step": 0,
            "success_threshold": 0.10,
            "target_position_range": {
              "x": _TARGET_X_RANGE_EASY,
              "y": _TARGET_Y_LEFT_RANGE_EASY,
              "z": _TARGET_Z_RANGE_EASY,
            },
            "object_pose_range": {
              "x": _OBJECT_X_RANGE_EASY,
              "y": _OBJECT_Y_LEFT_RANGE_EASY,
              "yaw": (-0.15, 0.15),
            },
          },
          {
            "step": 800 * 24,
            "success_threshold": 0.05,
            "target_position_range": {
              "x": _TARGET_X_RANGE_MID,
              "y": _TARGET_Y_LEFT_RANGE_MID,
              "z": _TARGET_Z_RANGE_MID,
            },
            "object_pose_range": {
              "x": _OBJECT_X_RANGE_MID,
              "y": _OBJECT_Y_LEFT_RANGE_MID,
              "yaw": (-0.25, 0.25),
            },
          },
          {
            "step": 2200 * 24,
            "success_threshold": 0.04,
            "target_position_range": {
              "x": _TARGET_X_RANGE,
              "y": _TARGET_Y_LEFT_RANGE,
              "z": _TARGET_Z_RANGE,
            },
            "object_pose_range": {
              "x": _OBJECT_X_RANGE,
              "y": _OBJECT_Y_LEFT_RANGE,
              "yaw": (-0.4, 0.4),
            },
          },
        ],
      },
    ),
  }
  cfg.viewer.body_name = "CHEST"
  cfg.viewer.distance = 2.2
  cfg.viewer.azimuth = 160.0
  cfg.viewer.elevation = -18.0

  cfg.sim.nconmax = 180
  cfg.sim.njmax = 1800
  cfg.sim.mujoco.iterations = 20
  cfg.sim.mujoco.ls_iterations = 50
  cfg.sim.mujoco.ccd_iterations = 500
  cfg.episode_length_s = 20.0

  if play:
    cfg.episode_length_s = 20.0
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("encoder_bias", None)
    cfg.curriculum = {}
    apply_wa1_d11_grasp_cylinder_stage(cfg, "final", command_name="lift_height_right")
    apply_wa1_d11_grasp_cylinder_stage(cfg, "final", command_name="lift_height_left")
    cfg.commands["lift_height_right"].resampling_time_range = (20.0, 20.0)
    cfg.commands["lift_height_left"].resampling_time_range = (20.0, 20.0)

  return cfg
