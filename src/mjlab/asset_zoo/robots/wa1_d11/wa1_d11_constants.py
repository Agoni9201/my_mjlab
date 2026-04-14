"""WA1_D11 robot constants for fixed-base upper-body tracking."""

from __future__ import annotations

import os
from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets

WA1_D11_XML_ENV_VAR = "MJLAB_WA1_D11_XML"
_DEFAULT_WA1_D11_XML = Path("/home/robot706/yx/mjlab_v3/src/mjlab/asset_zoo/robots/wa1_d11/xml/WA1_D11.xml")

WA1_TRACK_BODY_NAMES = (
  "BODY",
  "CHEST",
  "SCAPULA_L",
  "FOREARM_L",
  "WRIST_FLANGE_L",
  "SCAPULA_R",
  "FOREARM_R",
  "WRIST_FLANGE_R",
)

WA1_TRACK_JOINT_NAMES = (
  "Lifting_Z",
  "Waist_Z",
  "Waist_Y",
  "Shoulder_Y_L",
  "Shoulder_X_L",
  "Shoulder_Z_L",
  "Elbow_L",
  "Wrist_Z_L",
  "Wrist_Y_L",
  "Wrist_X_L",
  "Shoulder_Y_R",
  "Shoulder_X_R",
  "Shoulder_Z_R",
  "Elbow_R",
  "Wrist_Z_R",
  "Wrist_Y_R",
  "Wrist_X_R",
)
WA1_TORSO_JOINT_NAMES = ("Waist_Z", "Waist_Y")
WA1_LEFT_ARM_JOINT_NAMES = (
  "Shoulder_Y_L",
  "Shoulder_X_L",
  "Shoulder_Z_L",
  "Elbow_L",
  "Wrist_Z_L",
  "Wrist_Y_L",
  "Wrist_X_L",
)
WA1_RIGHT_ARM_JOINT_NAMES = (
  "Shoulder_Y_R",
  "Shoulder_X_R",
  "Shoulder_Z_R",
  "Elbow_R",
  "Wrist_Z_R",
  "Wrist_Y_R",
  "Wrist_X_R",
)

WA1_END_EFFECTOR_BODY_NAMES = ("WRIST_FLANGE_L", "WRIST_FLANGE_R")
WA1_LIFT_LOCK_POS = 0
WA1_RIGHT_GRASP_SITE_NAME = "grasp_right_site"
WA1_RIGHT_GRASP_SITE_POS = (0.0, 0.03, -0.15)
WA1_RIGHT_GRASP_SITE_SIZE = (0.012, 0.012, 0.012)
WA1_RIGHT_HAND_CYLINDER_GRASP_JOINT_POS = {
  "index_MCP_R": 0.53,
  "index_PIP_R": 0.81,
  "index_DIP_R": 0.79,
  "middle_MCP_R": 0.73,
  "middle_PIP_R": 1.06,
  "middle_DIP_R": 0.50,
  "ring_MCP_R": 0.66,
  "ring_PIP_R": 0.92,
  "ring_DIP_R": 0.70,
  "little_MCP_R": 0.58,
  "little_PIP_R": 1.08,
  "little_DIP_R": 0.58,
  "thumb_CMC_R": 0.83,
  "thumb_MP_R": -0.14,
  "thumb_IP_R": 0.62,
}
WA1_LEFT_GRASP_SITE_NAME = "grasp_left_site"
WA1_LEFT_GRASP_SITE_POS = (0.0, -0.03, -0.15)
WA1_LEFT_GRASP_SITE_SIZE = (0.012, 0.012, 0.012)
WA1_LEFT_HAND_CYLINDER_GRASP_JOINT_POS = {
  joint_name[:-1] + "L": target
  for joint_name, target in WA1_RIGHT_HAND_CYLINDER_GRASP_JOINT_POS.items()
}
WA1_FIXED_HAND_JOINT_POS = {
  **WA1_RIGHT_HAND_CYLINDER_GRASP_JOINT_POS,
  **WA1_LEFT_HAND_CYLINDER_GRASP_JOINT_POS,
}
_WA1_HAND_CYLINDER_QUAT = (0.70710678, 0.0, 0.70710678, 0.0)
_WA1_HAND_CYLINDER_SIZE = (0.03, 0.06, 0.0)
_WA1_HAND_CYLINDER_RGBA = (0.95, 0.55, 0.15, 0.7)
_WA1_ATTACHED_HAND_CYLINDERS = (
  ("WRIST_FLANGE_R", "wa1_right_hand_cylinder", (0.0, 0.03, -0.165)),
  ("WRIST_FLANGE_L", "wa1_left_hand_cylinder", (0.0, -0.03, -0.165)),
)
_WA1_FRONT_TABLE_BODY_NAME = "wa1_front_table"
_WA1_FRONT_TABLE_TOP_NAME = "wa1_front_table_top"
_WA1_FRONT_TABLE_POS = (0.9, 0.0, 0.55)   # 桌面整体高度（下调）
_WA1_FRONT_TABLE_TOP_SIZE = (0.40, 0.60, 0.035)
_WA1_FRONT_TABLE_TOP_RGBA = (0.62, 0.46, 0.30, 1.0)
_WA1_FRONT_TABLE_LEG_SIZE = (0.025, 0.025, 0.7)
_WA1_FRONT_TABLE_LEG_RGBA = (0.32, 0.22, 0.15, 1.0)
_WA1_FRONT_TABLE_LEG_OFFSETS = (
  (0.30, 0.50, -0.73),
  (0.30, -0.50, -0.73),
  (-0.30, 0.50, -0.73),
  (-0.30, -0.50, -0.73),
)
_WA1_FLOOR_POS = (0.0, 0.0, -0.25)
_WA1_FLOOR_SIZE = (0.0, 0.0, 1.0)
_WA1_FLOOR_TEXTURE_NAME = "grid"
_WA1_FLOOR_MATERIAL_NAME = "grid"
WA1_HAND_CYLINDER_QUAT = _WA1_HAND_CYLINDER_QUAT
WA1_HAND_CYLINDER_SIZE = _WA1_HAND_CYLINDER_SIZE
WA1_FRONT_TABLE_POS = _WA1_FRONT_TABLE_POS
WA1_FRONT_TABLE_TOP_SIZE = _WA1_FRONT_TABLE_TOP_SIZE

_LEFT_SHOULDER_JOINTS = ("Shoulder_Y_L", "Shoulder_X_L", "Shoulder_Z_L")
_RIGHT_SHOULDER_JOINTS = ("Shoulder_Y_R", "Shoulder_X_R", "Shoulder_Z_R")
_ELBOW_JOINTS = ("Elbow_L", "Elbow_R")
_WRIST_Z_JOINTS = ("Wrist_Z_L", "Wrist_Z_R")
_WRIST_XY_JOINTS = ("Wrist_Y_L", "Wrist_X_L", "Wrist_Y_R", "Wrist_X_R")
_NECK_JOINTS = ("Neck_Z", "Neck_Y")
_SUPPORT_JOINTS = (
  "L_F_D",
  "L_F_Roll",
  "R_F_D",
  "R_F_Roll",
  "L_B_D",
  "L_B_Roll",
  "R_B_D",
  "R_B_Roll",
  "Lift_Wheel",
  "Right_Wheel",
)
_FINGER_JOINTS = (
  "index_MCP_L",
  "index_PIP_L",
  "index_DIP_L",
  "thumb_CMC_L",
  "thumb_MP_L",
  "thumb_IP_L",
  "middle_MCP_L",
  "middle_PIP_L",
  "middle_DIP_L",
  "ring_MCP_L",
  "ring_PIP_L",
  "ring_DIP_L",
  "little_MCP_L",
  "little_PIP_L",
  "little_DIP_L",
  "index_MCP_R",
  "index_PIP_R",
  "index_DIP_R",
  "thumb_CMC_R",
  "thumb_MP_R",
  "thumb_IP_R",
  "middle_MCP_R",
  "middle_PIP_R",
  "middle_DIP_R",
  "ring_MCP_R",
  "ring_PIP_R",
  "ring_DIP_R",
  "little_MCP_R",
  "little_PIP_R",
  "little_DIP_R",
)
# 新增碰撞分组
# ARM_BODIES = {
# "SCAPULA_L", "SHOULDER_L", "UPPERARM_L", "FOREARM_L", "WRIST_REVOLUTE_L", "WRIST_UPDOWN_L", "WRIST_FLANGE_L",
# "SCAPULA_R", "SHOULDER_R", "UPPERARM_R", "FOREARM_R", "WRIST_REVOLUTE_R", "WRIST_UPDOWN_R", "WRIST_FLANGE_R",
# }
# TORSO_BODIES = {"BODY", "TORSO", "CHEST"}
# HAND_BODIES = {
#     "index_Link1_L", "index_Link2_L", "index_Link3_L",
#     "thumb_Link1_L", "thumb_Link2_L", "thumb_Link3_L",
#     "middle_Link1_L", "middle_Link2_L", "middle_Link3_L",
#     "ring_Link1_L", "ring_Link2_L", "ring_Link3_L",
#     "little_Link1_L", "little_Link2_L", "little_Link3_L",
#     "index_Link1_R", "index_Link2_R", "index_Link3_R",
#     "thumb_Link1_R", "thumb_Link2_R", "thumb_Link3_R",
#     "middle_Link1_R", "middle_Link2_R", "middle_Link3_R",
#     "ring_Link1_R", "ring_Link2_R", "ring_Link3_R",
#     "little_Link1_R", "little_Link2_R", "little_Link3_R",
# }
# CT_ENV = 1 # floor/table
# CT_ARM = 2 # 手臂
# CT_TORSO = 4 # 身体躯干
# CT_OBJ = 8 # 操作圆柱


_LIFT_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=("Lifting_Z",),
  stiffness=20_000.0,
  damping=400.0,
  effort_limit=6780.0,
)
_WAIST_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=("Waist_Z", "Waist_Y"),
  stiffness=250.0,
  damping=20.0,
  effort_limit=290.0,
)
_SHOULDER_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(*_LEFT_SHOULDER_JOINTS, *_RIGHT_SHOULDER_JOINTS),
  stiffness=90.0,
  damping=10.0,
  effort_limit=57.0,
)
_ELBOW_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=_ELBOW_JOINTS,
  stiffness=60.0,
  damping=8.0,
  effort_limit=31.0,
)
_WRIST_Z_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=_WRIST_Z_JOINTS,
  stiffness=35.0,
  damping=5.0,
  effort_limit=25.0,
)
_WRIST_XY_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=_WRIST_XY_JOINTS,
  stiffness=35.0,
  damping=5.0,
  effort_limit=16.0,
)
_NECK_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=_NECK_JOINTS,
  stiffness=20.0,
  damping=2.0,
  effort_limit=7.0,
)
_FINGER_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=_FINGER_JOINTS,
  stiffness=8.0,
  damping=1.0,
  effort_limit=4.0,
)
_SUPPORT_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=_SUPPORT_JOINTS,
  stiffness=80.0,
  damping=8.0,
  effort_limit=50.0,
)

WA1_D11_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    _LIFT_ACTUATOR,
    _WAIST_ACTUATOR,
    _SHOULDER_ACTUATOR,
    _ELBOW_ACTUATOR,
    _WRIST_Z_ACTUATOR,
    _WRIST_XY_ACTUATOR,
    _NECK_ACTUATOR,
    _SUPPORT_ACTUATOR,
  ),
  soft_joint_pos_limit_factor=0.95,
)


def _resolve_wa1_d11_xml() -> Path:
  env_path = os.environ.get(WA1_D11_XML_ENV_VAR)
  candidates = [Path(env_path).expanduser() if env_path else None, _DEFAULT_WA1_D11_XML]
  for candidate in candidates:
    if candidate is not None and candidate.exists():
      return candidate.resolve()
  raise FileNotFoundError(
    f"Could not find WA1_D11 XML. Set {WA1_D11_XML_ENV_VAR} "
    "to the full path of WA1_D11.xml."
  )


def _joint_equality_data(target: float) -> list[float]:
  return [target, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def _set_locked_joint_targets(
  spec: mujoco.MjSpec, joint_targets: dict[str, float]
) -> None:
  existing_equalities = {
    equality.name1.split("/")[-1]: equality
    for equality in spec.equalities
    if equality.type == mujoco.mjtEq.mjEQ_JOINT
  }
  for joint_name, target in joint_targets.items():
    equality = existing_equalities.get(joint_name)
    if equality is None:
      equality = spec.add_equality(
        name=f"lock_{joint_name.lower()}",
        type=mujoco.mjtEq.mjEQ_JOINT,
        name1=joint_name,
        data=_joint_equality_data(target),
      )
      existing_equalities[joint_name] = equality
    equality.data[:] = _joint_equality_data(target)

# 单独GMR的时候手上的圆柱体
def _add_attached_hand_cylinders(spec: mujoco.MjSpec) -> None:
  existing_geom_names = {geom.name for geom in spec.geoms}
  for body_name, geom_name, local_pos in _WA1_ATTACHED_HAND_CYLINDERS:
    if geom_name in existing_geom_names:
      continue
    body = spec.body(body_name)
    geom = body.add_geom(name=geom_name, type=mujoco.mjtGeom.mjGEOM_CYLINDER)
    geom.pos[:] = local_pos
    geom.quat[:] = _WA1_HAND_CYLINDER_QUAT
    geom.size[:] = _WA1_HAND_CYLINDER_SIZE
    geom.rgba[:] = _WA1_HAND_CYLINDER_RGBA
    geom.group = 1
    geom.contype = 0
    geom.conaffinity = 0
    geom.density = 0.0
    existing_geom_names.add(geom_name)


def _add_right_grasp_site(spec: mujoco.MjSpec) -> None:
  existing_site_names = {site.name for site in spec.sites}
  if WA1_RIGHT_GRASP_SITE_NAME in existing_site_names:
    return

  body = spec.body("WRIST_FLANGE_R")
  body.add_site(
    name=WA1_RIGHT_GRASP_SITE_NAME,
    pos=WA1_RIGHT_GRASP_SITE_POS,
    size=WA1_RIGHT_GRASP_SITE_SIZE,
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    rgba=(0.2, 0.8, 0.2, 0.35),
    group=4,
  )


def _add_left_grasp_site(spec: mujoco.MjSpec) -> None:
  existing_site_names = {site.name for site in spec.sites}
  if WA1_LEFT_GRASP_SITE_NAME in existing_site_names:
    return

  body = spec.body("WRIST_FLANGE_L")
  body.add_site(
    name=WA1_LEFT_GRASP_SITE_NAME,
    pos=WA1_LEFT_GRASP_SITE_POS,
    size=WA1_LEFT_GRASP_SITE_SIZE,
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    rgba=(0.2, 0.8, 0.2, 0.35),
    group=4,
  )


def _add_floor(spec: mujoco.MjSpec) -> None:
  existing_texture_names = {tex.name for tex in spec.textures}
  if _WA1_FLOOR_TEXTURE_NAME not in existing_texture_names:
    spec.add_texture(
      name=_WA1_FLOOR_TEXTURE_NAME,
      type=mujoco.mjtTexture.mjTEXTURE_2D,
      builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
      mark=mujoco.mjtMark.mjMARK_CROSS,
      rgb1=(0.2, 0.3, 0.4),
      rgb2=(0.1, 0.15, 0.2),
      markrgb=(0.8, 0.8, 0.8),
      width=512,
      height=512,
    )

  existing_material_names = {mat.name for mat in spec.materials}
  if _WA1_FLOOR_MATERIAL_NAME not in existing_material_names:
    material = spec.add_material(
      name=_WA1_FLOOR_MATERIAL_NAME,
      texuniform=True,
      texrepeat=(5.0, 5.0),
      reflectance=0.2,
    )
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB.value] = (
      _WA1_FLOOR_TEXTURE_NAME
    )

  existing_geom_names = {geom.name for geom in spec.geoms}
  if "floor" in existing_geom_names:
    return

  floor = spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE)
  floor.pos[:] = _WA1_FLOOR_POS
  floor.size[:] = _WA1_FLOOR_SIZE
  floor.material = _WA1_FLOOR_MATERIAL_NAME

# GMR中可视化的桌子
def _add_front_table(spec: mujoco.MjSpec) -> None:
  existing_body_names = {body.name for body in spec.bodies}
  if _WA1_FRONT_TABLE_BODY_NAME in existing_body_names:
    return

  table_body = spec.worldbody.add_body(name=_WA1_FRONT_TABLE_BODY_NAME)
  table_body.pos[:] = _WA1_FRONT_TABLE_POS

  tabletop = table_body.add_geom(
    name=_WA1_FRONT_TABLE_TOP_NAME,
    type=mujoco.mjtGeom.mjGEOM_BOX,
  )
  tabletop.size[:] = _WA1_FRONT_TABLE_TOP_SIZE
  tabletop.rgba[:] = _WA1_FRONT_TABLE_TOP_RGBA
  tabletop.group = 1
  tabletop.contype = 0   # 不主动撞别人
  tabletop.conaffinity = 0  # 不接受任何碰撞
  tabletop.density = 0.0

  for leg_index, leg_offset in enumerate(_WA1_FRONT_TABLE_LEG_OFFSETS):
    leg = table_body.add_geom(
      name=f"wa1_front_table_leg_{leg_index}",
      type=mujoco.mjtGeom.mjGEOM_BOX,
    )
    leg.pos[:] = leg_offset
    leg.size[:] = _WA1_FRONT_TABLE_LEG_SIZE
    leg.rgba[:] = _WA1_FRONT_TABLE_LEG_RGBA
    leg.group = 1
    leg.contype = 0
    leg.conaffinity = 0
    leg.density = 0.0


def get_spec(include_attached_hand_cylinders: bool = True) -> mujoco.MjSpec:
  xml_path = _resolve_wa1_d11_xml()
  spec = mujoco.MjSpec.from_file(str(xml_path))

  assets: dict[str, bytes] = {}
  meshdir = spec.meshdir
  if meshdir:
    meshdir_path = Path(meshdir)
    if not meshdir_path.is_absolute():
      meshdir_path = xml_path.parent / meshdir_path
    asset_root = meshdir_path
  else:
    asset_root = xml_path.parent
  if asset_root.exists():
    update_assets(assets, asset_root, meshdir, recursive=True)
  spec.assets = assets
  for key in tuple(spec.keys):
    spec.delete(key)

  # Replace the XML torque motors with position actuators so tracking can train
  # directly in joint space.
  spec.actuators.clear() # 清空原有的actuators 执行器
  _set_locked_joint_targets(    # 锁定一些关节
    spec,
    # {"Lifting_Z": WA1_LIFT_LOCK_POS, **WA1_FIXED_HAND_JOINT_POS},
    {**WA1_FIXED_HAND_JOINT_POS},
  )
  if include_attached_hand_cylinders:  # 如果需要添加手部圆柱体
    _add_attached_hand_cylinders(spec) 
  _add_floor(spec)
  _add_front_table(spec) #在抬升任务中不用可视化的桌子了，先注释掉

  # Keep robot-floor contact while disabling expensive self-collision between mesh
  # bodies. The visual meshes are already in group 1 and remain non-collidable.
  for geom in spec.geoms:
    if geom.name == "floor" or geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
      geom.contype = 1
      geom.conaffinity = 2
      geom.condim = 3
    elif geom.group == 1:
      geom.contype = 0
      geom.conaffinity = 0
    else:
      geom.contype = 2
      geom.conaffinity = 1
      geom.condim = 3
    # parent_name = geom.parent.name if geom.parent is not None else ""
    # if geom.name == "floor" or geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
    #   geom.contype = CT_ENV
    #   geom.conaffinity = CT_ARM
    #   geom.condim = 3
    #   continue
    # if geom.name == _WA1_FRONT_TABLE_TOP_NAME:
    #   geom.contype = CT_ENV
    #   geom.conaffinity = CT_ARM | CT_OBJ
    #   geom.condim = 3
    #   continue
    # if parent_name in HAND_BODIES:
    #   geom.contype = CT_ARM
    #   geom.conaffinity = CT_ENV | CT_OBJ
    #   geom.condim = 4
    #   continue
    # if parent_name in ARM_BODIES:
    #   geom.contype = CT_ARM
    #   geom.conaffinity = CT_TORSO | CT_ENV | CT_OBJ  # 表示这个碰撞体会和躯干、环境、物体发生碰撞，但不会和手臂自己发生碰撞
    #   geom.condim = 4  # 4维接触，包含位置和切向摩擦
    #   continue
    # if parent_name in TORSO_BODIES:
    #   geom.contype = CT_TORSO
    #   geom.conaffinity = CT_ARM 
    #   geom.condim = 3
    #   continue

    # geom.contype = 0
    # geom.conaffinity = 0
    # geom.condim = 3

  return spec

# 在执行操作任务时，去掉手部圆柱体以免干扰，且添加右手抓取点
def get_wa1_d11_manipulation_spec() -> mujoco.MjSpec:
  spec = get_spec(include_attached_hand_cylinders=False)
  _add_right_grasp_site(spec)
  _add_left_grasp_site(spec)
  return spec

# 关节的幅度系数
def _make_action_scale(
  joint_names: tuple[str, ...], effort_limit: float, stiffness: float
) -> dict[str, float]:
  scale = 0.25 * effort_limit / stiffness
  return {joint_name: scale for joint_name in joint_names}


WA1_D11_ACTION_SCALE: dict[str, float] = {}
# Keep the lifting column fixed at the motion home height. We still route it
# through the joint position action term so the actuator target stays at the
# default offset, but a zero action scale prevents the policy from changing it.
WA1_D11_ACTION_SCALE["Lifting_Z"] = WA1_LIFT_LOCK_POS
WA1_D11_ACTION_SCALE.update(_make_action_scale(("Waist_Z", "Waist_Y"), 290.0, 250.0))
WA1_D11_ACTION_SCALE.update(
  _make_action_scale(
    (*_LEFT_SHOULDER_JOINTS, *_RIGHT_SHOULDER_JOINTS), 57.0, 90.0
  )
)
WA1_D11_ACTION_SCALE.update(_make_action_scale(_ELBOW_JOINTS, 31.0, 60.0))
WA1_D11_ACTION_SCALE.update(_make_action_scale(_WRIST_Z_JOINTS, 25.0, 35.0))
WA1_D11_ACTION_SCALE.update(_make_action_scale(_WRIST_XY_JOINTS, 25.0, 35.0))

WA1_D11_MANIPULATION_ACTION_SCALE = dict(WA1_D11_ACTION_SCALE)
WA1_D11_MANIPULATION_ACTION_SCALE["Waist_Z"] *= 0.35
WA1_D11_MANIPULATION_ACTION_SCALE["Waist_Y"] *= 0.35 
# for _joint_name in WA1_LEFT_ARM_JOINT_NAMES:
#   WA1_D11_MANIPULATION_ACTION_SCALE[_joint_name] = 0.0
# for _joint_name in WA1_RIGHT_ARM_JOINT_NAMES:  # 右臂不参与操作任务，直接固定
#   WA1_D11_MANIPULATION_ACTION_SCALE[_joint_name] = 0.0

WA1_D11_TRACKING_HOME_JOINT_POS = {
  "Lifting_Z": 0.0,
  "Waist_Z": 0,
  "Waist_Y": 0,
  "Shoulder_Y_L": -0.1468,
  "Shoulder_X_L": 0.7672,
  "Shoulder_Z_L": 0.2116,
  "Elbow_L": -1.1176,
  "Wrist_Z_L": 0.0491,
  "Wrist_Y_L": -0.2541,
  "Wrist_X_L": -0.0264,
  "Shoulder_Y_R": -0.24,
  "Shoulder_X_R": -0.76,
  "Shoulder_Z_R": -0.21,
  "Elbow_R": -1.118,
  "Wrist_Z_R": -0.004,
  "Wrist_Y_R": -0.15,
  "Wrist_X_R": 0.029,
  **WA1_FIXED_HAND_JOINT_POS,
}


def get_wa1_d11_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      # resolve_expr() uses the first matching pattern, so the specific joint
      # names must come before the catch-all pattern.
      joint_pos={**WA1_D11_TRACKING_HOME_JOINT_POS, ".*": 0.0},
      joint_vel={".*": 0.0},
    ),
    spec_fn=get_spec,
    articulation=WA1_D11_ARTICULATION,
  )


def get_wa1_d11_manipulation_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      joint_pos={**WA1_D11_TRACKING_HOME_JOINT_POS, ".*": 0.0},
      joint_vel={".*": 0.0},
    ),
    spec_fn=get_wa1_d11_manipulation_spec,
    articulation=WA1_D11_ARTICULATION,
  )