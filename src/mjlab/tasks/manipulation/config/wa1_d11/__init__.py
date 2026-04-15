from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import wa1_d11_grasp_cylinder_env_cfg
from .rl_cfg import wa1_d11_grasp_cylinder_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Grasp-Cylinder-WA1-D11",
  env_cfg=wa1_d11_grasp_cylinder_env_cfg(),
  play_env_cfg=wa1_d11_grasp_cylinder_env_cfg(play=True),
  rl_cfg=wa1_d11_grasp_cylinder_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)
