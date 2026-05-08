from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def wa1_d11_grasp_cylinder_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.003,
      num_learning_epochs=8,
      num_mini_batches=4,
      learning_rate=3.0e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.008,
      max_grad_norm=1.0,
    ),
    experiment_name="wa1_d11_grasp_cylinder",
    logger="tensorboard",
    upload_model=False,
    teacher_task_id="Mjlab-Tracking-FixedBase-WA1-D11",
    teacher_guidance_weight=0.35,
    teacher_guidance_std=0.40,
    teacher_dist_far=0.20,
    teacher_dist_near=0.08,
    teacher_release_height=0.04,
    teacher_post_grasp_scale=0.03,
    teacher_contact_force_threshold=0.35,
    teacher_anneal_start_iter=0,
    teacher_anneal_end_iter=2000,
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=5_000,
  )
