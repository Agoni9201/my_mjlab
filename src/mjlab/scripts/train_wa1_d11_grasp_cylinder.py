"""Convenience launcher for the WA1_D11 table-cylinder pure RL task."""

import sys

import tyro

from mjlab.scripts.train import TrainConfig, launch_training

TASK_ID = "Mjlab-Grasp-Cylinder-WA1-D11"


def main() -> None:
  import mjlab.tasks  # noqa: F401

  cfg = tyro.cli(
    TrainConfig,
    default=TrainConfig.from_task(TASK_ID),
    prog=sys.argv[0],
    config=mjlab.TYRO_FLAGS,
  )
  launch_training(TASK_ID, cfg)


if __name__ == "__main__":
  main()
