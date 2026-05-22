from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_perceptive_rough_v1_env_cfg,
  unitree_g1_perceptive_rough_v2_env_cfg,
  unitree_g1_perceptive_rough_v2_step14_up_env_cfg,
  unitree_g1_perceptive_rough_v2_straight_env_cfg,
  unitree_g1_perceptive_rough_v2_stepup_finetune_env_cfg,
  unitree_g1_rough_env_cfg,
  unitree_g1_rough_step14_env_cfg,
  unitree_g1_rough_step14_down_env_cfg,
)
from .rl_cfg import (
  unitree_g1_perceptive_rough_v1_ppo_runner_cfg,
  unitree_g1_perceptive_rough_v2_ppo_runner_cfg,
  unitree_g1_perceptive_rough_v2_stepup_finetune_ppo_runner_cfg,
  unitree_g1_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Unitree-G1-Rough",
  env_cfg=unitree_g1_rough_env_cfg(),
  play_env_cfg=unitree_g1_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Perceptive-Rough-v1",
  env_cfg=unitree_g1_perceptive_rough_v1_env_cfg(),
  play_env_cfg=unitree_g1_perceptive_rough_v1_env_cfg(play=True),
  rl_cfg=unitree_g1_perceptive_rough_v1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Perceptive-Rough-v2",
  env_cfg=unitree_g1_perceptive_rough_v2_env_cfg(),
  play_env_cfg=unitree_g1_perceptive_rough_v2_env_cfg(play=True),
  rl_cfg=unitree_g1_perceptive_rough_v2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Perceptive-Rough-v2-Step14-Up",
  env_cfg=unitree_g1_perceptive_rough_v2_step14_up_env_cfg(),
  play_env_cfg=unitree_g1_perceptive_rough_v2_step14_up_env_cfg(play=True),
  rl_cfg=unitree_g1_perceptive_rough_v2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Perceptive-Rough-v2-Straight",
  env_cfg=unitree_g1_perceptive_rough_v2_straight_env_cfg(),
  play_env_cfg=unitree_g1_perceptive_rough_v2_straight_env_cfg(play=True),
  rl_cfg=unitree_g1_perceptive_rough_v2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Perceptive-Rough-v2-StepUp-Finetune",
  env_cfg=unitree_g1_perceptive_rough_v2_stepup_finetune_env_cfg(),
  play_env_cfg=unitree_g1_perceptive_rough_v2_stepup_finetune_env_cfg(play=True),
  rl_cfg=unitree_g1_perceptive_rough_v2_stepup_finetune_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Flat",
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Rough-Step14",
  env_cfg=unitree_g1_rough_step14_env_cfg(),
  play_env_cfg=unitree_g1_rough_step14_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Rough-Step14-Down",
  env_cfg=unitree_g1_rough_step14_down_env_cfg(),
  play_env_cfg=unitree_g1_rough_step14_down_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
