"""Headless checkpoint evaluation for velocity tasks."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.torch import configure_torch_backends


def evaluate_checkpoint(
  task_id: str,
  checkpoint: Path,
  *,
  num_envs: int,
  steps: int,
  command_x: float,
  preserve_command_ranges: bool,
  device: str,
  seed: int,
) -> dict[str, float | str]:
  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  env_cfg.seed = seed
  env_cfg.scene.num_envs = num_envs
  twist_cmd = env_cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  if not preserve_command_ranges:
    twist_cmd.ranges.lin_vel_x = (command_x, command_x)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (0.0, 0.0)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
  runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  obs, _ = wrapped.reset()
  robot = wrapped.unwrapped.scene["robot"]
  feet_cfg = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))
  feet_cfg.resolve(wrapped.unwrapped.scene)
  step_dt = wrapped.unwrapped.step_dt
  start_x = robot.data.root_link_pos_w[:, 0].detach().clone()
  start_heading = robot.data.heading_w.detach().clone()

  had_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
  fall_events = 0
  cmd_x_sum = 0.0
  cmd_y_sum = 0.0
  cmd_yaw_sum = 0.0
  cmd_abs_y_sum = 0.0
  cmd_abs_yaw_sum = 0.0
  speed_x_sum = 0.0
  speed_y_sum = 0.0
  abs_speed_y_sum = 0.0
  yaw_rate_sum = 0.0
  abs_yaw_rate_sum = 0.0
  base_z_sum = 0.0
  track_sum = 0.0
  rel_foot_height_sum = 0.0
  left_rel_foot_height_sum = 0.0
  right_rel_foot_height_sum = 0.0
  max_rel_foot_height = torch.zeros(num_envs, device=device)
  left_max_rel_foot_height = torch.zeros(num_envs, device=device)
  right_max_rel_foot_height = torch.zeros(num_envs, device=device)

  with torch.no_grad():
    for _ in range(steps):
      actions = policy(obs)
      obs, _, dones, _ = wrapped.step(actions)
      done_bool = dones.bool()
      had_done |= done_bool
      fall_events += int(done_bool.sum().item())

      command = wrapped.unwrapped.command_manager.get_command("twist")
      cmd_x_sum += float(command[:, 0].mean().detach().cpu())
      cmd_y_sum += float(command[:, 1].mean().detach().cpu())
      cmd_yaw_sum += float(command[:, 2].mean().detach().cpu())
      cmd_abs_y_sum += float(command[:, 1].abs().mean().detach().cpu())
      cmd_abs_yaw_sum += float(command[:, 2].abs().mean().detach().cpu())

      base_lin_vel_b = robot.data.root_link_lin_vel_b
      base_ang_vel_b = robot.data.root_link_ang_vel_b
      speed_x_sum += float(base_lin_vel_b[:, 0].mean().detach().cpu())
      speed_y_sum += float(base_lin_vel_b[:, 1].mean().detach().cpu())
      abs_speed_y_sum += float(base_lin_vel_b[:, 1].abs().mean().detach().cpu())
      yaw_rate_sum += float(base_ang_vel_b[:, 2].mean().detach().cpu())
      abs_yaw_rate_sum += float(base_ang_vel_b[:, 2].abs().mean().detach().cpu())
      base_z_sum += float(robot.data.root_link_pos_w[:, 2].mean().detach().cpu())

      xy_error = (command[:, 0] - base_lin_vel_b[:, 0]).square() + (
        command[:, 1] - base_lin_vel_b[:, 1]
      ).square()
      z_error = base_lin_vel_b[:, 2].square()
      track = torch.exp(-(xy_error + 2.0 * z_error) / 0.25)
      track_sum += float(track.mean().detach().cpu())

      foot_z = robot.data.site_pos_w[:, feet_cfg.site_ids, 2]
      rel_foot_height = foot_z - foot_z.min(dim=1, keepdim=True)[0]
      step_rel_max = rel_foot_height.max(dim=1)[0]
      rel_foot_height_sum += float(step_rel_max.mean().detach().cpu())
      left_rel_foot_height_sum += float(rel_foot_height[:, 0].mean().detach().cpu())
      right_rel_foot_height_sum += float(rel_foot_height[:, 1].mean().detach().cpu())
      max_rel_foot_height = torch.maximum(max_rel_foot_height, step_rel_max)
      left_max_rel_foot_height = torch.maximum(
        left_max_rel_foot_height, rel_foot_height[:, 0]
      )
      right_max_rel_foot_height = torch.maximum(
        right_max_rel_foot_height, rel_foot_height[:, 1]
      )

  end_x = robot.data.root_link_pos_w[:, 0].detach()
  never_fell = ~had_done
  if never_fell.any():
    clean_distance = float((end_x[never_fell] - start_x[never_fell]).mean().cpu())
    heading_delta = robot.data.heading_w.detach()[never_fell] - start_heading[never_fell]
    heading_delta = torch.atan2(torch.sin(heading_delta), torch.cos(heading_delta))
    abs_heading_delta = float(heading_delta.abs().mean().cpu())
  else:
    clean_distance = 0.0
    abs_heading_delta = 0.0

  wrapped.close()

  return {
    "checkpoint": checkpoint.name,
    "fall_env_pct": 100.0 * float(had_done.float().mean().cpu()),
    "fall_events_per_env": fall_events / num_envs,
    "clean_distance_m": clean_distance,
    "integrated_forward_m": (speed_x_sum / steps) * step_dt * steps,
    "mean_command_x": cmd_x_sum / steps,
    "mean_command_y": cmd_y_sum / steps,
    "mean_abs_command_y": cmd_abs_y_sum / steps,
    "mean_command_yaw": cmd_yaw_sum / steps,
    "mean_abs_command_yaw": cmd_abs_yaw_sum / steps,
    "mean_speed_x": speed_x_sum / steps,
    "mean_speed_y": speed_y_sum / steps,
    "mean_abs_speed_y": abs_speed_y_sum / steps,
    "mean_yaw_rate": yaw_rate_sum / steps,
    "mean_abs_yaw_rate": abs_yaw_rate_sum / steps,
    "abs_heading_delta_rad": abs_heading_delta,
    "mean_base_z": base_z_sum / steps,
    "track_linear_reward": track_sum / steps,
    "mean_rel_foot_height": rel_foot_height_sum / steps,
    "left_mean_rel_foot_height": left_rel_foot_height_sum / steps,
    "right_mean_rel_foot_height": right_rel_foot_height_sum / steps,
    "left_right_mean_rel_foot_height_diff": (
      left_rel_foot_height_sum - right_rel_foot_height_sum
    )
    / steps,
    "max_rel_foot_height": float(max_rel_foot_height.mean().detach().cpu()),
    "left_max_rel_foot_height": float(
      left_max_rel_foot_height.mean().detach().cpu()
    ),
    "right_max_rel_foot_height": float(
      right_max_rel_foot_height.mean().detach().cpu()
    ),
    "left_right_max_rel_foot_height_diff": float(
      (left_max_rel_foot_height - right_max_rel_foot_height).mean().detach().cpu()
    ),
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("task_id")
  parser.add_argument("checkpoints", nargs="+", type=Path)
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument("--steps", type=int, default=600)
  parser.add_argument("--command-x", type=float, default=0.4)
  parser.add_argument(
    "--preserve-command-ranges",
    action="store_true",
    help="Use the task play command ranges instead of forcing x/y/yaw to 0.4/0/0.",
  )
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
  args = parser.parse_args()

  configure_torch_backends()
  torch.manual_seed(args.seed)

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  rows = [
    evaluate_checkpoint(
      args.task_id,
      checkpoint,
      num_envs=args.num_envs,
      steps=args.steps,
      command_x=args.command_x,
      preserve_command_ranges=args.preserve_command_ranges,
      device=args.device,
      seed=args.seed,
    )
    for checkpoint in args.checkpoints
  ]

  columns = [
    "checkpoint",
    "fall_env_pct",
    "fall_events_per_env",
    "clean_distance_m",
    "integrated_forward_m",
    "mean_command_x",
    "mean_command_y",
    "mean_abs_command_y",
    "mean_command_yaw",
    "mean_abs_command_yaw",
    "mean_speed_x",
    "mean_speed_y",
    "mean_abs_speed_y",
    "mean_yaw_rate",
    "mean_abs_yaw_rate",
    "abs_heading_delta_rad",
    "track_linear_reward",
    "mean_rel_foot_height",
    "left_mean_rel_foot_height",
    "right_mean_rel_foot_height",
    "left_right_mean_rel_foot_height_diff",
    "max_rel_foot_height",
    "left_max_rel_foot_height",
    "right_max_rel_foot_height",
    "left_right_max_rel_foot_height_diff",
    "mean_base_z",
  ]
  print(",".join(columns))
  for row in rows:
    values = []
    for col in columns:
      value = row[col]
      if isinstance(value, float):
        values.append(f"{value:.6f}")
      else:
        values.append(str(value))
    print(",".join(values))


if __name__ == "__main__":
  main()
