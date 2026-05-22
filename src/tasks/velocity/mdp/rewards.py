from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + (2 * z_error)
  return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + (0.05 * xy_error)
  return torch.exp(-ang_vel_error / std**2)


def body_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward flat base orientation (robot being upright).

  If asset_cfg has body_ids specified, computes the projected gravity
  for that specific body. Otherwise, uses the root link projected gravity.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # If body_ids are specified, compute projected gravity for that body.
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
    body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    gravity_w = asset.data.gravity_vec_w  # [3]
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  else:
    # Use root link projected gravity.
    xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return xy_squared


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float = 0.4,
  command_name: str | None = None,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  air_time = sensor_data.current_air_time
  contact_time = sensor_data.current_contact_time
  in_contact = contact_time > 0.0
  in_mode_time = torch.where(in_contact, contact_time, air_time)
  single_stance = torch.mean(in_contact.float(), dim=1) == 0.5
  mode_time = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
  error = torch.abs(mode_time - threshold)
  reward = torch.clamp(threshold - error, min=0.0)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  relative_to_lowest_foot: bool = False,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize insufficient swing-foot clearance, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  if relative_to_lowest_foot:
    target_z = torch.min(foot_z, dim=1, keepdim=True)[0] + target_height
  else:
    target_z = torch.full_like(foot_z, target_height)
  clearance_deficit = torch.clamp(target_z - foot_z, min=0.0)  # [B, N]
  cost = torch.sum(clearance_deficit * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


def feet_gait(
        env: ManagerBasedRlEnv,
        period: float,
        offset: list[float],
        threshold: float,
        command_threshold: float,
        command_name: str,
        sensor_name: str,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    is_contact = sensor.data.current_contact_time > 0
    global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
    offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(1, -1)
    leg_phase = (global_phase + offsets) % 1.0
    is_stance = (leg_phase < threshold)
    reward = (is_stance == is_contact).float().mean(dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command > command_threshold).float()
            reward *= scale
    return reward


class feet_swing_height:
  """Penalize insufficient peak swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.sensor_name = cfg.params["sensor_name"]
    self.site_names = cfg.params["asset_cfg"].site_names
    self.peak_heights = torch.zeros(
      (env.num_envs, len(self.site_names)), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
    relative_to_lowest_foot: bool = False,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    if relative_to_lowest_foot:
      target_z = torch.min(foot_heights, dim=1, keepdim=True)[0] + target_height
    else:
      target_z = torch.full_like(self.peak_heights, target_height)
    height_deficit = torch.clamp(target_z - self.peak_heights, min=0.0)
    cost = (
      torch.sum(
        torch.square(height_deficit / target_height) * first_contact.float(), dim=1
      )
      * active
    )
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def _foot_pos_b(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Return selected foot site positions in the root body frame."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
  rel_pos_w = foot_pos_w - asset.data.root_link_pos_w.unsqueeze(1)
  num_envs, num_feet, _ = rel_pos_w.shape
  root_quat_w = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, num_feet, -1)
  return quat_apply_inverse(
    root_quat_w.reshape(num_envs * num_feet, 4),
    rel_pos_w.reshape(num_envs * num_feet, 3),
  ).reshape(num_envs, num_feet, 3)


def _height_grid_from_sensor(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  *,
  size_x_m: float,
  size_y_m: float,
  resolution_m: float,
  x_offset_m: float,
  y_offset_m: float,
  miss_value: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  sensor = env.scene[sensor_name]
  heights = env.scene[sensor_name].data.pos_w[:, 2].unsqueeze(1) - sensor.data.hit_pos_w[
    ..., 2
  ]
  miss_mask = sensor.data.distances < 0
  heights = torch.where(miss_mask, torch.full_like(heights, miss_value), heights)
  num_x = max(1, int(round(size_x_m / resolution_m)) + 1)
  num_y = max(1, int(round(size_y_m / resolution_m)) + 1)
  heights = heights.reshape(env.num_envs, num_y, num_x)
  x = (
    torch.arange(num_x, device=heights.device, dtype=torch.float32) * resolution_m
    - size_x_m * 0.5
    + x_offset_m
  )
  y = (
    torch.arange(num_y, device=heights.device, dtype=torch.float32) * resolution_m
    - size_y_m * 0.5
    + y_offset_m
  )
  return heights, x, y


def _nearest_scan_indices(
  foot_pos_b: torch.Tensor,
  x: torch.Tensor,
  y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  x_idx = torch.round((foot_pos_b[:, :, 0] - x[0]) / (x[1] - x[0])).long()
  y_idx = torch.round((foot_pos_b[:, :, 1] - y[0]) / (y[1] - y[0])).long()
  x_idx = torch.clamp(x_idx, 0, x.numel() - 1)
  y_idx = torch.clamp(y_idx, 0, y.numel() - 1)
  return x_idx, y_idx


def _gather_height_grid(
  grid: torch.Tensor,
  x_idx: torch.Tensor,
  y_idx: torch.Tensor,
) -> torch.Tensor:
  env_ids = torch.arange(grid.shape[0], device=grid.device).unsqueeze(1)
  return grid[env_ids, y_idx, x_idx]


def feet_swing_forward(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  target_x: float,
  command_name: str,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize swing feet that lift without moving forward in the base frame."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  foot_pos_b = _foot_pos_b(env, asset_cfg)
  in_air = contact_sensor.data.found == 0
  forward_deficit = torch.clamp(target_x - foot_pos_b[:, :, 0], min=0.0)
  active = (command[:, 0] > command_threshold).float()
  cost = torch.sum(forward_deficit * in_air.float(), dim=1) * active
  num_swing_feet = torch.sum(in_air.float())
  mean_swing_x = torch.sum(foot_pos_b[:, :, 0] * in_air.float()) / torch.clamp(
    num_swing_feet, min=1
  )
  env.extras["log"]["Metrics/swing_foot_x_mean"] = mean_swing_x
  return cost


def feet_landing_forward(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  target_x: float,
  command_name: str,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize landings that touch down too close to or behind the base."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  foot_pos_b = _foot_pos_b(env, asset_cfg)
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)
  landing_deficit = torch.clamp(target_x - foot_pos_b[:, :, 0], min=0.0)
  active = (command[:, 0] > command_threshold).float()
  cost = (
    torch.sum(torch.square(landing_deficit / target_x) * first_contact.float(), dim=1)
    * active
  )
  num_landings = torch.sum(first_contact.float())
  mean_landing_x = torch.sum(foot_pos_b[:, :, 0] * first_contact.float()) / torch.clamp(
    num_landings, min=1
  )
  env.extras["log"]["Metrics/landing_foot_x_mean"] = mean_landing_x
  return cost


def feet_landing_edge_penalty(
  env: ManagerBasedRlEnv,
  contact_sensor_name: str,
  terrain_sensor_name: str,
  command_name: str,
  edge_threshold: float = 0.06,
  command_threshold: float = 0.1,
  size_x_m: float = 1.6,
  size_y_m: float = 1.0,
  resolution_m: float = 0.1,
  x_offset_m: float = 0.35,
  y_offset_m: float = 0.0,
  miss_value: float = 5.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize first contacts on cells with large local terrain height changes."""
  contact_sensor: ContactSensor = env.scene[contact_sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None

  grid, x, y = _height_grid_from_sensor(
    env,
    terrain_sensor_name,
    size_x_m=size_x_m,
    size_y_m=size_y_m,
    resolution_m=resolution_m,
    x_offset_m=x_offset_m,
    y_offset_m=y_offset_m,
    miss_value=miss_value,
  )
  foot_pos_b = _foot_pos_b(env, asset_cfg)
  x_idx, y_idx = _nearest_scan_indices(foot_pos_b, x, y)

  center = _gather_height_grid(grid, x_idx, y_idx)
  x_left = _gather_height_grid(grid, torch.clamp(x_idx - 1, min=0), y_idx)
  x_right = _gather_height_grid(grid, torch.clamp(x_idx + 1, max=x.numel() - 1), y_idx)
  y_left = _gather_height_grid(grid, x_idx, torch.clamp(y_idx - 1, min=0))
  y_right = _gather_height_grid(grid, x_idx, torch.clamp(y_idx + 1, max=y.numel() - 1))
  edge_score = torch.maximum(
    torch.maximum(torch.abs(center - x_left), torch.abs(center - x_right)),
    torch.maximum(torch.abs(center - y_left), torch.abs(center - y_right)),
  )

  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt).float()
  active = (command[:, 0] > command_threshold).float()
  cost = (
    torch.sum(torch.square(torch.clamp(edge_score - edge_threshold, min=0.0)) * first_contact, dim=1)
    * active
  )
  num_landings = torch.sum(first_contact)
  edge_mean = torch.sum(edge_score * first_contact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_edge_score_mean"] = edge_mean
  return cost


def early_lift_without_step_penalty(
  env: ManagerBasedRlEnv,
  contact_sensor_name: str,
  terrain_sensor_name: str,
  command_name: str,
  lift_height_threshold: float = 0.12,
  obstacle_height_threshold: float = 0.06,
  command_threshold: float = 0.1,
  x_window: tuple[float, float] = (0.25, 0.65),
  y_abs: float = 0.35,
  size_x_m: float = 1.6,
  size_y_m: float = 1.0,
  resolution_m: float = 0.1,
  x_offset_m: float = 0.35,
  y_offset_m: float = 0.0,
  miss_value: float = 5.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize high swing feet when no step edge is in the useful trigger window."""
  contact_sensor: ContactSensor = env.scene[contact_sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None

  grid, x, y = _height_grid_from_sensor(
    env,
    terrain_sensor_name,
    size_x_m=size_x_m,
    size_y_m=size_y_m,
    resolution_m=resolution_m,
    x_offset_m=x_offset_m,
    y_offset_m=y_offset_m,
    miss_value=miss_value,
  )
  x_mask = (x >= x_window[0]) & (x <= x_window[1])
  y_mask = torch.abs(y) <= y_abs
  window = grid[:, y_mask, :][:, :, x_mask]
  window_relief = torch.amax(window, dim=(1, 2)) - torch.amin(window, dim=(1, 2))
  no_step_in_window = window_relief < obstacle_height_threshold

  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  rel_foot_height = foot_z - foot_z.min(dim=1, keepdim=True)[0]
  in_air = (contact_sensor.data.found == 0).float()
  high_lift = torch.clamp(rel_foot_height - lift_height_threshold, min=0.0)
  active = ((command[:, 0] > command_threshold) & no_step_in_window).float()
  cost = torch.sum(torch.square(high_lift) * in_air, dim=1) * active
  env.extras["log"]["Metrics/early_lift_cost_mean"] = torch.mean(cost)
  env.extras["log"]["Metrics/step_window_relief_mean"] = torch.mean(window_relief)
  return cost


def forward_velocity_floor(
  env: ManagerBasedRlEnv,
  min_velocity: float,
  command_name: str,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize moving too slowly when a forward command is active."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  actual_x = asset.data.root_link_lin_vel_b[:, 0]
  active = (command[:, 0] > command_threshold).float()
  deficit = torch.clamp(min_velocity - actual_x, min=0.0)
  env.extras["log"]["Metrics/forward_velocity_deficit_mean"] = torch.mean(deficit)
  return torch.square(deficit / min_velocity) * active


def yaw_rate_l2_when_straight(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_x_threshold: float = 0.1,
  command_y_threshold: float = 0.05,
  command_yaw_threshold: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize yaw drift only when the command asks for straight forward walking."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  yaw_rate = asset.data.root_link_ang_vel_b[:, 2]
  active = (
    (command[:, 0] > command_x_threshold)
    & (torch.abs(command[:, 1]) < command_y_threshold)
    & (torch.abs(command[:, 2]) < command_yaw_threshold)
  ).float()
  env.extras["log"]["Metrics/straight_yaw_rate_abs_mean"] = torch.mean(
    torch.abs(yaw_rate) * active
  )
  return torch.square(yaw_rate) * active


class heading_drift_l2_when_straight:
  """Penalize accumulated heading drift during straight forward walking."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.heading_ref = torch.zeros(env.num_envs, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    command_x_threshold: float = 0.1,
    command_y_threshold: float = 0.05,
    command_yaw_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    heading = asset.data.heading_w
    reset_mask = env.episode_length_buf <= 1
    self.heading_ref = torch.where(reset_mask, heading, self.heading_ref)

    error = heading - self.heading_ref
    error = torch.atan2(torch.sin(error), torch.cos(error))
    active = (
      (command[:, 0] > command_x_threshold)
      & (torch.abs(command[:, 1]) < command_y_threshold)
      & (torch.abs(command[:, 2]) < command_yaw_threshold)
    ).float()
    env.extras["log"]["Metrics/straight_heading_error_abs_mean"] = torch.mean(
      torch.abs(error) * active
    )
    return torch.square(error) * active


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


def stand_still(
        env: ManagerBasedRlEnv,
        command_name: str,
        command_threshold: float = 0.1,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(diff_angle), dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command <= command_threshold).float()
            reward *= scale
    return reward
