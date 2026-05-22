from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def phase(env: ManagerBasedRlEnv, period: float, command_name: str) -> torch.Tensor:
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    stand_mask = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
    return phase


def mid360_height_scan(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  *,
  size_x_m: float = 1.6,
  size_y_m: float = 1.0,
  resolution_m: float = 0.1,
  x_offset_m: float = 0.0,
  y_offset_m: float = 0.0,
  visible_x_min_m: float = 0.25,
  visible_x_max_m: float = 1.20,
  visible_y_abs_m: float | None = None,
  keep_prob_min: float = 0.18,
  keep_prob_max: float = 0.35,
  frame_dropout_prob: float = 0.08,
  miss_value: float = 5.0,
) -> torch.Tensor:
  """Height scan with a MID360-like sparse visible region.

  The policy input size stays identical to the ideal terrain raycast, but cells
  outside the head-mounted MID360's useful near-front region, plus randomly
  dropped visible cells, are replaced by ``miss_value``.
  """

  scan = envs_mdp.height_scan(
    env,
    sensor_name=sensor_name,
    miss_value=miss_value,
  )

  num_x = max(1, int(round(size_x_m / resolution_m)) + 1)
  num_y = max(1, int(round(size_y_m / resolution_m)) + 1)
  if scan.shape[1] != num_x * num_y:
    return scan

  device = scan.device
  x = (
    torch.arange(num_x, device=device, dtype=torch.float32) * resolution_m
    - size_x_m * 0.5
    + x_offset_m
  )
  y = (
    torch.arange(num_y, device=device, dtype=torch.float32) * resolution_m
    - size_y_m * 0.5
    + y_offset_m
  )
  grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")

  visible = (grid_x >= visible_x_min_m) & (grid_x <= visible_x_max_m)
  if visible_y_abs_m is not None:
    visible = visible & (torch.abs(grid_y) <= visible_y_abs_m)
  visible = visible.flatten().unsqueeze(0)

  keep_prob = torch.empty(
    (env.num_envs, 1), device=device, dtype=torch.float32
  ).uniform_(keep_prob_min, keep_prob_max)
  random_keep = torch.rand_like(scan) < keep_prob

  frame_keep = (
    torch.rand((env.num_envs, 1), device=device, dtype=torch.float32)
    >= frame_dropout_prob
  )
  keep_mask = visible & random_keep & frame_keep

  return torch.where(keep_mask, scan, torch.full_like(scan, miss_value))


def bridged_mid360_height_scan(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  mid360_sensor_name: str,
  *,
  size_x_m: float = 1.6,
  size_y_m: float = 1.0,
  resolution_m: float = 0.1,
  x_offset_m: float = 0.0,
  y_offset_m: float = 0.0,
  visible_x_min_m: float = 0.25,
  visible_x_max_m: float = 1.20,
  visible_y_abs_m: float | None = None,
  keep_prob_min: float = 0.65,
  keep_prob_max: float = 1.0,
  frame_dropout_prob: float = 0.02,
  switch_start_step: int = 500 * 24,
  switch_end_step: int = 2500 * 24,
  sparse_start_step: int = 2500 * 24,
  sparse_end_step: int = 6000 * 24,
  fixed_curriculum_step: int | None = None,
  miss_value: float = 5.0,
) -> torch.Tensor:
  """Bridge old dense terrain scans to MID360-like sparse scans.

  The old checkpoint expects a dense, pelvis-centered ``terrain_scan``. This
  function keeps that input early in fine-tuning, then gradually moves a larger
  fraction of environments to the forward MID360 scan. Random sparsity is phased
  in later so PPO does not immediately optimize around a broken rollout.
  """

  dense_scan = envs_mdp.height_scan(
    env,
    sensor_name=sensor_name,
    miss_value=miss_value,
  )
  mid360_scan = envs_mdp.height_scan(
    env,
    sensor_name=mid360_sensor_name,
    miss_value=miss_value,
  )

  if dense_scan.shape != mid360_scan.shape:
    return dense_scan

  num_x = max(1, int(round(size_x_m / resolution_m)) + 1)
  num_y = max(1, int(round(size_y_m / resolution_m)) + 1)
  if dense_scan.shape[1] != num_x * num_y:
    return dense_scan

  if fixed_curriculum_step is None:
    step = int(getattr(env, "common_step_counter", 0))
  else:
    step = fixed_curriculum_step

  def ramp(start: int, end: int) -> float:
    if end <= start:
      return 1.0 if step >= end else 0.0
    return min(max((step - start) / (end - start), 0.0), 1.0)

  switch_alpha = ramp(switch_start_step, switch_end_step)
  sparse_alpha = ramp(sparse_start_step, sparse_end_step)

  device = dense_scan.device
  x = (
    torch.arange(num_x, device=device, dtype=torch.float32) * resolution_m
    - size_x_m * 0.5
    + x_offset_m
  )
  y = (
    torch.arange(num_y, device=device, dtype=torch.float32) * resolution_m
    - size_y_m * 0.5
    + y_offset_m
  )
  grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")

  visible = (grid_x >= visible_x_min_m) & (grid_x <= visible_x_max_m)
  if visible_y_abs_m is not None:
    visible = visible & (torch.abs(grid_y) <= visible_y_abs_m)
  visible = visible.flatten().unsqueeze(0)

  effective_keep_min = 1.0 + (keep_prob_min - 1.0) * sparse_alpha
  effective_keep_max = 1.0 + (keep_prob_max - 1.0) * sparse_alpha
  effective_dropout = frame_dropout_prob * sparse_alpha

  keep_prob = torch.empty(
    (env.num_envs, 1), device=device, dtype=torch.float32
  ).uniform_(effective_keep_min, effective_keep_max)
  random_keep = torch.rand_like(mid360_scan) < keep_prob
  frame_keep = (
    torch.rand((env.num_envs, 1), device=device, dtype=torch.float32)
    >= effective_dropout
  )
  keep_mask = visible & random_keep & frame_keep
  sparse_scan = torch.where(
    keep_mask, mid360_scan, torch.full_like(mid360_scan, miss_value)
  )

  use_mid360 = (
    torch.rand((env.num_envs, 1), device=device, dtype=torch.float32)
    < switch_alpha
  )
  return torch.where(use_mid360, sparse_scan, dense_scan)


class bridged_mid360_height_scan_memory:
  """MID360-like scan with compressed short-term memory for the near blind zone."""

  def __init__(self):
    self.memory: torch.Tensor | None = None
    self.age: torch.Tensor | None = None
    self.forward_residual: torch.Tensor | None = None

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if self.memory is None or self.age is None:
      return
    if env_ids is None:
      self.memory.fill_(0.0)
      self.age.fill_(10_000)
      if self.forward_residual is not None:
        self.forward_residual.fill_(0.0)
      return
    self.memory[env_ids] = 0.0
    self.age[env_ids] = 10_000
    if self.forward_residual is not None:
      self.forward_residual[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    mid360_sensor_name: str,
    *,
    size_x_m: float = 1.6,
    size_y_m: float = 1.0,
    resolution_m: float = 0.1,
    x_offset_m: float = 0.0,
    y_offset_m: float = 0.0,
    visible_x_min_m: float = 0.25,
    visible_x_max_m: float = 1.20,
    visible_y_abs_m: float | None = None,
    keep_prob_min: float = 0.65,
    keep_prob_max: float = 1.0,
    frame_dropout_prob: float = 0.02,
    switch_start_step: int = 500 * 24,
    switch_end_step: int = 2500 * 24,
    sparse_start_step: int = 2500 * 24,
    sparse_end_step: int = 6000 * 24,
    fixed_curriculum_step: int | None = None,
    memory_max_age_s: float = 0.8,
    enable_memory_shift: bool = True,
    miss_value: float = 5.0,
  ) -> torch.Tensor:
    dense_scan = envs_mdp.height_scan(
      env,
      sensor_name=sensor_name,
      miss_value=miss_value,
    )
    mid360_scan = envs_mdp.height_scan(
      env,
      sensor_name=mid360_sensor_name,
      miss_value=miss_value,
    )

    if dense_scan.shape != mid360_scan.shape:
      return dense_scan

    num_x = max(1, int(round(size_x_m / resolution_m)) + 1)
    num_y = max(1, int(round(size_y_m / resolution_m)) + 1)
    if dense_scan.shape[1] != num_x * num_y:
      return dense_scan

    if fixed_curriculum_step is None:
      step = int(getattr(env, "common_step_counter", 0))
    else:
      step = fixed_curriculum_step

    def ramp(start: int, end: int) -> float:
      if end <= start:
        return 1.0 if step >= end else 0.0
      return min(max((step - start) / (end - start), 0.0), 1.0)

    switch_alpha = ramp(switch_start_step, switch_end_step)
    sparse_alpha = ramp(sparse_start_step, sparse_end_step)

    device = dense_scan.device
    x = (
      torch.arange(num_x, device=device, dtype=torch.float32) * resolution_m
      - size_x_m * 0.5
      + x_offset_m
    )
    y = (
      torch.arange(num_y, device=device, dtype=torch.float32) * resolution_m
      - size_y_m * 0.5
      + y_offset_m
    )
    grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")

    visible = (grid_x >= visible_x_min_m) & (grid_x <= visible_x_max_m)
    if visible_y_abs_m is not None:
      visible = visible & (torch.abs(grid_y) <= visible_y_abs_m)
    visible = visible.flatten().unsqueeze(0)

    effective_keep_min = 1.0 + (keep_prob_min - 1.0) * sparse_alpha
    effective_keep_max = 1.0 + (keep_prob_max - 1.0) * sparse_alpha
    effective_dropout = frame_dropout_prob * sparse_alpha

    keep_prob = torch.empty(
      (env.num_envs, 1), device=device, dtype=torch.float32
    ).uniform_(effective_keep_min, effective_keep_max)
    random_keep = torch.rand_like(mid360_scan) < keep_prob
    frame_keep = (
      torch.rand((env.num_envs, 1), device=device, dtype=torch.float32)
      >= effective_dropout
    )
    keep_mask = visible & random_keep & frame_keep
    sparse_scan = torch.where(
      keep_mask, mid360_scan, torch.full_like(mid360_scan, miss_value)
    )

    if self.memory is None or self.memory.shape != sparse_scan.shape:
      self.memory = torch.full_like(sparse_scan, miss_value)
      self.age = torch.full_like(sparse_scan, 10_000, dtype=torch.int32)
      self.forward_residual = torch.zeros(
        env.num_envs, device=device, dtype=torch.float32
      )
    assert self.age is not None
    assert self.forward_residual is not None

    if enable_memory_shift and env.step_dt > 0:
      robot: Entity = env.scene["robot"]
      dx = torch.clamp(robot.data.root_link_lin_vel_b[:, 0] * env.step_dt, min=0.0)
      self.forward_residual = self.forward_residual + dx
      shift_cells = torch.floor(self.forward_residual / resolution_m).to(torch.long)
      self.forward_residual = self.forward_residual - shift_cells.float() * resolution_m

      memory_grid = self.memory.reshape(env.num_envs, num_y, num_x)
      age_grid = self.age.reshape(env.num_envs, num_y, num_x)
      for shift in torch.unique(shift_cells).tolist():
        if shift <= 0:
          continue
        env_ids = torch.nonzero(shift_cells == shift, as_tuple=False).flatten()
        shift = min(int(shift), num_x)
        if shift >= num_x:
          memory_grid[env_ids] = miss_value
          age_grid[env_ids] = 10_000
        else:
          memory_grid[env_ids, :, :-shift] = memory_grid[env_ids, :, shift:].clone()
          memory_grid[env_ids, :, -shift:] = miss_value
          age_grid[env_ids, :, :-shift] = age_grid[env_ids, :, shift:].clone()
          age_grid[env_ids, :, -shift:] = 10_000

    self.memory = torch.where(keep_mask, sparse_scan, self.memory)
    self.age = torch.where(
      keep_mask,
      torch.zeros_like(self.age),
      torch.clamp(self.age + 1, max=10_000),
    )

    max_age_steps = max(1, int(round(memory_max_age_s / env.step_dt)))
    memory_valid = self.age <= max_age_steps
    memory_scan = torch.where(
      memory_valid, self.memory, torch.full_like(self.memory, miss_value)
    )
    remembered_sparse_scan = torch.where(keep_mask, sparse_scan, memory_scan)

    env.extras.setdefault("log", {})["Metrics/height_scan_memory_fill_ratio"] = (
      (~keep_mask & memory_valid).float().mean()
    )

    use_mid360 = (
      torch.rand((env.num_envs, 1), device=device, dtype=torch.float32)
      < switch_alpha
    )
    return torch.where(use_mid360, remembered_sparse_scan, dense_scan)
