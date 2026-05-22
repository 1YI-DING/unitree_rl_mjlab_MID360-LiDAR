"""Unitree G1 velocity environment configurations."""

from copy import deepcopy
from dataclasses import replace

from src.assets.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from src.tasks.velocity import mdp
from src.tasks.velocity.velocity_env_cfg import OffsetGridPatternCfg, make_velocity_env_cfg


def _set_fixed_stair_terrain(
  cfg: ManagerBasedRlEnvCfg,
  *,
  step_height_m: float,
  stair_mode: str = "up",
) -> None:
  """Restrict rough terrain generation to fixed-height stairs for play testing."""
  if cfg.scene.terrain is None or cfg.scene.terrain.terrain_generator is None:
    return

  terrain = cfg.scene.terrain
  generator = terrain.terrain_generator

  generator.curriculum = False
  generator.num_cols = 1
  generator.num_rows = 1
  generator.border_width = 10.0
  terrain.max_init_terrain_level = 0

  if stair_mode == "up":
    stair_names = {"pyramid_stairs_inv"}
  elif stair_mode == "down":
    stair_names = {"pyramid_stairs"}
  elif stair_mode == "mixed":
    stair_names = {"pyramid_stairs", "pyramid_stairs_inv"}
  else:
    raise ValueError(f"Unsupported stair_mode: {stair_mode}")

  updated_sub_terrains = {}
  for name, sub_cfg in generator.sub_terrains.items():
    proportion = 1.0 / len(stair_names) if name in stair_names else 0.0
    if name in {"pyramid_stairs", "pyramid_stairs_inv"}:
      updated_sub_terrains[name] = replace(
        sub_cfg,
        proportion=proportion,
        step_height_range=(step_height_m, step_height_m),
      )
    else:
      updated_sub_terrains[name] = replace(sub_cfg, proportion=0.0)
  generator.sub_terrains = updated_sub_terrains


def _perceptive_rough_terrain_cfg(play: bool = False):
  """Bias rough terrain towards stair-like obstacles for lidar-guided locomotion."""
  terrain_cfg = deepcopy(ROUGH_TERRAINS_CFG)
  terrain_cfg.curriculum = not play
  terrain_cfg.num_rows = 5 if play else 10
  terrain_cfg.num_cols = 5 if play else 6
  terrain_cfg.border_width = 10.0 if play else 20.0

  stair_height_up = (0.14, 0.14) if play else (0.04, 0.16)
  stair_height_down = (0.14, 0.14) if play else (0.03, 0.14)

  updated_sub_terrains = {}
  for name, sub_cfg in terrain_cfg.sub_terrains.items():
    if name == "flat":
      updated_sub_terrains[name] = replace(sub_cfg, proportion=0.15)
    elif name == "pyramid_stairs":
      updated_sub_terrains[name] = replace(
        sub_cfg,
        proportion=0.30,
        step_height_range=stair_height_up,
        step_width=0.45,
      )
    elif name == "pyramid_stairs_inv":
      updated_sub_terrains[name] = replace(
        sub_cfg,
        proportion=0.20,
        step_height_range=stair_height_down,
        step_width=0.45,
      )
    elif name == "hf_pyramid_slope":
      updated_sub_terrains[name] = replace(
        sub_cfg,
        proportion=0.10,
        slope_range=(0.0, 0.35),
        platform_width=2.5,
        border_width=0.5,
      )
    elif name == "hf_pyramid_slope_inv":
      updated_sub_terrains[name] = replace(
        sub_cfg,
        proportion=0.05,
        slope_range=(0.0, 0.30),
        platform_width=2.5,
        border_width=0.5,
      )
    elif name == "random_rough":
      updated_sub_terrains[name] = replace(
        sub_cfg,
        proportion=0.10,
        noise_range=(0.005, 0.05),
        noise_step=0.01,
        border_width=0.5,
      )
    elif name == "wave_terrain":
      updated_sub_terrains[name] = replace(
        sub_cfg,
        proportion=0.10,
        amplitude_range=(0.0, 0.08),
        num_waves=3,
        border_width=0.5,
      )
    else:
      updated_sub_terrains[name] = replace(sub_cfg, proportion=0.0)
  terrain_cfg.sub_terrains = updated_sub_terrains
  return terrain_cfg


def _perceptive_rough_v2_terrain_cfg(play: bool = False):
  """Bias perceptive rough training toward unavoidable stair ascent."""
  terrain_cfg = deepcopy(ROUGH_TERRAINS_CFG)
  terrain_cfg.curriculum = not play
  terrain_cfg.num_rows = 5 if play else 10
  terrain_cfg.num_cols = 5 if play else 6
  terrain_cfg.border_width = 10.0 if play else 20.0

  stair_height_up = (0.14, 0.14) if play else (0.04, 0.18)
  stair_height_down = (0.14, 0.14) if play else (0.04, 0.14)

  updated_sub_terrains = {}
  for name, sub_cfg in terrain_cfg.sub_terrains.items():
    if name == "flat":
      updated_sub_terrains[name] = replace(sub_cfg, proportion=0.08)
    elif name == "pyramid_stairs":
      updated_sub_terrains[name] = replace(
        sub_cfg,
        proportion=0.14,
        step_height_range=stair_height_down,
        step_width=0.40,
      )
    elif name == "pyramid_stairs_inv":
      updated_sub_terrains[name] = replace(
        sub_cfg,
        proportion=0.30,
        step_height_range=stair_height_up,
        step_width=0.40,
      )
    elif name == "hf_pyramid_slope":
      updated_sub_terrains[name] = replace(
        sub_cfg,
        proportion=0.08,
        slope_range=(0.0, 0.40),
        platform_width=2.0,
        border_width=0.5,
      )
    elif name == "hf_pyramid_slope_inv":
      updated_sub_terrains[name] = replace(
        sub_cfg,
        proportion=0.18,
        slope_range=(0.0, 0.35),
        platform_width=2.0,
        border_width=0.5,
      )
    elif name == "random_rough":
      updated_sub_terrains[name] = replace(
        sub_cfg,
        proportion=0.12,
        noise_range=(0.005, 0.06),
        noise_step=0.01,
        border_width=0.5,
      )
    elif name == "wave_terrain":
      updated_sub_terrains[name] = replace(
        sub_cfg,
        proportion=0.10,
        amplitude_range=(0.0, 0.10),
        num_waves=3,
        border_width=0.5,
      )
    else:
      updated_sub_terrains[name] = replace(sub_cfg, proportion=0.0)
  terrain_cfg.sub_terrains = updated_sub_terrains
  return terrain_cfg


def _perceptive_rough_v2_stepup_terrain_cfg(play: bool = False):
  """Bias v2 fine-tuning toward forward ascent rather than generic rough walking."""
  terrain_cfg = _perceptive_rough_v2_terrain_cfg(play=play)
  updated_sub_terrains = {}
  for name, sub_cfg in terrain_cfg.sub_terrains.items():
    if name == "flat":
      updated_sub_terrains[name] = replace(sub_cfg, proportion=0.05)
    elif name == "pyramid_stairs":
      updated_sub_terrains[name] = replace(sub_cfg, proportion=0.05)
    elif name == "pyramid_stairs_inv":
      updated_sub_terrains[name] = replace(sub_cfg, proportion=0.60)
    elif name == "hf_pyramid_slope":
      updated_sub_terrains[name] = replace(sub_cfg, proportion=0.04)
    elif name == "hf_pyramid_slope_inv":
      updated_sub_terrains[name] = replace(sub_cfg, proportion=0.06)
    elif name == "random_rough":
      updated_sub_terrains[name] = replace(sub_cfg, proportion=0.10)
    elif name == "wave_terrain":
      updated_sub_terrains[name] = replace(sub_cfg, proportion=0.10)
    else:
      updated_sub_terrains[name] = replace(sub_cfg, proportion=0.0)
  terrain_cfg.sub_terrains = updated_sub_terrains
  return terrain_cfg


def unitree_g1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 128

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  # Set raycast sensor frame to G1 pelvis.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "pelvis"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Rationale for std values:
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # - Waist roll/pitch stay tight to keep the torso upright and stable.
  # - Shoulders/elbows get moderate freedom for natural arm swing during walking.
  # - Wrists are loose (0.3) since they don't affect balance much.
  # Running values are ~1.5-2x walking values to accommodate larger motion range.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.15,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.15,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.25,
    r".*hip_yaw.*": 0.25,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.25,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.25,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
  }

  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_g1_perceptive_rough_v1_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create a stair-focused rough task that keeps lidar-compatible height scan input."""
  cfg = unitree_g1_rough_env_cfg(play=play)

  cfg.sim.nconmax = 256
  cfg.sim.njmax = 4096
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 128

  if cfg.scene.terrain is not None:
    cfg.scene.terrain.terrain_generator = _perceptive_rough_terrain_cfg(play=play)
    cfg.scene.terrain.max_init_terrain_level = 2

  actor_terms = cfg.observations["actor"].terms
  actor_terms["base_ang_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)
  actor_terms["projected_gravity"].noise = Unoise(n_min=-0.06, n_max=0.06)
  actor_terms["joint_pos"].noise = Unoise(n_min=-0.012, n_max=0.012)
  actor_terms["joint_vel"].noise = Unoise(n_min=-1.8, n_max=1.8)
  actor_terms["height_scan"].noise = Unoise(n_min=-0.12, n_max=0.12)

  actor_terms["height_scan"].delay_min_lag = 0
  actor_terms["height_scan"].delay_max_lag = 1
  actor_terms["height_scan"].delay_hold_prob = 0.35
  actor_terms["height_scan"].delay_update_period = 2

  cfg.rewards["track_linear_velocity"].weight = 1.5
  cfg.rewards["track_angular_velocity"].weight = 0.75
  cfg.rewards["body_orientation_l2"].weight = -1.5
  cfg.rewards["body_ang_vel"].weight = -0.08
  cfg.rewards["angular_momentum"].weight = -0.03
  cfg.rewards["action_rate_l2"].weight = -0.04
  cfg.rewards["foot_clearance"].weight = -1.5
  cfg.rewards["foot_clearance"].params["target_height"] = 0.16
  cfg.rewards["foot_slip"].weight = -0.35
  cfg.rewards["soft_landing"].weight = -0.003
  cfg.rewards["stand_still"].weight = -0.5
  cfg.rewards["self_collisions"].weight = -2.0

  cfg.rewards["foot_swing_height"] = RewardTermCfg(
    func=mdp.feet_swing_height,
    weight=-0.75,
    params={
      "sensor_name": "feet_ground_contact",
      "target_height": 0.18,
      "command_name": "twist",
      "command_threshold": 0.1,
      "relative_to_lowest_foot": False,
      "asset_cfg": deepcopy(cfg.rewards["foot_clearance"].params["asset_cfg"]),
    },
  )

  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (0.0, 0.5),
        "lin_vel_y": (-0.15, 0.15),
        "ang_vel_z": (-0.25, 0.25),
      },
      {
        "step": 5000 * 24,
        "lin_vel_x": (0.0, 0.8),
        "lin_vel_y": (-0.2, 0.2),
        "ang_vel_z": (-0.35, 0.35),
      },
      {
        "step": 10000 * 24,
        "lin_vel_x": (-0.1, 1.0),
        "lin_vel_y": (-0.3, 0.3),
        "ang_vel_z": (-0.5, 0.5),
      },
      {
        "step": 20000 * 24,
        "lin_vel_x": (-0.2, 1.2),
        "lin_vel_y": (-0.4, 0.4),
        "ang_vel_z": (-0.7, 0.7),
      },
    ]

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  if play:
    twist_cmd.ranges.lin_vel_x = (0.0, 0.6)
    twist_cmd.ranges.lin_vel_y = (-0.2, 0.2)
    twist_cmd.ranges.ang_vel_z = (-0.35, 0.35)

  return cfg


def unitree_g1_perceptive_rough_v2_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create a sim2real-oriented perceptive rough task with stronger sensing mismatch."""
  cfg = unitree_g1_perceptive_rough_v1_env_cfg(play=play)

  mid360_scan = {
    "size_x_m": 1.6,
    "size_y_m": 1.0,
    "resolution_m": 0.1,
    "x_offset_m": 0.35,
    "y_offset_m": 0.0,
  }

  mid360_sensor_name = "terrain_scan_mid360"
  mid360_sensor_exists = False
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      mid360_sensor = deepcopy(sensor)
      mid360_sensor.name = mid360_sensor_name
      mid360_sensor.pattern = OffsetGridPatternCfg(
        size=(mid360_scan["size_x_m"], mid360_scan["size_y_m"]),
        resolution=mid360_scan["resolution_m"],
        x_offset=mid360_scan["x_offset_m"],
        y_offset=mid360_scan["y_offset_m"],
      )
      mid360_sensor_exists = True
      break
  if mid360_sensor_exists:
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (mid360_sensor,)

  if cfg.scene.terrain is not None:
    cfg.scene.terrain.terrain_generator = _perceptive_rough_v2_terrain_cfg(play=play)
    cfg.scene.terrain.max_init_terrain_level = 1 if play else 2

  cfg.observations["actor"].history_length = 3
  cfg.observations["critic"].history_length = 1

  actor_terms = cfg.observations["actor"].terms
  actor_terms["base_ang_vel"].noise = Unoise(n_min=-0.3, n_max=0.3)
  actor_terms["projected_gravity"].noise = Unoise(n_min=-0.08, n_max=0.08)
  actor_terms["joint_pos"].noise = Unoise(n_min=-0.015, n_max=0.015)
  actor_terms["joint_vel"].noise = Unoise(n_min=-2.0, n_max=2.0)
  actor_terms["actions"].noise = Unoise(n_min=-0.03, n_max=0.03)
  actor_terms["height_scan"].func = mdp.bridged_mid360_height_scan_memory()
  mid360_randomization = (
    {
      "keep_prob_min": 1.0,
      "keep_prob_max": 1.0,
      "frame_dropout_prob": 0.0,
      "fixed_curriculum_step": 6000 * 24,
    }
    if play
    else {
      "keep_prob_min": 0.65,
      "keep_prob_max": 1.0,
      "frame_dropout_prob": 0.02,
      "fixed_curriculum_step": None,
    }
  )
  actor_terms["height_scan"].params = {
    "sensor_name": "terrain_scan",
    "mid360_sensor_name": mid360_sensor_name,
    **mid360_scan,
    "visible_x_min_m": 0.25,
    "visible_x_max_m": 1.15,
    "visible_y_abs_m": 0.50,
    **mid360_randomization,
    "switch_start_step": 500 * 24,
    "switch_end_step": 2500 * 24,
    "sparse_start_step": 2500 * 24,
    "sparse_end_step": 6000 * 24,
    "miss_value": 5.0,
    "memory_max_age_s": 0.8,
    "enable_memory_shift": True,
  }
  actor_terms["height_scan"].noise = Unoise(n_min=-0.15, n_max=0.15)
  actor_terms["height_scan"].delay_min_lag = 0
  actor_terms["height_scan"].delay_max_lag = 0 if play else 2
  actor_terms["height_scan"].delay_hold_prob = 0.0 if play else 0.35
  actor_terms["height_scan"].delay_update_period = 0 if play else 3

  cfg.commands["twist"].rel_standing_envs = 0.03

  cfg.events["foot_friction"].params["ranges"] = (0.25, 1.8)
  cfg.events["encoder_bias"].params["bias_range"] = (-0.02, 0.02)
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.075, 0.075),
    1: (-0.05, 0.05),
    2: (-0.04, 0.04),
  }

  cfg.rewards["track_linear_velocity"].weight = 1.5
  cfg.rewards["track_angular_velocity"].weight = 0.7
  cfg.rewards["pose"].weight = 0.75
  cfg.rewards["body_orientation_l2"].weight = -1.6
  cfg.rewards["body_ang_vel"].weight = -0.10
  cfg.rewards["angular_momentum"].weight = -0.035
  cfg.rewards["action_rate_l2"].weight = -0.06
  cfg.rewards["foot_clearance"].weight = -1.8
  cfg.rewards["foot_clearance"].params["target_height"] = 0.22
  cfg.rewards["foot_clearance"].params["relative_to_lowest_foot"] = True
  cfg.rewards["foot_slip"].weight = -0.45
  cfg.rewards["soft_landing"].weight = -0.004
  cfg.rewards["stand_still"].weight = -0.5
  cfg.rewards["self_collisions"].weight = -2.5
  cfg.rewards["foot_swing_height"].weight = -1.0
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.24
  cfg.rewards["foot_swing_height"].params["relative_to_lowest_foot"] = True

  # Allow more hip/knee/ankle pitch articulation when stepping onto obstacles.
  cfg.rewards["pose"].params["std_walking"].update(
    {
      r".*hip_pitch.*": 0.6,
      r".*knee.*": 0.6,
      r".*ankle_pitch.*": 0.22,
    }
  )
  cfg.rewards["pose"].params["std_running"].update(
    {
      r".*hip_pitch.*": 0.65,
      r".*knee.*": 0.65,
      r".*ankle_pitch.*": 0.3,
    }
  )

  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (0.15, 0.45),
        "lin_vel_y": (-0.05, 0.05),
        "ang_vel_z": (-0.10, 0.10),
      },
      {
        "step": 4000 * 24,
        "lin_vel_x": (0.15, 0.65),
        "lin_vel_y": (-0.08, 0.08),
        "ang_vel_z": (-0.18, 0.18),
      },
      {
        "step": 9000 * 24,
        "lin_vel_x": (0.10, 0.90),
        "lin_vel_y": (-0.12, 0.12),
        "ang_vel_z": (-0.30, 0.30),
      },
      {
        "step": 16000 * 24,
        "lin_vel_x": (0.05, 1.10),
        "lin_vel_y": (-0.18, 0.18),
        "ang_vel_z": (-0.45, 0.45),
      },
    ]

  if not play:
    cfg.curriculum["track_linear_velocity_weight"] = CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
        "reward_name": "track_linear_velocity",
        "weight_stages": [
          {"step": 0, "weight": 1.5},
          {"step": 7000 * 24, "weight": 1.35},
          {"step": 14000 * 24, "weight": 1.2},
        ],
      },
    )
    cfg.curriculum["foot_clearance_weight"] = CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
        "reward_name": "foot_clearance",
        "weight_stages": [
          {"step": 0, "weight": -1.8},
          {"step": 8000 * 24, "weight": -1.4},
          {"step": 14000 * 24, "weight": -1.1},
        ],
      },
    )
    cfg.curriculum["foot_swing_height_weight"] = CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
        "reward_name": "foot_swing_height",
        "weight_stages": [
          {"step": 0, "weight": -1.0},
          {"step": 8000 * 24, "weight": -0.75},
          {"step": 14000 * 24, "weight": -0.5},
        ],
      },
    )
    cfg.curriculum["pose_weight"] = CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
        "reward_name": "pose",
        "weight_stages": [
          {"step": 0, "weight": 0.75},
          {"step": 8000 * 24, "weight": 0.6},
          {"step": 14000 * 24, "weight": 0.5},
        ],
      },
    )

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  if play:
    twist_cmd.ranges.lin_vel_x = (0.20, 0.55)
    twist_cmd.ranges.lin_vel_y = (-0.10, 0.10)
    twist_cmd.ranges.ang_vel_z = (-0.10, 0.10)

  return cfg


def unitree_g1_rough_step14_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create G1 rough configuration with fixed 14 cm stairs for sim ascent testing."""
  cfg = unitree_g1_rough_env_cfg(play=play)
  _set_fixed_stair_terrain(cfg, step_height_m=0.14, stair_mode="up")
  return cfg


def unitree_g1_rough_step14_down_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create G1 rough configuration with fixed 14 cm stairs for sim descent testing."""
  cfg = unitree_g1_rough_env_cfg(play=play)
  _set_fixed_stair_terrain(cfg, step_height_m=0.14, stair_mode="down")
  return cfg


def unitree_g1_perceptive_rough_v2_step14_up_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create G1 perceptive rough v2 configuration with fixed 14 cm ascent stairs."""
  cfg = unitree_g1_perceptive_rough_v2_env_cfg(play=play)
  _set_fixed_stair_terrain(cfg, step_height_m=0.14, stair_mode="up")
  return cfg


def unitree_g1_perceptive_rough_v2_straight_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create G1 perceptive rough v2 configuration with strictly straight commands."""
  cfg = unitree_g1_perceptive_rough_v2_env_cfg(play=play)

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.heading_command = False
  twist_cmd.rel_heading_envs = 0.0
  twist_cmd.rel_standing_envs = 0.0
  twist_cmd.ranges.lin_vel_x = (0.4, 0.4) if play else (0.25, 0.6)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
  twist_cmd.ranges.heading = None

  if "command_vel" in cfg.curriculum:
    cfg.curriculum.pop("command_vel")

  return cfg


def unitree_g1_perceptive_rough_v2_stepup_finetune_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a v2 fine-tuning task that rewards forward foot placement on ascent."""
  cfg = unitree_g1_perceptive_rough_v2_env_cfg(play=play)

  mid360_scan = {
    "size_x_m": 1.6,
    "size_y_m": 1.0,
    "resolution_m": 0.1,
    "x_offset_m": 0.35,
    "y_offset_m": 0.0,
  }

  if cfg.scene.terrain is not None:
    cfg.scene.terrain.terrain_generator = _perceptive_rough_v2_stepup_terrain_cfg(
      play=play
    )
    cfg.scene.terrain.max_init_terrain_level = 1 if play else 2

  # Keep the 59998 stair-clearance behavior, but prevent the conservative
  # low-step gait observed after the first step-up fine-tune.
  cfg.rewards["track_linear_velocity"].weight = 2.0
  cfg.rewards["track_angular_velocity"].weight = 1.15
  cfg.rewards["body_orientation_l2"].weight = -1.8
  cfg.rewards["foot_clearance"].weight = -1.7
  cfg.rewards["foot_clearance"].params["target_height"] = 0.22
  cfg.rewards["foot_swing_height"].weight = -1.1
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.24
  cfg.rewards["soft_landing"].weight = -0.006

  feet_asset_cfg = deepcopy(cfg.rewards["foot_clearance"].params["asset_cfg"])
  cfg.rewards["foot_swing_forward"] = RewardTermCfg(
    func=mdp.feet_swing_forward,
    weight=-0.6,
    params={
      "sensor_name": "feet_ground_contact",
      "target_x": 0.16,
      "command_name": "twist",
      "command_threshold": 0.1,
      "asset_cfg": feet_asset_cfg,
    },
  )
  cfg.rewards["foot_landing_forward"] = RewardTermCfg(
    func=mdp.feet_landing_forward,
    weight=-0.4,
    params={
      "sensor_name": "feet_ground_contact",
      "target_x": 0.16,
      "command_name": "twist",
      "command_threshold": 0.1,
      "asset_cfg": deepcopy(feet_asset_cfg),
    },
  )
  cfg.rewards["forward_velocity_floor"] = RewardTermCfg(
    func=mdp.forward_velocity_floor,
    weight=-1.2,
    params={
      "min_velocity": 0.22,
      "command_name": "twist",
      "command_threshold": 0.1,
    },
  )
  cfg.rewards["straight_yaw_rate_l2"] = RewardTermCfg(
    func=mdp.yaw_rate_l2_when_straight,
    weight=-0.60,
    params={
      "command_name": "twist",
      "command_x_threshold": 0.1,
      "command_y_threshold": 0.08,
      "command_yaw_threshold": 0.04,
    },
  )
  cfg.rewards["straight_heading_drift_l2"] = RewardTermCfg(
    func=mdp.heading_drift_l2_when_straight,
    weight=-0.85,
    params={
      "command_name": "twist",
      "command_x_threshold": 0.1,
      "command_y_threshold": 0.08,
      "command_yaw_threshold": 0.04,
    },
  )
  cfg.rewards["early_lift_without_step"] = RewardTermCfg(
    func=mdp.early_lift_without_step_penalty,
    weight=-0.35,
    params={
      "contact_sensor_name": "feet_ground_contact",
      "terrain_sensor_name": "terrain_scan_mid360",
      "command_name": "twist",
      "lift_height_threshold": 0.12,
      "obstacle_height_threshold": 0.06,
      "command_threshold": 0.1,
      "x_window": (0.25, 0.65),
      "y_abs": 0.35,
      **mid360_scan,
      "miss_value": 5.0,
      "asset_cfg": deepcopy(feet_asset_cfg),
    },
  )
  cfg.rewards["landing_edge_penalty"] = RewardTermCfg(
    func=mdp.feet_landing_edge_penalty,
    weight=-1.0,
    params={
      "contact_sensor_name": "feet_ground_contact",
      "terrain_sensor_name": "terrain_scan_mid360",
      "command_name": "twist",
      "edge_threshold": 0.06,
      "command_threshold": 0.1,
      **mid360_scan,
      "miss_value": 5.0,
      "asset_cfg": deepcopy(feet_asset_cfg),
    },
  )

  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (0.25, 0.50),
        "lin_vel_y": (-0.04, 0.04),
        "ang_vel_z": (0.0, 0.0),
      },
      {
        "step": 4000 * 24,
        "lin_vel_x": (0.30, 0.60),
        "lin_vel_y": (-0.06, 0.06),
        "ang_vel_z": (0.0, 0.0),
      },
    ]

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  if play:
    twist_cmd.ranges.lin_vel_x = (0.35, 0.45)
    twist_cmd.ranges.lin_vel_y = (-0.10, 0.10)
    twist_cmd.ranges.ang_vel_z = (-0.10, 0.10)

  return cfg


def unitree_g1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain velocity configuration."""
  cfg = unitree_g1_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg
