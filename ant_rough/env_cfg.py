# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import RewardsCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from . import mdp
from .terrain_cfg import SINGLE_DIFFICULTY_TERRAINS_CFG
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
import math

##
# Pre-defined configs
##
from isaaclab_assets.robots.ant import ANT_CFG  # isort: skip


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with an ant robot."""

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=SINGLE_DIFFICULTY_TERRAINS_CFG,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # robot
    robot = ANT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.usd_path = robot.spawn.usd_path.replace("ant_instanceable.usd", "ant.usd")
    robot.spawn.activate_contact_sensors = True
    robot.spawn.copy_from_source = True

    # Four feet remain available for future rewards; torso contact is diagnostic only.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_foot", history_length=3, debug_vis=False,
    )
    torso_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso", history_length=1, debug_vis=False,
    )
    # x: -0.4..1.4 m, y: -0.8..0.8 m relative to torso heading.
    # Rays originate high above the torso so uphill terrain is not above the ray origin.
    # The observation uses torso Z, NOT the ray launch height.
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso",
        offset=RayCasterCfg.OffsetCfg(pos=(0.5, 0.0, 5.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.8, 1.6)),
        mesh_prim_paths=["/World/ground"],
        max_distance=20.0,
        debug_vis=False,
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=[".*"], scale=7.5)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_yaw_roll = ObsTerm(func=mdp.base_yaw_roll)
        base_angle_to_target = ObsTerm(func=mdp.base_angle_to_target, params={"target_pos": (1000.0, 0.0, 0.0)})
        base_up_proj = ObsTerm(func=mdp.base_up_proj)
        base_heading_proj = ObsTerm(func=mdp.base_heading_proj, params={"target_pos": (1000.0, 0.0, 0.0)})
        joint_pos_norm = ObsTerm(func=mdp.joint_pos_limit_normalized)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.2)
        feet_body_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.1,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=["front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot"]
                )
            },
        )
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.terrain_heights_relative_to_torso,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "clip_height": 1.0},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # Must run before reset_base: reset_root_state_uniform reads env_origins.
    select_terrain = EventTerm(
        func=mdp.reselect_terrain, mode="reset", params={"margin_tiles": 2},
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {"z": (0.15, 0.15)}, "velocity_range": {}},
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.1, 0.1),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length is exceeded
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Administrative truncation before reaching the physical map edge; not a fall.
    map_boundary = DoneTerm(
        func=mdp.outside_map_safe_area, time_out=True, params={"margin_m": 2.0},
    )
    # (2) Terminate if the robot falls
    body_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.pi / 2})
    # An observation-only monitor: always False, never ends an episode.
    torso_contact_monitor = DoneTerm(
        func=mdp.TorsoContactMonitor,
        params={
            "sensor_cfg": SceneEntityCfg("torso_contact"), "force_threshold": 1.0,
            "stagnation_window_s": 1.0, "stagnation_speed_mps": 0.05,
            "target_pos": (1000.0, 0.0),
        },
    )


@configclass
class AntEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the MuJoCo-style Ant walking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=64, env_spacing=5.0, clone_in_fabric=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 16.0
        # simulation settings
        self.sim.dt = 1 / 120.0
        self.sim.render_interval = self.decimation
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (3.0, 3.0, 2.2)
        self.viewer.lookat = (0.3, 0.0, 0.0)
        self.sim.physx.bounce_threshold_velocity = 0.2
        # default friction material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0


def _expand_terrain(terrain, dimensions):
    """Keep the original slot/key order, which affects seeded terrain generation."""
    import copy

    base = copy.deepcopy(terrain.sub_terrains)
    expanded = {}
    for slot, (slope, height) in enumerate(dimensions):
        for kind, cfg in base.items():
            sub = copy.deepcopy(cfg)
            sub.proportion = 1.0
            if kind in ("slope", "slope_inv"):
                sub.slope_range = (slope, slope)
            elif kind in ("stairs", "stairs_inv"):
                sub.step_height_range = (height, height)
            elif kind == "boxes":
                sub.grid_height_range = (height, height)
            if hasattr(sub, "platform_width"):
                sub.platform_width = 2.0
            expanded[f"{kind}__slot{slot}"] = sub
    terrain.sub_terrains = expanded
    terrain.curriculum = False
    terrain.difficulty_range = (0.0, 0.0)


@configclass
class AntSelfEvaluationCfg(AntEnvCfg):
    """Frozen map71, scored using the unmodified original Ant reward class."""

    def __post_init__(self):
        super().__post_init__()
        terrain = self.scene.terrain.terrain_generator
        _expand_terrain(terrain, ((0.12, 0.04), (0.22, 0.065), (0.28, 0.08)))
        terrain.seed = 71
        self.scene.num_envs = 100


@configclass
class AntMediumPreviewCfg(AntEnvCfg):
    """Training geometry (map42), with original rewards for inference scoring."""

    def __post_init__(self):
        super().__post_init__()
        terrain = self.scene.terrain.terrain_generator
        _expand_terrain(terrain, ((0.20, 0.06),) * 3)
        terrain.seed = 42
