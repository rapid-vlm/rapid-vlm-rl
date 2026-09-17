# Copyright (c) 2026, The RAPID Authors.
# SPDX-License-Identifier: MIT
import os
_TASK_DIR = os.path.dirname(os.path.abspath(__file__))


from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg, CameraCfg, TiledCameraCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz
import math
import torch


def euler_xyz_to_quat(roll: float, pitch: float, yaw: float, degrees: bool = True) -> tuple[float, float, float, float]:
    """Convert Euler angles (XYZ order) to quaternion (w,x,y,z).

    Args:
        roll: rotation about X.
        pitch: rotation about Y.
        yaw: rotation about Z.
        degrees: If True, interpret inputs as degrees; else radians.
    Returns:
        Quaternion as (w, x, y, z) floats.
    """
    if degrees:
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)
    r = torch.tensor([roll], dtype=torch.float32)
    p = torch.tensor([pitch], dtype=torch.float32)
    y = torch.tensor([yaw], dtype=torch.float32)
    q = quat_from_euler_xyz(r, p, y)[0]
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))

from . import mdp

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy() # type: ignore
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


##
# Scene definition
##


@configclass
class SweepIntoSceneCfg(InteractiveSceneCfg):
    """Configuration for the sweep into scene with a robot, cube, and table with hole."""

    # robots, Will be populated by agent env cfg
    robot: ArticulationCfg = MISSING # type: ignore
    # End-effector, Will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING # type: ignore

    
    # Cube (rigid object)
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{_TASK_DIR}/cube.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.967),
        ),
    )

    table_with_hole = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{_TASK_DIR}/table_with_hole.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.0),
            rot=euler_xyz_to_quat(0.0, 0.0, -90.0, degrees=True),
        ),
    )

    robot_stand = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RobotStand",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{_TASK_DIR}/robot_stand.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.04, 0.0, 0.0),
            rot=euler_xyz_to_quat(0.0, 0.0, 0.0, degrees=True),
        ),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(size=(500.0, 500.0)),
        collision_group=-1,
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # Tiled camera (single sensor producing per-env tiles)
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/TiledCamera",
        # Per-tile resolution
        width=128,
        height=128,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            clipping_range=(0.05, 5.0),
        ),
        # sweep_into config
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.05, 0.0, 1.35),
            rot=euler_xyz_to_quat(-145.0, 0.0, 90.0, degrees=True),
            convention="ros",
        ),
    ) # type: ignore


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: mdp.JointPositionActionCfg = MISSING # type: ignore
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING # type: ignore

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=base_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=base_mdp.joint_vel_rel)

        ee_pos = ObsTerm(func=mdp.ee_pos)
        cube_pos = ObsTerm(func=mdp.cube_pos, params={"cube_cfg": SceneEntityCfg("cube")})
        cube_lin_vel = ObsTerm(func=mdp.cube_lin_vel, params={"cube_cfg": SceneEntityCfg("cube")})
        rel_ee_cube = ObsTerm(func=mdp.rel_ee_cube, params={"cube_cfg": SceneEntityCfg("cube")})
        rel_cube_hole = ObsTerm(
            func=mdp.rel_cube_hole,
            params={"cube_cfg": SceneEntityCfg("cube"), "table_cfg": SceneEntityCfg("table_with_hole")},
        )

        actions = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    robot_physics_material = EventTerm(
        func=base_mdp.randomize_rigid_body_material, # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.25),
            "dynamic_friction_range": (0.8, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    cube_physics_material = EventTerm(
        func=base_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("cube", body_names=".*"),
            "static_friction_range": (0.4, 1.0),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 16,
        },
    )

    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")

    reset_robot_joints = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    dummy_term = RewTerm(func=mdp.is_alive, weight=0.0)
    

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

##
# Environment configuration
##


@configclass
class SweepIntoEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the sweep into environment."""

    # Scene settings
    scene: SweepIntoSceneCfg = SweepIntoSceneCfg(num_envs=64, env_spacing=100.0)
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
        self.episode_length_s = 4.0
        self.viewer.eye = (-2.0, 2.0, 2.0)
        self.viewer.lookat = (0.8, 0.0, 0.5)
        # simulation settings
        self.sim.dt = 1 / 50  
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        
        # Fix ghosting/motion blur in camera images by disabling denoiser and adjusting anti-aliasing
        # See: https://forums.developer.nvidia.com/t/turn-offasset-fading-when-changed/342609
        self.sim.render.enable_dl_denoiser = False
        self.sim.render.antialiasing_mode = "FXAA"
