# Copyright (c) 2026, The RAPID Authors.
# SPDX-License-Identifier: MIT
import os
_TASK_DIR = os.path.dirname(os.path.abspath(__file__))


from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
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
class CabinetSceneCfg(InteractiveSceneCfg):
    """Configuration for the cabinet scene with a robot and a drawer cabinet.

    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the robot and end-effector frames
    """

    # robots, Will be populated by agent env cfg
    robot: ArticulationCfg = MISSING # type: ignore
    # End-effector, Will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING # type: ignore

    cabinet = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{_TASK_DIR}/drawer.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.8, 0, 0.08),
            rot=euler_xyz_to_quat(0.0, 0.0, -90.0, degrees=True),
            joint_pos={
                "drawer_joint": -0.16, # open: -0.16, closed: 0.0
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["drawer_joint"],
                effort_limit_sim=87.0,
                stiffness=10.0,
                damping=1.0,
            ),
        },
    )

    # Frame definitions for the drawercabinet
    cabinet_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet/drawer",
        # Turn off debug_vis so the coordinate frame axes are not spawned (and won't appear in saved camera images).
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/CabinetFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Cabinet/drawer_handle",
                name="drawer_handle",
                offset=OffsetCfg(
                    pos=(0.305, 0.0, 0.01),
                    rot=(0.5, 0.5, -0.5, -0.5),  # align with end-effector frame
                ),
            ),
        ],
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(),
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
        
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.25, -0.35, 0.5),
            rot=euler_xyz_to_quat(-135.0, 0.0, -45.0, degrees=True),
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

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        cabinet_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_joint"])},
        )
        cabinet_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_joint"])},
        )
        rel_ee_drawer_distance = ObsTerm(func=mdp.rel_ee_drawer_distance)

        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material, # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.25),
            "dynamic_friction_range": (0.8, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    cabinet_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material, # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("cabinet", body_names="drawer_handle"),
            "static_friction_range": (1.0, 1.25),
            "dynamic_friction_range": (1.25, 1.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
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
class CabinetEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cabinet environment."""

    # Scene settings
    scene: CabinetSceneCfg = CabinetSceneCfg(num_envs=4096, env_spacing=2.0)
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
        self.decimation = 1
        self.episode_length_s = 4.0
       
        self.viewer.eye = (-2.0, 2.0, 2.0)
        self.viewer.lookat = (0.8, 0.0, 0.5)
        # simulation settings
        self.sim.dt = 1 / 50  # 60Hz 
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625

        # Fix ghosting/motion blur in camera images by disabling denoiser and adjusting anti-aliasing
        # See: https://forums.developer.nvidia.com/t/turn-offasset-fading-when-changed/342609
        self.sim.render.enable_dl_denoiser = False
        self.sim.render.antialiasing_mode = "FXAA"


