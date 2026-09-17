# Copyright (c) 2026, The RAPID Authors.
# SPDX-License-Identifier: MIT
import os
_TASK_DIR = os.path.dirname(os.path.abspath(__file__))


from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg, TiledCameraCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz
import math
import torch

from . import mdp

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()  # type: ignore
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


def euler_xyz_to_quat(roll: float, pitch: float, yaw: float, degrees: bool = True) -> tuple[float, float, float, float]:
    """Convert Euler angles (XYZ order) to quaternion (w, x, y, z)."""
    if degrees:
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)
    r = torch.tensor([roll], dtype=torch.float32)
    p = torch.tensor([pitch], dtype=torch.float32)
    y = torch.tensor([yaw], dtype=torch.float32)
    q = quat_from_euler_xyz(r, p, y)[0]
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


@configclass
class WindowOpenSceneCfg(InteractiveSceneCfg):
    """Configuration for the window-open scene with a robot and a sliding window articulation."""

    # Robot and end-effector — populated by the Franka-specific subclass.
    robot: ArticulationCfg = MISSING  # type: ignore
    ee_frame: FrameTransformerCfg = MISSING  # type: ignore

    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{_TASK_DIR}/table_high_wall.usd",
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
            rot=euler_xyz_to_quat(0.0, 0.0, 90.0, degrees=True),
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
            pos=(0.06, 0.0, 0.0),
            rot=euler_xyz_to_quat(0.0, 0.0, 0.0, degrees=True),
        ),
    )

    window = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Window",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{_TASK_DIR}/window.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.8, 0.0, 1.11),
            rot=euler_xyz_to_quat(0.0, 0.0, -90.0, degrees=True),
            joint_pos={
                # Start fully closed; the policy learns to slide open to 0.2 m
                "window_joint": 0.0,
            },
        ),
        actuators={
            "windows": ImplicitActuatorCfg(
                joint_names_expr=["window_joint"],
                effort_limit_sim=87.0,
                stiffness=0.0,
                damping=5.0,
            ),
        },
    )

    window_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Window/window",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/WindowFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Window/window",
                name="window_handle",
                offset=OffsetCfg(
                    pos=(0.0, 0.15, 0.0),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
            ),
        ],
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/TiledCamera",
        width=128,
        height=128,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            clipping_range=(0.05, 5.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.2, 0.0, 1.35),
            rot=euler_xyz_to_quat(-110.0, 0.0, -90.0, degrees=True),
            convention="ros",
        ),
    )  # type: ignore


@configclass
class ActionsCfg:
    arm_action: mdp.JointPositionActionCfg = MISSING  # type: ignore
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING  # type: ignore


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        window_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("window", joint_names=["window_joint"])},
        )
        window_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("window", joint_names=["window_joint"])},
        )
        rel_ee_window_distance = ObsTerm(func=mdp.rel_ee_window_distance)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.25),
            "dynamic_friction_range": (0.8, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    window_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("window", body_names=".*"),
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
    dummy_term = RewTerm(func=mdp.is_alive, weight=0.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class WindowOpenEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the window-open environment."""

    scene: WindowOpenSceneCfg = WindowOpenSceneCfg(num_envs=4096, env_spacing=20.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 1
        self.episode_length_s = 4.0
        self.viewer.eye = (-2.0, 2.0, 2.0)
        self.viewer.lookat = (0.8, 0.0, 0.5)

        self.sim.dt = 1 / 50
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        
        # Fix ghosting/motion blur in camera images by disabling denoiser and adjusting anti-aliasing
        # See: https://forums.developer.nvidia.com/t/turn-offasset-fading-when-changed/342609
        self.sim.render.enable_dl_denoiser = False
        self.sim.render.antialiasing_mode = "FXAA"
