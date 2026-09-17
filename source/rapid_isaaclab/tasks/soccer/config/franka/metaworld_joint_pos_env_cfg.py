# Copyright (c) 2026, The RAPID Authors.
# SPDX-License-Identifier: MIT

from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from rapid_isaaclab.tasks.soccer import mdp

from rapid_isaaclab.tasks.soccer.metaworld_soccer_env_cfg import (  # isort: skip
    FRAME_MARKER_SMALL_CFG,
    SoccerEnvCfg,
)

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


@configclass
class FrankaSoccerEnvCfg(SoccerEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=FRANKA_PANDA_CFG.init_state.replace(
                pos=(0.0, 0.0, 0.92),
                joint_pos={
                    "panda_joint1": 0.0,
                    "panda_joint2": 0.287,
                    "panda_joint3": 0.0,
                    "panda_joint4": -2.03,
                    "panda_joint5": 0.0,
                    "panda_joint6": 2.58,
                    "panda_joint7": 0.741,
                    "panda_finger_joint.*": 0.04,
                },
            ),
            ) # type: ignore

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale={
                "panda_joint[1-5]": 0.8,
                "panda_joint[6-7]": 0.2,
            },
        ) # type: ignore

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        # Listens to the required transforms
        # IMPORTANT: The order of the frames in the list is important. The first frame is the tool center point (TCP)
        # the other frames are the fingers
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.1034),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )
