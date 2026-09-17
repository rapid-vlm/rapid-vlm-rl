# Copyright (c) 2026, The RAPID Authors.
# SPDX-License-Identifier: MIT

from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from rapid_isaaclab.tasks.window_open import mdp

from rapid_isaaclab.tasks.window_open.metaworld_window_open_env_cfg import (  # isort: skip
    FRAME_MARKER_SMALL_CFG,
    WindowOpenEnvCfg,
)

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


@configclass
class FrankaWindowEnvCfg(WindowOpenEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = FRANKA_PANDA_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=FRANKA_PANDA_CFG.init_state.replace(pos=(0.1, 0.0, 0.92))
            )  # type: ignore

        self.actions.arm_action = mdp.EMAJointPositionToLimitsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            alpha=0.5,
        )  # type: ignore
        
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

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
