# Copyright (c) 2026, The RAPID Authors.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerData

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ball_pos(env: ManagerBasedRLEnv, ball_cfg: SceneEntityCfg = SceneEntityCfg("ball")) -> torch.Tensor:
    """Ball position in environment frame (world - env_origin)."""
    ball: RigidObject = env.scene[ball_cfg.name]
    return ball.data.root_pos_w - env.scene.env_origins


def ball_lin_vel(env: ManagerBasedRLEnv, ball_cfg: SceneEntityCfg = SceneEntityCfg("ball")) -> torch.Tensor:
    """Ball linear velocity in world frame."""
    ball: RigidObject = env.scene[ball_cfg.name]
    return ball.data.root_lin_vel_w


def rel_ee_ball(env: ManagerBasedRLEnv, ball_cfg: SceneEntityCfg = SceneEntityCfg("ball")) -> torch.Tensor:
    """Vector from end-effector TCP to the ball (world frame)."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ball: RigidObject = env.scene[ball_cfg.name]
    return ball.data.root_pos_w - ee_tf_data.target_pos_w[..., 0, :]


def rel_ball_goal(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
) -> torch.Tensor:
    """Vector from ball to goal reference (world frame)."""
    ball: RigidObject = env.scene[ball_cfg.name]
    goal: RigidObject = env.scene[goal_cfg.name]
    return goal.data.root_pos_w - ball.data.root_pos_w


def fingertips_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The position of the fingertips relative to the environment origins."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    fingertips_pos = ee_tf_data.target_pos_w[..., 1:, :] - env.scene.env_origins.unsqueeze(1)

    return fingertips_pos.view(env.num_envs, -1)


def ee_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The position of the end-effector relative to the environment origins."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_pos = ee_tf_data.target_pos_w[..., 0, :] - env.scene.env_origins

    return ee_pos


def ee_quat(env: ManagerBasedRLEnv, make_quat_unique: bool = True) -> torch.Tensor:
    """The orientation of the end-effector in the environment frame.

    If :attr:`make_quat_unique` is True, the quaternion is made unique by ensuring the real part is positive.
    """
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_quat = ee_tf_data.target_quat_w[..., 0, :]
    # make first element of quaternion positive
    return math_utils.quat_unique(ee_quat) if make_quat_unique else ee_quat
