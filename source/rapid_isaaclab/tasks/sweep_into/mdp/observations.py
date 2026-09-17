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


def cube_pos(env: ManagerBasedRLEnv, cube_cfg: SceneEntityCfg = SceneEntityCfg("cube")) -> torch.Tensor:
    """Cube position in environment frame (world - env_origin)."""
    cube: RigidObject = env.scene[cube_cfg.name]
    return cube.data.root_pos_w - env.scene.env_origins


def cube_lin_vel(env: ManagerBasedRLEnv, cube_cfg: SceneEntityCfg = SceneEntityCfg("cube")) -> torch.Tensor:
    """Cube linear velocity in world frame."""
    cube: RigidObject = env.scene[cube_cfg.name]
    return cube.data.root_lin_vel_w


def rel_ee_cube(env: ManagerBasedRLEnv, cube_cfg: SceneEntityCfg = SceneEntityCfg("cube")) -> torch.Tensor:
    """Vector from end-effector TCP to the cube (world frame)."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    cube: RigidObject = env.scene[cube_cfg.name]
    return cube.data.root_pos_w - ee_tf_data.target_pos_w[..., 0, :]


def rel_cube_hole(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    table_cfg: SceneEntityCfg = SceneEntityCfg("table_with_hole"),
) -> torch.Tensor:
    """Vector from cube to hole center (world frame). The hole is in the center of the table."""
    cube: RigidObject = env.scene[cube_cfg.name]
    table: RigidObject = env.scene[table_cfg.name]
    # The hole is at the center of the table
    return table.data.root_pos_w - cube.data.root_pos_w


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
