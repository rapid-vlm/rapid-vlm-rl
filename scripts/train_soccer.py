#!/usr/bin/env python3
"""Task wrapper for RAPID soccer task.

This task has no scalar joint to measure progress, so we expose a task-specific
progress scalar:
- scalar_abs = distance (meters) from ball position to the goal success AABB
  (0 if inside the box).
"""

from __future__ import annotations

import argparse
from typing import Tuple

import numpy as np

import train_core as core


class SoccerTaskSpec(core.IsaacLabTaskSpec):
    def __init__(self) -> None:
        self.name = "soccer"
        self.description = "RAPID: Soccer (put ball in goal net)"

        self.gym_id = "Isaac-Soccer-Franka-v0"

        def _cfg_factory():
            args = getattr(core, "_LAST_PARSED_ARGS", None)
            use_ik = bool(getattr(args, "ik_rel_actions", False)) if args is not None else False
            if use_ik:
                from rapid_isaaclab.tasks.soccer.config.franka.metaworld_ik_rel_env_cfg import (
                    FrankaSoccerIKRelEnvCfg,
                )

                return FrankaSoccerIKRelEnvCfg()
            else:
                from rapid_isaaclab.tasks.soccer.config.franka.metaworld_joint_pos_env_cfg import (
                    FrankaSoccerEnvCfg,
                )

                return FrankaSoccerEnvCfg()

        self.cfg_factory = _cfg_factory

        self.env_spacing = 100.0

        self.scene_entity = "robot"
        self.joint_name = ""
        self.obs_term_name = ""
        self.fallback_obs_index = None

        self.default_exp_name = "soccer"
        self.train_min_log_key = "min_goal_dist"
        self.eval_mean_min_key = "mean_min_goal_dist"
        self.eval_print_label = "min_goal_dist"
        self.best_min_ckpt_prefix = "best_min_goal_dist"

    def add_task_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--ik_rel_actions",
            action="store_true",
            help=(
                "Use task-space relative IK actions instead of joint-position actions. "
                "This often makes exploration/step-size tuning easier for pushing the ball."
            ),
        )
        parser.add_argument(
            "--goal_half_extents",
            type=float,
            nargs=3,
            default=(0.055, 0.095, 0.075),
            metavar=("HX", "HY", "HZ"),
            help="Goal success box half extents (m) in the GOAL FRAME. Ball is successful if inside this oriented box.",
        )
        parser.add_argument(
            "--goal_offset",
            type=float,
            nargs=3,
            default=(0.0, 0.0, 0.0),
            metavar=("OX", "OY", "OZ"),
            help="Offset (m) of the success box center expressed in the GOAL FRAME.",
        )

    def get_goal_abs(self, args: argparse.Namespace) -> float:
        # Our scalar is a distance-to-success-region; goal is distance==0
        return 0.0

    def get_init_joint_abs(self, args: argparse.Namespace) -> float | None:
        return None

    def get_vlm_env_name(self, args: argparse.Namespace) -> str:
        return "isaaclab_soccer_put_ball_in_goal"

    @staticmethod
    def _goal_params(args: argparse.Namespace) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        half_raw = getattr(args, "goal_half_extents", (0.08, 0.18, 0.12))
        offs_raw = getattr(args, "goal_offset", (0.0, 0.0, 0.0))
        half = (float(half_raw[0]), float(half_raw[1]), float(half_raw[2]))
        offs = (float(offs_raw[0]), float(offs_raw[1]), float(offs_raw[2]))
        return (half, offs)

    @staticmethod
    def _distance_to_goal_box_goal_frame(env, half_extents: Tuple[float, float, float], offset: Tuple[float, float, float]):
        """Compute distance to the goal-frame success box for all envs (torch tensor shape (N,))."""
        import torch
        from isaaclab.utils.math import quat_apply_inverse

        base_env = getattr(env, "unwrapped", env)
        ball = base_env.scene["ball"]
        goal = base_env.scene["goal"]

        # Transform ball position into goal frame.
        rel_w = ball.data.root_pos_w - goal.data.root_pos_w
        rel_g = quat_apply_inverse(goal.data.root_quat_w, rel_w)

        center = torch.tensor(offset, device=rel_g.device, dtype=rel_g.dtype)
        half = torch.tensor(half_extents, device=rel_g.device, dtype=rel_g.dtype)

        # Distance from point to AABB (in goal frame): 0 if inside, else L2 norm of outside distances.
        q = torch.abs(rel_g - center) - half
        outside = torch.clamp(q, min=0.0)
        dist = torch.linalg.norm(outside, dim=1)
        return dist

    # ---- Overrides used by train_isaaclab_core ----

    def get_scalar_abs_from_scene(self, env) -> float | None:
        try:
            args = getattr(core, "_LAST_PARSED_ARGS", None)
            half, offs = self._goal_params(args) if args is not None else ((0.08, 0.18, 0.12), (0.0, 0.0, 0.0))
            d = self._distance_to_goal_box_goal_frame(env, half, offs)
            v = d[0]
            return float(v.item()) if hasattr(v, "item") else float(v)
        except Exception:
            return None

    def get_scalar_abs_vec_from_scene(self, env, num_envs: int) -> np.ndarray | None:
        try:
            args = getattr(core, "_LAST_PARSED_ARGS", None)
            half, offs = self._goal_params(args) if args is not None else ((0.08, 0.18, 0.12), (0.0, 0.0, 0.0))
            d = self._distance_to_goal_box_goal_frame(env, half, offs)
            d = d[: int(num_envs)]
            return d.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
        except Exception:
            return None


if __name__ == "__main__":
    core.main(task=SoccerTaskSpec())
