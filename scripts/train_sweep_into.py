#!/usr/bin/env python3
"""Task wrapper for RAPID sweep_into task.

This task has no scalar joint to measure progress, so we expose a task-specific
progress scalar:
- scalar_abs = height of cube above the hole threshold (0.967m). Zero or negative means success.
"""

from __future__ import annotations

import argparse
from typing import Tuple

import numpy as np

import train_core as core


class SweepIntoTaskSpec(core.IsaacLabTaskSpec):
    def __init__(self) -> None:
        self.name = "sweep_into"
        self.description = "RAPID: Sweep Into (sweep cube into table hole)"

        self.gym_id = "Isaac-Sweep-Into-Franka-v0"

        def _cfg_factory():
            args = getattr(core, "_LAST_PARSED_ARGS", None)
            use_ik = bool(getattr(args, "ik_rel_actions", False)) if args is not None else False
            if use_ik:
                from rapid_isaaclab.tasks.sweep_into.config.franka.metaworld_ik_rel_env_cfg import (
                    FrankaSweepIntoIKRelEnvCfg,
                )

                return FrankaSweepIntoIKRelEnvCfg()
            else:
                from rapid_isaaclab.tasks.sweep_into.config.franka.metaworld_joint_pos_env_cfg import (
                    FrankaSweepIntoEnvCfg,
                )

                return FrankaSweepIntoEnvCfg()

        self.cfg_factory = _cfg_factory

        self.env_spacing = 100.0

        self.scene_entity = "robot"
        self.joint_name = ""
        self.obs_term_name = ""
        self.fallback_obs_index = None

        self.default_exp_name = "sweep_into"
        self.train_min_log_key = "min_hole_dist"
        self.eval_mean_min_key = "mean_min_hole_dist"
        self.eval_print_label = "min_hole_dist"
        self.best_min_ckpt_prefix = "best_min_hole_dist"

    def add_task_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--ik_rel_actions",
            action="store_true",
            help=(
                "Use task-space relative IK actions instead of joint-position actions. "
                "This often makes exploration/step-size tuning easier for sweeping the cube."
            ),
        )
        parser.add_argument(
            "--hole_z_threshold",
            type=float,
            default=0.967,
            help="Z position threshold (m) below which the cube is considered inside the hole.",
        )
        # The following args are for offline preference collection
        parser.add_argument(
            "--hole_center_local_x",
            type=float,
            default=0.0,
            help="Hole center x-position (m) in the saved table frame for offline preference collection.",
        )
        parser.add_argument(
            "--hole_center_local_y",
            type=float,
            default=0.24,
            help="Hole center y-position (m) in the saved table frame for offline preference collection.",
        )
        parser.add_argument(
            "--hole_half_extent_x",
            type=float,
            default=0.08,
            help="Half-width (m) of the square hole region along table-frame x for offline preference collection.",
        )
        parser.add_argument(
            "--hole_half_extent_y",
            type=float,
            default=0.08,
            help="Half-width (m) of the square hole region along table-frame y for offline preference collection.",
        )
        parser.add_argument(
            "--sweep_success_mode",
            type=str,
            default="inside_xy",
            choices=["inside_xy", "inside_xy_and_z"],
            help=(
                "Success definition saved for offline preference collection. "
                "inside_xy treats any cube inside the square hole region as successful; "
                "inside_xy_and_z also requires cube_z below hole_z_threshold."
            ),
        )

    def get_goal_abs(self, args: argparse.Namespace) -> float:
        return 0.0

    def get_init_joint_abs(self, args: argparse.Namespace) -> float | None:
        return None

    def get_vlm_env_name(self, args: argparse.Namespace) -> str:
        return "isaaclab_sweep_into_cube_in_hole"

    @staticmethod
    def _distance_to_hole(env, hole_z_threshold: float):
        """Compute distance to hole for all envs (torch tensor shape (N,)). 
        Returns z - threshold, where positive means above hole, zero or negative means success."""
        import torch

        base_env = getattr(env, "unwrapped", env)
        cube = base_env.scene["cube"]

        # Get cube z position in world frame
        cube_pos_w = cube.data.root_pos_w
        cube_z = cube_pos_w[:, 2]

        # Distance above threshold (0 or negative means success)
        dist = cube_z - hole_z_threshold
        
        # Clamp to 0 minimum so that we return 0 for success, not negative values
        dist = torch.clamp(dist, min=0.0)
        
        return dist

    # ---- Overrides used by train_isaaclab_core ----

    def get_scalar_abs_from_scene(self, env) -> float | None:
        try:
            args = getattr(core, "_LAST_PARSED_ARGS", None)
            threshold = float(getattr(args, "hole_z_threshold", 0.967)) if args is not None else 0.967
            d = self._distance_to_hole(env, threshold)
            v = d[0]
            return float(v.item()) if hasattr(v, "item") else float(v)
        except Exception:
            return None

    def get_scalar_abs_vec_from_scene(self, env, num_envs: int) -> np.ndarray | None:
        try:
            args = getattr(core, "_LAST_PARSED_ARGS", None)
            threshold = float(getattr(args, "hole_z_threshold", 0.967)) if args is not None else 0.967
            d = self._distance_to_hole(env, threshold)
            d = d[: int(num_envs)]
            return d.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
        except Exception:
            return None


if __name__ == "__main__":
    core.main(task=SweepIntoTaskSpec())
