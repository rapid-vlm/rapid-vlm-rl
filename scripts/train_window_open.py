#!/usr/bin/env python3
"""Task wrapper for RAPID sliding window open task."""

from __future__ import annotations

import argparse

import train_core as core


class WindowOpenTaskSpec(core.IsaacLabTaskSpec):
    def __init__(self) -> None:
        self.name = "window_open"
        self.description = "RAPID: Open Window (VLM preferences)"

        self.gym_id = "Isaac-Open-Window-Franka-v0"

        def _cfg_factory():
            from rapid_isaaclab.tasks.window_open.config.franka.metaworld_joint_pos_env_cfg import (
                FrankaWindowEnvCfg,
            )

            return FrankaWindowEnvCfg()

        self.cfg_factory = _cfg_factory
        self.env_spacing = 2.5

        self.scene_entity = "window"
        self.joint_name = "window_joint"
        self.obs_term_name = "window_joint_pos"
        self.fallback_obs_index = 18

        self.default_exp_name = "window_open"
        self.train_min_log_key = "min_window"
        self.eval_mean_min_key = "mean_min_window"
        self.eval_print_label = "min_window"
        self.best_min_ckpt_prefix = "best_min_window"

    def add_task_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--close_target",
            type=float,
            default=0.0,
            help="Absolute joint position for CLOSED window (lower limit, default: 0.0 m).",
        )
        parser.add_argument(
            "--open_target",
            type=float,
            default=0.2,
            help=(
                "Absolute joint target for OPEN window in metres "
                "(default: 0.2 m = fully open). "
                "Success if |window_abs - open_target| <= success_thresh."
            ),
        )

    def get_goal_abs(self, args: argparse.Namespace) -> float:
        return float(getattr(args, "open_target", 0.2))

    def get_init_joint_abs(self, args: argparse.Namespace) -> float | None:
        return float(getattr(args, "close_target", 0.0))

    def get_vlm_env_name(self, args: argparse.Namespace) -> str:
        return "isaaclab_window_open"


if __name__ == "__main__":
    core.main(task=WindowOpenTaskSpec())
