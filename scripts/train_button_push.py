#!/usr/bin/env python3
"""Task wrapper for RAPID button push task."""

from __future__ import annotations

import argparse

import train_core as core


class ButtonPushTaskSpec(core.IsaacLabTaskSpec):
    def __init__(self) -> None:
        self.name = "button_push"
        self.description = "RAPID: Push Button (VLM preferences)"

        self.gym_id = "Isaac-Push-Button-Franka-v0"

        def _cfg_factory():
            from rapid_isaaclab.tasks.button.config.franka.metaworld_joint_pos_env_cfg import (
                FrankaButtonEnvCfg,
            )

            return FrankaButtonEnvCfg()

        self.cfg_factory = _cfg_factory
        self.env_spacing = 20.0

        self.scene_entity = "buttonbox"
        self.joint_name = "button_joint"
        self.obs_term_name = "button_joint_pos"
        self.fallback_obs_index = 0

        self.default_exp_name = "button_push"
        self.train_min_log_key = "min_dist"
        self.eval_mean_min_key = "mean_min_dist"
        self.eval_print_label = "min_dist"
        self.best_min_ckpt_prefix = "best_min_dist"

    def add_task_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--start_pos",
            type=float,
            default=0.0,
            help="Absolute initial button joint position (up).",
        )
        parser.add_argument(
            "--push_target",
            type=float,
            default=-0.06,
            help="Absolute joint target for pushed button (down). Success if |button_abs - push_target| <= success_thresh.",
        )

    def get_goal_abs(self, args: argparse.Namespace) -> float:
        return float(getattr(args, "push_target", -0.06))

    def get_init_joint_abs(self, args: argparse.Namespace) -> float | None:
        return float(getattr(args, "start_pos", 0.0))

    def get_vlm_env_name(self, args: argparse.Namespace) -> str:
        return "isaaclab_button_push"


if __name__ == "__main__":
    core.main(task=ButtonPushTaskSpec())
