#!/usr/bin/env python3
"""Task wrapper for RAPID drawer open task."""

from __future__ import annotations

import argparse

import train_core as core


class DrawerOpenTaskSpec(core.IsaacLabTaskSpec):
    def __init__(self) -> None:
        self.name = "drawer_open"
        self.description = "RAPID: Open Drawer (VLM preferences)"

        self.gym_id = "Isaac-Open-Drawer-Franka-v1"

        def _cfg_factory():
            from rapid_isaaclab.tasks.cabinet.config.franka.metaworld_joint_pos_env_cfg import (
                FrankaCabinetEnvCfg,
            )

            return FrankaCabinetEnvCfg()

        self.cfg_factory = _cfg_factory
        self.env_spacing = 2.5

        self.scene_entity = "cabinet"
        self.joint_name = "drawer_joint"
        self.obs_term_name = "cabinet_joint_pos"
        self.fallback_obs_index = 18

        self.default_exp_name = "drawer_open"
        self.train_min_log_key = "min_drawer"
        self.eval_mean_min_key = "mean_min_drawer"
        self.eval_print_label = "min_drawer"
        self.best_min_ckpt_prefix = "best_min_drawer"

    def add_task_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--close_target",
            type=float,
            default=0.0,
            help="Absolute joint position for CLOSED drawer (used as initial state).",
        )
        parser.add_argument(
            "--open_target",
            type=float,
            default=-0.16,
            help="Absolute joint target for OPEN drawer. Success if |drawer_abs - open_target| <= success_thresh.",
        )

    def get_goal_abs(self, args: argparse.Namespace) -> float:
        return float(getattr(args, "open_target", -0.16))

    def get_init_joint_abs(self, args: argparse.Namespace) -> float | None:
        # Start closed then learn to open
        return float(getattr(args, "close_target", 0.0))

    def get_vlm_env_name(self, args: argparse.Namespace) -> str:
        return "isaaclab_cabinet_drawer_open"


if __name__ == "__main__":
    core.main(task=DrawerOpenTaskSpec())
