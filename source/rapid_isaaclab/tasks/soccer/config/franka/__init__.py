# Copyright (c) 2026, The RAPID Authors.
# SPDX-License-Identifier: MIT

import gymnasium as gym

##
# Register Gym environments.
##


gym.register(
    id="Isaac-Soccer-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.metaworld_joint_pos_env_cfg:FrankaSoccerEnvCfg",
    },
    disable_env_checker=True,
)