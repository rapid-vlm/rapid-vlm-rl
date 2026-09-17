# Adding a New Task


## 1. Create the IsaacLab task package

Add a directory under `source/rapid_isaaclab/tasks/<task_name>/`:

```
source/rapid_isaaclab/tasks/<task_name>/
├── __init__.py
├── <object_name>.usd
├── <task>_env_cfg.py
├── config/
│   └── franka/
│       ├── __init__.py                      # gym.register()
│       └── joint_pos_env_cfg.py
└── mdp/
    ├── __init__.py
    ├── observations.py
    └── rewards.py
```

### Base env config: `<task>_env_cfg.py`

This defines the scene (robot, objects, tiled camera) and the MDP (observations, rewards, terminations, events). See `source/rapid_isaaclab/tasks/button/button_env_cfg.py` as a reference.

> **Note on rewards:** you need to define at least a dummy reward term here. IsaacLab requires it to build the env, but it is not actually used during training (the VLM preference signal is the reward).

### Robot config

In `config/franka/joint_pos_env_cfg.py`:

```python
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

@configclass
class Franka<Task>EnvCfg(<Task>EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # configure arm/gripper actions and ee_frame sensor
```

### Gym registration

In `config/franka/__init__.py`:

```python
import gymnasium as gym

gym.register(
    id="Isaac-<TaskName>-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:Franka<Task>EnvCfg",
    },
    disable_env_checker=True,
)
```


## 2. Register the package

In `source/rapid_isaaclab/tasks/__init__.py`, add your task to the import line, this is what triggers the `gym.register()` call:

```python
from . import soccer, sweep_into, button, cabinet, window_open, <task_name>
```


## 3. Add a VLM prompt

In `source/rapid/prompt.py`, add one entry to `goal_env_prompts`.
```python
goal_env_prompts = {
    "isaaclab_cabinet_drawer_open": "to open the drawer",
    "isaaclab_button_push": "to push the button down",
    "isaaclab_soccer_put_ball_in_goal": "to move the soccer ball into the goal",
    "isaaclab_sweep_into_cube_in_hole": "to sweep the cube into the hole in the table",
    "isaaclab_window_open": "to slide the window open as much as possible",
    "isaaclab_<task_name>": "to <goal description>",   # <-- add this
}
```


## 4. Write the training script

Create `scripts/train_<task_name>.py`:

```python
#!/usr/bin/env python3
"""RAPID: <Task Name>."""
from __future__ import annotations
import argparse
import train_core as core


class <Task>TaskSpec(core.IsaacLabTaskSpec):
    def __init__(self) -> None:
        self.name = "<task_name>"
        self.description = "RAPID: <Task Name>"

        self.gym_id = "Isaac-<TaskName>-Franka-v0"

        def _cfg_factory():
            from rapid_isaaclab.tasks.manager_based.manipulation.<task_name>.config.franka.joint_pos_env_cfg import (
                Franka<Task>EnvCfg,
            )
            return Franka<Task>EnvCfg()

        self.cfg_factory = _cfg_factory
        self.env_spacing = 2.5          # increase if parallel envs interfere

        # scalar joint
        self.scene_entity = "<scene_object_name>"
        self.joint_name   = "<joint_name>"
        self.obs_term_name      = "<obs_term_key>"
        self.fallback_obs_index = None

        # logging
        self.default_exp_name    = "<task_name>"
        self.train_min_log_key   = "min_<metric>"
        self.eval_mean_min_key   = "mean_min_<metric>"
        self.eval_print_label    = "min_<metric>"
        self.best_min_ckpt_prefix = "best_min_<metric>"

    def add_task_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--goal_target", type=float, default=0.0,
                            help="Joint/scalar value considered a success.")

    def get_goal_abs(self, args: argparse.Namespace) -> float:
        return float(args.goal_target)

    def get_init_joint_abs(self, args: argparse.Namespace) -> float | None:
        return None
        
    def get_vlm_env_name(self, args: argparse.Namespace) -> str:
        return "isaaclab_<task_name>"   # must match the key in prompt.py


if __name__ == "__main__":
    core.main(task=<Task>TaskSpec())
```

### Object-position tasks (no joint to track)

For tasks like soccer or sweep_into there is no scalar joint, instead you compute a distance to the goal region. Set the joint fields to empty placeholders and override the scalar methods:

```python
        self.scene_entity       = "robot"   # unused, just a placeholder
        self.joint_name         = ""
        self.obs_term_name      = ""
        self.fallback_obs_index = None

    def get_init_joint_abs(self, args):
        return None

    def get_goal_abs(self, args):
        return 0.0  # success when distance == 0

    def get_scalar_abs_from_scene(self, env) -> float | None:
        """Non-negative distance to goal (0 = success, higher = further away)."""
        try:
            base_env = getattr(env, "unwrapped", env)
            obj = base_env.scene["<your_object>"]
            # compute and return your distance metric as a float
            ...
        except Exception:
            return None

    def get_scalar_abs_vec_from_scene(self, env, num_envs: int):
        """Same thing but for all envs, returns shape (num_envs,) array."""
        import numpy as np
        vals = []
        for i in range(num_envs):
            v = self.get_scalar_abs_from_scene(env)
            vals.append(v if v is not None else 0.0)
        return np.array(vals, dtype=np.float32)
```


## 5. Add to the eval script

`scripts/eval_record_policy.py` manages its own IsaacLab `AppLauncher`, so it has its own per-task setup.

### 5a. Env factory function

Add a `make_<task>_env()` function near the other `make_*_env()` definitions:

```python
def make_<task>_env(
    device: str | None,
    seed: int | None = None,
    camera_width: int = 0,
    camera_height: int = 0,
):
    from rapid_isaaclab.tasks.manager_based.manipulation.<task_name>.config.franka.joint_pos_env_cfg import (
        Franka<Task>EnvCfg,
    )
    cfg = Franka<Task>EnvCfg()
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 2.5
    try:
        if int(camera_width) > 0:
            cfg.scene.tiled_camera.width = int(camera_width)
        if int(camera_height) > 0:
            cfg.scene.tiled_camera.height = int(camera_height)
    except Exception:
        pass
    if device is not None:
        cfg.sim.device = device
    if seed is not None:
        try:
            cfg.seed = int(seed)
        except Exception:
            pass
    env = gym.make("Isaac-<TaskName>-Franka-v0", cfg=cfg, render_mode=None)
    return env
```

For joint-based tasks you can also set the initial joint position inside here (e.g. see `make_drawer_env()` for an example).

### 5b. `--task` choices

```python
choices=["drawer", "window_open", "button_push", "soccer", "sweep_into", "<task_name>"], # <-- add your task here
```

### 5c. Env creation branch

```python
elif task_id == '<task_name>':
    env = make_<task>_env(
        args.device,
        seed=(int(args.seed) if args.seed is not None else None),
        camera_width=int(getattr(args, 'camera_width', 0) or 0),
        camera_height=int(getattr(args, 'camera_height', 0) or 0),
    )
```

### 5d. Success scalar (`drawer_goal_abs`)

```python
elif task_id == '<task_name>':
    drawer_goal_abs = float(getattr(args, '<your_goal_arg>', <default_value>))
```

For object-position tasks simply use `drawer_goal_abs = 0.0`.

### 5e. `get_abs_fn` lambda

```python
# joint-based
elif task_id == '<task_name>':
    get_abs_fn = lambda e: _get_joint_abs_from_scene(e, scene_entity='<scene_entity>', joint_name_suffix='<joint_name>')

# object-position (custom distance function)
elif task_id == '<task_name>':
    threshold = float(getattr(args, '<threshold_arg>', <default>))
    get_abs_fn = lambda e: _<your>_distance(e, threshold=threshold)
```
