#!/usr/bin/env python3
"""Evaluate a trained SAC policy on IsaacLab and (optionally) save video (MP4) and/or GIF."""
import argparse
import os
from pathlib import Path
import sys
import numpy as np

from isaaclab.app import AppLauncher
import gymnasium as gym

# RAPID components
from rapid.agent.sac import SACAgent

import torch
from typing import Optional

def flatten_obs(obs):
    import numpy as np
    import torch as _torch
    def to_np(a):
        if isinstance(a, _torch.Tensor):
            return a.detach().cpu().numpy()
        return np.asarray(a)
    def flatten_any(x):
        if isinstance(x, dict):
            arrs = []
            for k in sorted(x.keys()):
                arrs.extend(flatten_any(x[k]))
            return arrs
        arr = to_np(x)
        return [arr.astype(np.float32, copy=False).reshape(-1)]
    parts = flatten_any(obs)
    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)


def make_drawer_env(
    device: str | None,
    close_target: float = 0.0,
    open_target: float = -0.16,
    seed: int | None = None,
    camera_width: int = 0,
    camera_height: int = 0,
):
    from rapid_isaaclab.tasks.cabinet.config.franka.metaworld_joint_pos_env_cfg import (
        FrankaCabinetEnvCfg,
    )

    cfg = FrankaCabinetEnvCfg()
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 2.5
    try:
        if int(camera_width) > 0:
            cfg.scene.tiled_camera.width = int(camera_width)
        if int(camera_height) > 0:
            cfg.scene.tiled_camera.height = int(camera_height)
    except Exception:
        pass
    # Start closed for drawer-open evaluation
    try:
        cfg.scene.cabinet.init_state.joint_pos["drawer_joint"] = float(close_target)
    except Exception:
        pass
    if device is not None:
        cfg.sim.device = device
    if seed is not None:
        try:
            cfg.seed = int(seed)
        except Exception:
            pass
    env = gym.make("Isaac-Open-Drawer-Franka-v1", cfg=cfg, render_mode=None)
    return env


def make_window_open_env(
    device: str | None,
    close_target: float = 0.0,
    seed: int | None = None,
    camera_width: int = 0,
    camera_height: int = 0,
):
    from rapid_isaaclab.tasks.window_open.config.franka.metaworld_joint_pos_env_cfg import (
        FrankaWindowEnvCfg,
    )

    cfg = FrankaWindowEnvCfg()
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 2.5
    try:
        if int(camera_width) > 0:
            cfg.scene.tiled_camera.width = int(camera_width)
        if int(camera_height) > 0:
            cfg.scene.tiled_camera.height = int(camera_height)
    except Exception:
        pass
    try:
        cfg.scene.window.init_state.joint_pos["window_joint"] = float(close_target)
    except Exception:
        pass
    if device is not None:
        cfg.sim.device = device
    if seed is not None:
        try:
            cfg.seed = int(seed)
        except Exception:
            pass
    env = gym.make("Isaac-Open-Window-Franka-v0", cfg=cfg, render_mode=None)
    return env


def make_button_env(
    device: str | None,
    start_pos: float = 0.0,
    seed: int | None = None,
    camera_width: int = 0,
    camera_height: int = 0,
):
    from rapid_isaaclab.tasks.button.config.franka.metaworld_joint_pos_env_cfg import (
        FrankaButtonEnvCfg,
    )

    cfg = FrankaButtonEnvCfg()
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 20.0
    try:
        if int(camera_width) > 0:
            cfg.scene.tiled_camera.width = int(camera_width)
        if int(camera_height) > 0:
            cfg.scene.tiled_camera.height = int(camera_height)
    except Exception:
        pass
    try:
        cfg.scene.buttonbox.init_state.joint_pos["button_joint"] = float(start_pos)
    except Exception:
        pass
    if device is not None:
        cfg.sim.device = device
    if seed is not None:
        try:
            cfg.seed = int(seed)
        except Exception:
            pass
    env = gym.make("Isaac-Push-Button-Franka-v0", cfg=cfg, render_mode=None)
    return env


def make_soccer_env(
    device: str | None,
    seed: int | None = None,
    camera_width: int = 0,
    camera_height: int = 0,
):
    from rapid_isaaclab.tasks.soccer.config.franka.metaworld_joint_pos_env_cfg import (
        FrankaSoccerEnvCfg,
    )

    cfg = FrankaSoccerEnvCfg()
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 100.0
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
    env = gym.make("Isaac-Soccer-Franka-v0", cfg=cfg, render_mode=None)
    return env


def make_sweep_into_env(
    device: str | None,
    seed: int | None = None,
    camera_width: int = 0,
    camera_height: int = 0,
):
    from rapid_isaaclab.tasks.sweep_into.config.franka.metaworld_joint_pos_env_cfg import (
        FrankaSweepIntoEnvCfg,
    )

    cfg = FrankaSweepIntoEnvCfg()
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 100.0
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
    env = gym.make("Isaac-Sweep-Into-Franka-v0", cfg=cfg, render_mode=None)
    return env


def steps_per_episode(env) -> int:
    # Derive episode steps from cfg (keep consistent with training)
    try:
        cfg = getattr(getattr(env, "cfg", None), "sim", None)
        episode_length_s = getattr(getattr(env, "cfg", None), "episode_length_s", 8.0)
        dt = getattr(cfg, "dt", 1 / 60)
        decimation = getattr(getattr(env, "cfg", None), "decimation", 1)
        return int(episode_length_s / dt / max(1, decimation))
    except Exception:
        return 480


def hide_show_robot(env, make_invisible: bool, env_index: int = 0):
    """Toggle robot visibility in the USD stage to ensure desired appearance in recordings."""
    try:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        prim_path = f"/World/envs/env_{env_index}/Robot"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return None
        attr = prim.GetAttribute("visibility")
        if not attr:
            return None
        old = attr.Get() or "inherited"
        if make_invisible:
            attr.Set("invisible")
        else:
            attr.Set("inherited")
        return old
    except Exception:
        return None


def record_rollouts(env,
                    agent: SACAgent,
                    episodes: int,
                    out_dir: Path | None,
                    save_gif: bool,
                    gif_fps: int,
                    mp4_fps: int):
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    import imageio
    try:
        import imageio_ffmpeg  # noqa: F401
        have_ffmpeg = True
    except Exception:
        have_ffmpeg = False

    for ep in range(episodes):
        obs, _ = env.reset()
        obs = flatten_obs(obs)
        ep_frames = []
        done = False
        # infer horizon like in training
        horizon = 480
        steps = 0
        while not done and steps < horizon:
            with torch.no_grad():
                action = agent.act(obs, sample=False)
            action_env = torch.from_numpy(np.asarray(action, dtype=np.float32)).unsqueeze(0)
            step_out = env.step(action_env)
            if len(step_out) == 5:
                next_obs, _, terminated, truncated, _ = step_out
                done = bool(terminated or truncated)
            else:
                next_obs, _, done, _ = step_out
            obs = flatten_obs(next_obs)

            frame = None
            try:
                frame = env.render()
                if isinstance(frame, dict):
                    frame = frame.get("rgb_array", None)
            except Exception:
                frame = None

            if frame is None:
                # Fallback to the tiled camera
                try:
                    base_env = getattr(env, "unwrapped", env)
                    scene = getattr(base_env, "scene", None)
                    cam = None
                    if scene is not None and hasattr(scene, "sensors"):
                        if "tiled_camera" in scene.sensors:
                            cam = scene.sensors["tiled_camera"]
                        elif "overhead_cam" in scene.sensors:
                            cam = scene.sensors["overhead_cam"]
                    if cam is not None:
                        outputs = getattr(cam.data, "output", {})
                        key = "rgb" if "rgb" in outputs else ("rgba" if "rgba" in outputs else None)
                        if key is not None:
                            img = outputs[key]
                            if hasattr(img, "cpu"):
                                img = img.cpu().numpy()
                            if img.ndim == 4 and img.shape[0] >= 1:
                                img = img[0]
                            if img.ndim == 3 and img.shape[-1] >= 3:
                                frame = img[..., :3]
                except Exception:
                    frame = None

            if frame is not None:
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                ep_frames.append(frame)
                frames.append(frame)
            steps += 1

        # Save per-episode outputs
        if out_dir is not None:
            epi_base = out_dir / f"episode_{ep:02d}"
            if ep_frames:
                if have_ffmpeg:
                    imageio.mimsave(str(epi_base.with_suffix('.mp4')), ep_frames, fps=mp4_fps)
                if save_gif:
                    gif_duration_ms = max(1, int(round(1000.0 / max(1, gif_fps))))
                    imageio.mimsave(str(epi_base.with_suffix('.gif')), ep_frames, duration=gif_duration_ms)

    # Also save a combined GIF/MP4 across all episodes
    if out_dir is not None and frames:
        if have_ffmpeg:
            imageio.mimsave(str((out_dir / 'all_episodes').with_suffix('.mp4')), frames, fps=mp4_fps)
        if save_gif:
            gif_duration_ms = max(1, int(round(1000.0 / max(1, gif_fps))))
            imageio.mimsave(str((out_dir / 'all_episodes').with_suffix('.gif')), frames, duration=gif_duration_ms)


def load_eval_checkpoint(agent: SACAgent, checkpoint_dir: Path, ck_tag: str) -> None:
    """Load the actor required for rollout."""
    actor_path = checkpoint_dir / f"actor_{ck_tag}.pt"

    if not actor_path.exists():
        raise FileNotFoundError(f"Missing required actor checkpoint: {actor_path}")

    agent.actor.load_state_dict(torch.load(str(actor_path), map_location=agent.device))


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained policy and (optionally) save video/GIF")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to models dir with actor_<step or tag>.pt")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--step", type=int, help="Step number used in saved checkpoint filenames (actor_<step>.pt)")
    group.add_argument("--tag", type=str, help="Tag used in checkpoint filename (actor_<tag>.pt), e.g., best_success_300000")
    parser.add_argument("--out_dir", type=str, default=str(Path("outputs/eval_videos").absolute()), help="Directory to save videos/GIFs; if omitted and not headless, acts like 'play' mode")
    parser.add_argument("--save_gif", action="store_true")
    parser.add_argument("--gif_fps", type=int, default=15)
    parser.add_argument("--mp4_fps", type=int, default=30)
    parser.add_argument(
        "--camera_width",
        type=int,
        default=0,
        help="Override per-tile camera width for recordings (0 keeps env default).",
    )
    parser.add_argument(
        "--camera_height",
        type=int,
        default=0,
        help="Override per-tile camera height for recordings (0 keeps env default).",
    )
    parser.add_argument("--hide_robot", action="store_true", help="Hide robot visuals (default: show robot)")
    parser.add_argument(
        "--task",
        type=str,
        default="drawer",
        choices=["drawer", "window_open", "button_push", "soccer", "sweep_into"],
        help="Which IsaacLab task/environment to evaluate.",
    )
    parser.add_argument("--close_target", type=float, default=0.0)
    parser.add_argument("--open_target", type=float, default=-0.16)
    parser.add_argument("--start_pos", type=float, default=0.0, help="(button_push) Initial absolute button joint position")
    parser.add_argument("--push_target", type=float, default=-0.06, help="(button_push) Target absolute button joint position (down)")
    parser.add_argument(
        "--goal_half_extents",
        type=float,
        nargs=3,
        default=(0.055, 0.095, 0.075),
        metavar=("HX", "HY", "HZ"),
        help="(soccer) Goal success box half extents (m) in the GOAL FRAME.",
    )
    parser.add_argument(
        "--goal_offset",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("OX", "OY", "OZ"),
        help="(soccer) Goal success box center offset (m) expressed in the GOAL FRAME.",
    )
    parser.add_argument(
        "--hole_z_threshold",
        type=float,
        default=0.967,
        help="(sweep_into) Z position threshold (m) below which the cube is considered inside the hole.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Set environment seed for deterministic eval (optional)")

    # App flags
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()

    # Ensure cameras enabled for rendering
    if hasattr(args, "enable_cameras"):
        setattr(args, "enable_cameras", True)

    # Initialize app
    sys.argv = [sys.argv[0]]
    app = AppLauncher(args)
    simulation_app = app.app

    # Build environment
    task_id = str(getattr(args, 'task', 'drawer')).lower().strip()
    if task_id == 'button_push':
        env = make_button_env(
            args.device,
            start_pos=float(getattr(args, 'start_pos', 0.0)),
            seed=(int(args.seed) if args.seed is not None else None),
            camera_width=int(getattr(args, 'camera_width', 0) or 0),
            camera_height=int(getattr(args, 'camera_height', 0) or 0),
        )
    elif task_id == 'window_open':
        env = make_window_open_env(
            args.device,
            close_target=float(getattr(args, 'close_target', 0.0)),
            seed=(int(args.seed) if args.seed is not None else None),
            camera_width=int(getattr(args, 'camera_width', 0) or 0),
            camera_height=int(getattr(args, 'camera_height', 0) or 0),
        )
    elif task_id == 'soccer':
        env = make_soccer_env(
            args.device,
            seed=(int(args.seed) if args.seed is not None else None),
            camera_width=int(getattr(args, 'camera_width', 0) or 0),
            camera_height=int(getattr(args, 'camera_height', 0) or 0),
        )
    elif task_id == 'sweep_into':
        env = make_sweep_into_env(
            args.device,
            seed=(int(args.seed) if args.seed is not None else None),
            camera_width=int(getattr(args, 'camera_width', 0) or 0),
            camera_height=int(getattr(args, 'camera_height', 0) or 0),
        )
    else:
        env = make_drawer_env(
            args.device,
            close_target=float(getattr(args, 'close_target', 0.0)),
            open_target=float(getattr(args, 'open_target', -0.16)),
            seed=(int(args.seed) if args.seed is not None else None),
            camera_width=int(getattr(args, 'camera_width', 0) or 0),
            camera_height=int(getattr(args, 'camera_height', 0) or 0),
        )

    # Ensure desired robot visibility (default: visible)
    hide_show_robot(env, bool(getattr(args, 'hide_robot', False)), env_index=0)

    obs0, _ = env.reset()
    obs_arr = flatten_obs(obs0)
    obs_dim = int(obs_arr.size)
    act_sample = np.asarray(env.action_space.sample())
    action_dim = int(act_sample.size)
    try:
        low = getattr(env.action_space, 'low', None)
        high = getattr(env.action_space, 'high', None)
        if low is not None and high is not None:
            action_range = [float(np.min(low)), float(np.max(high))]
        else:
            action_range = [-1.0, 1.0]
    except Exception:
        action_range = [-1.0, 1.0]

    critic_cfg = {
        "_target_": "rapid.agent.critic.DoubleQCritic",
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_dim": 256,
        "hidden_depth": 3,
    }
    actor_cfg = {
        "_target_": "rapid.agent.actor.DiagGaussianActor",
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_dim": 256,
        "hidden_depth": 3,
        "log_std_bounds": [-5, 2],
    }

    agent = SACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_range=action_range,
        device="cuda" if (args.device or "cuda").startswith("cuda") else "cpu",
        critic_cfg=critic_cfg,
        actor_cfg=actor_cfg,
        discount=0.99,
        init_temperature=0.1,
        alpha_lr=1e-4,
        alpha_betas=[0.9, 0.999],
        actor_lr=3e-4,
        actor_betas=[0.9, 0.999],
        actor_update_frequency=1,
        critic_lr=3e-4,
        critic_betas=[0.9, 0.999],
        critic_tau=0.005,
        critic_target_update_frequency=2,
        batch_size=512,
        learnable_temperature=True,
        normalize_state_entropy=True,
    )

    # Load checkpoint
    checkpoint_dir = Path(args.checkpoint_dir)

    ck_tag = str(args.step) if args.step is not None else str(args.tag)
    load_eval_checkpoint(agent, checkpoint_dir, ck_tag)

    # Record (or just play if out_dir is empty string)
    out_dir: Path | None
    try:
        out_dir = Path(args.out_dir) if args.out_dir else None
    except Exception:
        out_dir = Path("outputs/eval_videos").absolute()
    record_rollouts(
        env,
        agent,
        args.episodes,
        out_dir,
        save_gif=args.save_gif,
        gif_fps=args.gif_fps,
        mp4_fps=args.mp4_fps,
    )

    try:
        env.close()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
