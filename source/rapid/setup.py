"""Initialization helpers for the RAPID training loop."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from rapid.agent.sac import SACAgent
from rapid.isaaclab_utils import flatten_obs_batch
from rapid.replay_buffer import ReplayBuffer
from rapid.reward_model import RewardModel


def detect_obs_action_dims(env, num_envs: int):
    """Determine observation and action dimensions.

    Returns:
        (obs_dim, obs_shape, act_dim, action_range)
    """
    obs0, _ = env.reset()
    obs_arr0 = flatten_obs_batch(obs0, num_envs)[0]
    obs_dim = int(np.asarray(obs_arr0).reshape(-1).size)
    obs_shape = (obs_dim,)

    space = getattr(env, "action_space", None)
    if space is None:
        return obs_dim, obs_shape, 1, [-1.0, 1.0]

    try:
        act_sample = np.asarray(space.sample())
        act_dim = int(act_sample.shape[-1]) if act_sample.ndim >= 2 else int(act_sample.size)
        if hasattr(space, "low") and hasattr(space, "high"):
            low = np.asarray(space.low)
            high = np.asarray(space.high)
            if low.ndim >= 2:
                low = low.reshape(-1, low.shape[-1])[0]
            if high.ndim >= 2:
                high = high.reshape(-1, high.shape[-1])[0]
            action_range = [float(np.min(low)), float(np.max(high))]
        else:
            action_range = [-1.0, 1.0]
    except Exception:
        act_dim = 1
        action_range = [-1.0, 1.0]

    return obs_dim, obs_shape, act_dim, action_range


def detect_camera_resolution(env, default: int = 128) -> int:
    """Read camera resolution from the env's tiled_camera sensor.

    Falls back to *default* if the sensor cannot be inspected.
    """
    try:
        base_env = getattr(env, "unwrapped", env)
        scene = getattr(base_env, "scene", None)
        if scene is not None and hasattr(scene, "sensors"):
            if "tiled_camera" in scene.sensors:
                cam_cfg = getattr(scene.sensors["tiled_camera"], "cfg", None)
                if cam_cfg is not None:
                    w = getattr(cam_cfg, "width", default)
                    h = getattr(cam_cfg, "height", default)
                    if w == h:
                        print(f"[INFO] Using camera resolution from env: {w}x{h}")
                        return int(w)
    except Exception as e:
        print(f"[WARNING] Could not read camera resolution, using {default}x{default}: {e}")
    return default


def create_sac_agent(args, obs_dim: int, act_dim: int, action_range: list) -> SACAgent:
    """Create and return a SAC agent from parsed CLI arguments."""
    device = "cuda" if (args.device or "cuda").startswith("cuda") else "cpu"
    cfg = {
        "obs_dim": obs_dim,
        "action_dim": act_dim,
        "action_range": action_range,
        "device": device,
        "critic_cfg": {
            "_target_": "rapid.agent.critic.DoubleQCritic",
            "obs_dim": obs_dim,
            "action_dim": act_dim,
            "hidden_dim": args.hidden_dim,
            "hidden_depth": args.hidden_depth,
        },
        "actor_cfg": {
            "_target_": "rapid.agent.actor.DiagGaussianActor",
            "obs_dim": obs_dim,
            "action_dim": act_dim,
            "hidden_dim": args.hidden_dim,
            "hidden_depth": args.hidden_depth,
            "log_std_bounds": [-5, 2],
        },
        "discount": 0.99,
        "init_temperature": 0.1,
        "alpha_lr": 1e-4,
        "alpha_betas": [0.9, 0.999],
        "actor_lr": args.actor_lr,
        "actor_betas": [0.9, 0.999],
        "actor_update_frequency": 1,
        "critic_lr": args.critic_lr,
        "critic_betas": [0.9, 0.999],
        "critic_tau": 0.005,
        "critic_target_update_frequency": 2,
        "batch_size": args.batch_size,
        "learnable_temperature": True,
        "normalize_state_entropy": True,
    }
    return SACAgent(**cfg)


def create_replay_buffer(args, obs_shape, act_dim: int, image_hw: int,
                         device: str) -> ReplayBuffer:
    """Create a replay buffer with image storage enabled."""
    capacity = int(min(
        int(args.total_steps),
        int(max(1, int(getattr(args, 'replay_buffer_capacity', 200_000) or 200_000))),
    ))
    return ReplayBuffer(
        obs_shape,
        (act_dim,),
        capacity,
        device,
        store_image=True,
        image_size=image_hw,
    )


def create_reward_model(args, obs_dim: int, act_dim: int, image_hw: int,
                        log_dir: Path, vlm_env_name: str) -> RewardModel:
    """Create the VLM-based reward model."""
    return RewardModel(
        obs_dim,
        act_dim,
        ensemble_size=3,
        lr=3e-4,
        mb_size=args.reward_batch,
        size_segment=1,
        max_size=int(getattr(args, 'vlm_pool_size', 100)),
        activation="tanh",
        capacity=args.max_feedback * 2,
        large_batch=1,
        label_margin=0.0,
        teacher_beta=-1,
        teacher_gamma=1,
        teacher_eps_mistake=0,
        teacher_eps_skip=0,
        teacher_eps_equal=0,
        vlm_label=True,
        env_name=vlm_env_name,
        vlm="openrouter_gemma3",
        log_dir=str(log_dir),
        flip_vlm_label=False,
        cached_label_path=None,
        progress_bar=bool(args.vlm_progress),
        vlm_single_prompt=bool(getattr(args, 'one_prompt', False)),
        image_reward=True,
        image_height=image_hw,
        image_width=image_hw,
        resize_factor=1,
        resnet=False,
        conv_kernel_sizes=[5, 3, 3, 3],
        conv_n_channels=[16, 32, 64, 128],
        conv_strides=[3, 2, 2, 2],
    )


def load_checkpoint(args, agent: SACAgent, reward_model: RewardModel) -> int:
    """Load agent and reward model from a checkpoint.

    Returns the env_step to resume from (0 on failure or if no checkpoint).
    """
    if not (args.resume_from_checkpoint and args.resume_step):
        return 0

    ckpt_dir = Path(args.resume_from_checkpoint) / "models"
    if not ckpt_dir.exists():
        print(f"[RESUME] Checkpoint directory not found: {ckpt_dir}")
        print("[RESUME] Starting from scratch")
        return 0

    try:
        print(f"[RESUME] Loading checkpoint from {ckpt_dir} at step {args.resume_step}")
        agent.load(str(ckpt_dir.parent / "models"), args.resume_step)
        print(f"[RESUME] Successfully loaded agent checkpoint at step {args.resume_step}")
        try:
            reward_model.load(str(ckpt_dir.parent / "models"), args.resume_step)
            print(f"[RESUME] Successfully loaded reward model checkpoint at step {args.resume_step}")
        except Exception as e:
            print(f"[RESUME] Warning: Could not load reward model: {e}")
            print("[RESUME] Continuing with fresh reward model")
        print(f"[RESUME] Will resume from step {args.resume_step} to {args.total_steps}")
        return int(args.resume_step)
    except Exception as e:
        print(f"[RESUME] Error loading checkpoint: {e}")
        print("[RESUME] Starting from scratch")
        return 0
