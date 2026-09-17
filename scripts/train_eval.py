"""
Policy evaluation utilities for RAPID.

Provides serial and parallel episode rollouts over the training environment.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from rapid.agent.sac import SACAgent
from rapid.isaaclab_utils import (
    _to_np,
    _write_video,
    capture_vlm_image,
    eval_mode,
    extract_scalar_rel_from_obs,
    flatten_obs_batch,
    get_active_task,
    get_scalar_abs_from_scene,
    get_scalar_abs_vec_from_scene,
    get_scalar_default_abs,
    get_scalar_slice_from_env,
)

def evaluate_policy_on_env(
    env_eval,
    agent: SACAgent,
    episodes: int,
    horizon: int,
    success_thresh: float,
    record_dir: Path | None = None,
    record_fps: int = 15,
    goal_abs: float = 0.0,
    disable_video_capture: bool = False,
    image_hw: int = 128,
) -> dict:
    """Evaluate the current policy using the provided env (no new env is spawned).

    Note
    - This avoids spawning a second Isaac Lab environment during training, which can fail or
      conflict with an already-instantiated USD stage. Instead, we temporarily use the same
      environment at episode boundaries for evaluation and then reset back to training.
    """
    successes = 0
    mins: list[float] = []
    ep_lengths: list[int] = []
    # Compute slice once for faster fallback path
    scalar_obs_slice = get_scalar_slice_from_env(env_eval)
    # NOTE: env_eval may be vectorized (num_envs>1) --> evaluation is defined on env0 only
    for _ in range(max(1, int(episodes))):
        # Start a fresh episode for evaluation
        obs, _ = env_eval.reset()
        try:
            # If obs is batched, take env0
            if isinstance(obs, dict):
                any_v = next(iter(obs.values()))
                n_eval = int(np.asarray(_to_np(any_v)).reshape(-1, np.asarray(_to_np(any_v)).shape[-1]).shape[0]) if np.asarray(_to_np(any_v)).ndim >= 2 else 1
            else:
                n_eval = int(np.asarray(_to_np(obs)).shape[0]) if np.asarray(_to_np(obs)).ndim >= 2 else 1
        except Exception:
            n_eval = 1
        obs = flatten_obs_batch(obs, max(1, n_eval))[0]
        done = False
        steps = 0
        min_dist = float("inf")
        frames: list[np.ndarray] = []
        while not done and steps < horizon:
            with torch.no_grad():
                action = agent.act(obs, sample=False)
            action_env = torch.from_numpy(np.asarray(action, dtype=np.float32)).unsqueeze(0)
            step_out = env_eval.step(action_env)
            if len(step_out) == 5:
                next_obs, _, terminated, truncated, _ = step_out
                try:
                    terminated0 = bool(np.asarray(_to_np(terminated)).reshape(-1)[0])
                except Exception:
                    terminated0 = False
                try:
                    truncated0 = bool(np.asarray(_to_np(truncated)).reshape(-1)[0])
                except Exception:
                    truncated0 = False
                done = bool(terminated0 or truncated0)
            else:
                next_obs, _, done, _ = step_out
            obs = flatten_obs_batch(next_obs, max(1, n_eval))[0]
            # Frame capture for optional video writing
            frame = None
            if not bool(disable_video_capture):
                try:
                    frame = capture_vlm_image(env_eval, save_path=None, step=steps, hide_robot_flag=False)
                except Exception:
                    frame = None
                if frame is None:
                    try:
                        frame = env_eval.render()
                        if isinstance(frame, dict):
                            frame = frame.get("rgb_array", None)
                    except Exception:
                        frame = None

                if frame is not None and isinstance(frame, np.ndarray) and frame.ndim == 3:
                    if record_dir is not None:
                        fr = frame if frame.dtype == np.uint8 else np.clip(frame, 0, 255).astype(np.uint8)
                        frames.append(fr)

            scalar_abs = get_scalar_abs_from_scene(env_eval)
            if scalar_abs is None:
                scalar_rel = extract_scalar_rel_from_obs(obs, scalar_obs_slice)
                if scalar_rel is not None:
                    try:
                        default_abs = get_scalar_default_abs(env_eval)
                        if default_abs is not None and np.isfinite(default_abs):
                            scalar_abs = float(scalar_rel + float(default_abs))
                    except Exception:
                        scalar_abs = None
            if scalar_abs is not None and np.isfinite(scalar_abs):
                dist = abs(float(scalar_abs) - float(goal_abs))
                min_dist = min(min_dist, dist)
            steps += 1
        mins.append(min_dist if np.isfinite(min_dist) else float("nan"))
        ep_lengths.append(int(steps))
        if np.isfinite(min_dist) and min_dist <= success_thresh:
            successes += 1
        # Save video for this episode if requested and capture enabled
        if (not bool(disable_video_capture)) and record_dir is not None and frames:
            try:
                record_dir.mkdir(parents=True, exist_ok=True)
                out_path = record_dir / f"episode_{len(ep_lengths)-1:03d}.mp4"
                ok = _write_video(frames, out_path, fps=int(record_fps))
                if not ok:
                    # If writing fails, ignore silently --> we don't want eval to crash training
                    pass
            except Exception:
                pass
    rate = successes / max(1, int(episodes))
    mean_min = float(np.nanmean(mins)) if mins else float("nan")
    mean_len = float(np.mean(ep_lengths)) if ep_lengths else float("nan")
    task = get_active_task()
    return {
        "success_rate": float(rate),
        str(task.eval_mean_min_key): mean_min,
        "mean_episode_length": mean_len,
    }


def evaluate_policy_on_env_parallel(
    env_eval,
    agent: SACAgent,
    episodes: int,
    horizon: int,
    success_thresh: float,
    record_dir: Path | None = None,
    record_fps: int = 15,
    goal_abs: float = 0.0,
    disable_video_capture: bool = False,
) -> dict:
    """Evaluate policy by running episodes in parallel across env instances.

    Implementation notes
    - Uses the provided envs (no extra env spawned), same as serial eval.
    - Runs up to `episodes` episodes by batching them across the env's vector dimension.
      If the env has N instances, one batch can evaluate up to N episodes concurrently.
    - For simplicity and robustness, video capture (if enabled) records env0 only.
    """
    episodes = max(1, int(episodes))
    successes = 0
    mins: list[float] = []
    ep_lengths: list[int] = []

    # Determine vector width from a reset observation
    obs, _ = env_eval.reset()
    try:
        if isinstance(obs, dict):
            any_v = next(iter(obs.values()))
            n_envs = int(np.asarray(_to_np(any_v)).shape[0]) if np.asarray(_to_np(any_v)).ndim >= 2 else 1
        else:
            n_envs = int(np.asarray(_to_np(obs)).shape[0]) if np.asarray(_to_np(obs)).ndim >= 2 else 1
    except Exception:
        n_envs = 1
    n_envs = max(1, int(n_envs))

    remaining = int(episodes)
    batch_idx = 0
    while remaining > 0:
        batch_episodes = min(int(remaining), int(n_envs))
        # Start fresh batch
        obs, _ = env_eval.reset()
        obs_batch = flatten_obs_batch(obs, n_envs)

        done_mask = np.zeros((batch_episodes,), dtype=bool)
        min_dist = np.full((batch_episodes,), float("inf"), dtype=np.float32)
        lengths = np.zeros((batch_episodes,), dtype=np.int32)

        frames0: list[np.ndarray] = []

        try:
            sample = np.asarray(env_eval.action_space.sample())
            if sample.ndim >= 2:
                act_dim = int(sample.shape[-1])
            elif sample.ndim == 1:
                act_dim = int(sample.shape[0])
            else:
                act_dim = int(sample.reshape(-1).shape[0])
        except Exception:
            try:
                shp = tuple(getattr(env_eval.action_space, "shape", ()))
                act_dim = int(shp[-1]) if len(shp) >= 1 else 0
            except Exception:
                act_dim = 0
        if act_dim <= 0:
            raise RuntimeError("Could not infer action dimension for eval.")

        for step_idx in range(int(horizon)):
            with eval_mode(agent):
                a0 = agent.act(obs_batch[0], sample=False)
                a0 = np.asarray(a0, dtype=np.float32).reshape(-1)
                action_mat = np.tile(a0.reshape(1, -1), (n_envs, 1)).astype(np.float32, copy=False)
                for i in range(batch_episodes):
                    if done_mask[i]:
                        continue
                    ai = np.asarray(agent.act(obs_batch[i], sample=False), dtype=np.float32).reshape(-1)
                    action_mat[i] = ai

            action_env = torch.as_tensor(action_mat, dtype=torch.float32)
            try:
                action_env = action_env.to(getattr(env_eval, "device", action_env.device))
            except Exception:
                pass
            if action_env.ndim != 2 or action_env.shape[0] != n_envs or action_env.shape[1] != act_dim:
                raise RuntimeError(
                    f"Invalid action_env shape in eval: got {tuple(action_env.shape)}, expected ({n_envs}, {act_dim})"
                )

            step_out = env_eval.step(action_env)
            if len(step_out) == 5:
                next_obs, _, terminated, truncated, _ = step_out
                done_vec = np.asarray(_to_np(terminated)) | np.asarray(_to_np(truncated))
            else:
                next_obs, _, done, _ = step_out
                done_vec = np.asarray(_to_np(done))

            obs_batch = flatten_obs_batch(next_obs, n_envs)

            # Optional env0-only recording
            if (not bool(disable_video_capture)) and record_dir is not None:
                try:
                    frame = capture_vlm_image(env_eval, save_path=None, step=step_idx, hide_robot_flag=False)
                except Exception:
                    frame = None
                if frame is None:
                    try:
                        frame = env_eval.render()
                        if isinstance(frame, dict):
                            frame = frame.get("rgb_array", None)
                    except Exception:
                        frame = None
                if frame is not None and isinstance(frame, np.ndarray) and frame.ndim == 3:
                    fr = frame if frame.dtype == np.uint8 else np.clip(frame, 0, 255).astype(np.uint8)
                    frames0.append(fr)

            abs_vec = get_scalar_abs_vec_from_scene(env_eval, batch_episodes)
            if abs_vec is not None and abs_vec.size >= int(batch_episodes):
                dist_vec = np.abs(abs_vec[:batch_episodes].astype(np.float32) - float(goal_abs))
                active_mask = ~done_mask
                if np.any(active_mask):
                    min_dist[active_mask] = np.minimum(min_dist[active_mask], dist_vec[active_mask])
            for i in range(batch_episodes):
                if not done_mask[i]:
                    lengths[i] += 1
            try:
                done_sub = np.asarray(done_vec).reshape(-1)[:batch_episodes]
            except Exception:
                done_sub = np.zeros((batch_episodes,), dtype=bool)
            done_mask = done_mask | done_sub.astype(bool)

            if bool(np.all(done_mask)):
                break

        for i in range(batch_episodes):
            md = float(min_dist[i]) if np.isfinite(min_dist[i]) else float("nan")
            mins.append(md)
            ep_lengths.append(int(lengths[i]))
            if np.isfinite(min_dist[i]) and float(min_dist[i]) <= float(success_thresh):
                successes += 1

        # Save env0 video for this batch if requested
        if (not bool(disable_video_capture)) and record_dir is not None and frames0:
            try:
                record_dir.mkdir(parents=True, exist_ok=True)
                out_path = record_dir / f"batch_{batch_idx:03d}_env0.mp4"
                _write_video(frames0, out_path, fps=int(record_fps))
            except Exception:
                pass

        remaining -= int(batch_episodes)
        batch_idx += 1

    rate = successes / float(max(1, int(episodes)))
    mean_min = float(np.nanmean(mins)) if mins else float("nan")
    mean_len = float(np.mean(ep_lengths)) if ep_lengths else float("nan")
    task = get_active_task()
    return {
        "success_rate": float(rate),
        str(task.eval_mean_min_key): mean_min,
        "mean_episode_length": mean_len,
    }

