"""RAPID environment and observation utilities for IsaacLab tasks."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

class IsaacLabTaskSpec:
    """Task spec for the shared RAPID runner.

    This runner assumes task progress/success can be measured by a single scalar value
    from the IsaacLab scene.

    By default, the scalar is read as an absolute joint position (for articulation-based
    tasks like drawer/button). Tasks may override this by implementing
    :meth:`get_scalar_abs_from_scene` / :meth:`get_scalar_abs_vec_from_scene`.
    """

    name: str
    description: str

    gym_id: str
    cfg_factory: Any
    env_spacing: float

    # Scalar joint definitions
    scene_entity: str
    joint_name: str
    obs_term_name: str
    fallback_obs_index: int | None

    # Defaults + logging
    default_exp_name: str
    train_min_log_key: str
    eval_mean_min_key: str
    eval_print_label: str
    best_min_ckpt_prefix: str

    def add_task_args(self, parser: argparse.ArgumentParser) -> None:
        raise NotImplementedError

    def get_goal_abs(self, args: argparse.Namespace) -> float:
        raise NotImplementedError

    def get_init_joint_abs(self, args: argparse.Namespace) -> float | None:
        return None

    def get_vlm_env_name(self, args: argparse.Namespace) -> str:
        raise NotImplementedError

    def post_parse_args(self, args: argparse.Namespace) -> None:
        """Called once after argument parsing. Override to mutate args in-place
        (e.g. unit conversions). Default is a no-op."""
        pass

    # Optional overrides for non-articulation tasks
    def get_scalar_abs_from_scene(self, env) -> float | None:
        """Return a per-env0 progress scalar (absolute, lower-is-better distance-to-goal).

        If None, the runner falls back to reading the configured scalar joint position.
        """
        return None

    def get_scalar_abs_vec_from_scene(self, env, num_envs: int) -> np.ndarray | None:
        """Return progress scalar for multiple envs as shape (num_envs,)."""
        return None

    def get_scalar_default_abs_from_scene(self, env) -> float | None:
        """Optional default absolute scalar used when the policy observation is relative."""
        return None


_ACTIVE_TASK: IsaacLabTaskSpec | None = None


def set_active_task(task: IsaacLabTaskSpec) -> None:
    global _ACTIVE_TASK
    _ACTIVE_TASK = task


def get_active_task() -> IsaacLabTaskSpec:
    if _ACTIVE_TASK is None:
        raise RuntimeError("Active task is not set. Call main(task=...) or set_active_task().")
    return _ACTIVE_TASK


def seed_everywhere(seed: int):
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class eval_mode:
    def __init__(self, *models):
        self.models = models
        self.prev = []

    def __enter__(self):
        self.prev = []
        for m in self.models:
            self.prev.append(getattr(m, "training", False))
            m.train(False)
        return self

    def __exit__(self, exc_type, exc, tb):
        for m, state in zip(self.models, self.prev):
            m.train(state)
        return False


def flatten_obs(obs) -> np.ndarray:
    """Convert observation to a 1D float32 vector."""
    def to_np(a) -> np.ndarray:
        try:
            import torch as _torch  # local import to avoid top-level if not installed
        except Exception:
            _torch = None  # type: ignore
        if _torch is not None and isinstance(a, _torch.Tensor):
            return a.detach().cpu().numpy()
        return np.asarray(a)

    def flatten_any(x) -> list[np.ndarray]:
        if isinstance(x, dict):
            out: list[np.ndarray] = []
            for k in sorted(x.keys()):
                out.extend(flatten_any(x[k]))
            return out
        else:
            arr = to_np(x)
            try:
                arr = arr.astype(np.float32, copy=False).reshape(-1)
            except Exception:
                arr = np.asarray(arr, dtype=np.float32).reshape(-1)
            return [arr]

    parts = flatten_any(obs)
    if len(parts) == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)


def _to_np(a):
    try:
        import torch as _torch
        if isinstance(a, _torch.Tensor):
            return a.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(a)


def flatten_obs_batch(obs, num_envs: int) -> np.ndarray:
    """Return flattened observations for vectorized envs."""
    if num_envs <= 1:
        return flatten_obs(obs).reshape(1, -1)

    if isinstance(obs, dict):
        parts_per_term = []
        for k in sorted(obs.keys()):
            v = _to_np(obs[k])
            if v.ndim == 1:
                v = v.reshape(num_envs, -1)
            else:
                v = v.reshape(num_envs, -1)
            parts_per_term.append(v.astype(np.float32, copy=False))
        return np.concatenate(parts_per_term, axis=1).astype(np.float32, copy=False)

    arr = _to_np(obs)
    if arr.ndim == 1:
        return arr.astype(np.float32, copy=False).reshape(1, -1)
    return arr.astype(np.float32, copy=False).reshape(num_envs, -1)


def get_scalar_slice_from_env(env) -> slice | None:
    """Compute the slice in the concatenated policy observation vector for the active task term."""
    try:
        task = get_active_task()
        base_env = getattr(env, "unwrapped", env)
        obs_man = getattr(base_env, "observation_manager", None)
        if obs_man is None:
            return None
        if not bool(obs_man.group_obs_concatenate.get("policy", True)):
            return None
        names = list(obs_man._group_obs_term_names["policy"])  # type: ignore[attr-defined]
        dims = list(obs_man._group_obs_term_dim["policy"])  # type: ignore[attr-defined]
        offset = 0
        for name, shape in zip(names, dims):
            length = int(np.prod(tuple(shape)))
            if str(name) == str(task.obs_term_name):
                return slice(offset, offset + length)
            offset += length
    except Exception:
        return None
    return None


def extract_scalar_rel_from_obs(obs_vec: np.ndarray, obs_slice: slice | None) -> float | None:
    """Extract the scalar joint value from a flattened observation using slice/fallback."""
    try:
        task = get_active_task()
        if obs_vec is None:
            return None
        if obs_vec.ndim != 1:
            obs_vec = obs_vec.reshape(-1)
        if isinstance(obs_slice, slice) and obs_slice.start is not None and obs_slice.stop is not None:
            if 0 <= obs_slice.start < obs_vec.size and obs_slice.stop <= obs_vec.size:
                seg = obs_vec[obs_slice]
                return float(np.asarray(seg).reshape(-1)[0])
        if task.fallback_obs_index is not None and obs_vec.size > int(task.fallback_obs_index):
            return float(obs_vec[int(task.fallback_obs_index)])
    except Exception:
        pass
    return None


def _get_scalar_joint_id(env) -> int | None:
    try:
        task = get_active_task()
        base_env = getattr(env, "unwrapped", env)
        entity = base_env.scene[task.scene_entity]
        ids, _ = entity.find_joints([task.joint_name])  # type: ignore[attr-defined]
        if isinstance(ids, (list, tuple)) and len(ids) > 0:
            return int(ids[0])
        import torch as _torch
        if isinstance(ids, _torch.Tensor) and ids.numel() > 0:
            return int(ids.flatten()[0].item())
    except Exception:
        return None
    return None


def get_scalar_abs_from_scene(env) -> float | None:
    """Read the absolute scalar joint position from the scene for env0."""
    try:
        task = get_active_task()
        # Task override (e.g., rigid object distance metrics)
        try:
            v = task.get_scalar_abs_from_scene(env)
            if v is not None and np.isfinite(float(v)):
                return float(v)
        except Exception:
            pass
        base_env = getattr(env, "unwrapped", env)
        jid = _get_scalar_joint_id(base_env)
        if jid is None:
            return None
        entity = base_env.scene[task.scene_entity]
        val = entity.data.joint_pos[0, jid]
        if hasattr(val, "item"):
            return float(val.item())
        return float(val)
    except Exception:
        return None


def get_scalar_abs_vec_from_scene(env, num_envs: int) -> np.ndarray | None:
    """Read the absolute scalar joint position from the scene for multiple envs.

    Returns an array of shape (num_envs,) on success, else None.
    """
    try:
        task = get_active_task()
        # Task override
        try:
            v = task.get_scalar_abs_vec_from_scene(env, int(num_envs))
            if v is not None:
                arr = np.asarray(v, dtype=np.float32).reshape(-1)
                if arr.size >= int(num_envs):
                    return arr[: int(num_envs)]
        except Exception:
            pass
        base_env = getattr(env, "unwrapped", env)
        jid = _get_scalar_joint_id(base_env)
        if jid is None:
            return None
        entity = base_env.scene[task.scene_entity]
        import torch as _torch
        vals = entity.data.joint_pos[: int(num_envs), jid]
        if isinstance(vals, _torch.Tensor):
            return vals.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
        return np.asarray(vals, dtype=np.float32).reshape(-1)
    except Exception:
        return None


def get_scalar_default_abs(env) -> float | None:
    """Read the default absolute scalar joint position from the scene for env0."""
    try:
        task = get_active_task()
        # Task override
        try:
            v = task.get_scalar_default_abs_from_scene(env)
            if v is not None and np.isfinite(float(v)):
                return float(v)
        except Exception:
            pass
        base_env = getattr(env, "unwrapped", env)
        jid = _get_scalar_joint_id(base_env)
        if jid is None:
            return None
        entity = base_env.scene[task.scene_entity]
        val = entity.data.default_joint_pos[0, jid]
        if hasattr(val, "item"):
            return float(val.item())
        return float(val)
    except Exception:
        return None


def _split_tiled_rgb(
    img: np.ndarray,
    num_envs: int,
    *,
    rows: int | None = None,
    cols: int | None = None,
) -> list[np.ndarray] | None:
    """
    Split a tiled camera RGB image into per-env RGB frames.
    Returns a list of ``num_envs`` images (H,W,3) uint8-like or None if splitting fails.
    """
    try:
        if img is None:
            return None
        arr = np.asarray(img)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            return None
        n = int(num_envs)
        if n <= 1:
            return [arr]
        if cols is None or int(cols) <= 0:
            cols_i = int(math.ceil(math.sqrt(n)))
        else:
            cols_i = int(cols)
        if rows is None or int(rows) <= 0:
            rows_i = int(math.ceil(float(n) / float(cols_i)))
        else:
            rows_i = int(rows)
        h, w = int(arr.shape[0]), int(arr.shape[1])
        if rows_i <= 0 or cols_i <= 0:
            return None
        if h % rows_i != 0 or w % cols_i != 0:
            return None
        th = h // rows_i
        tw = w // cols_i
        tiles: list[np.ndarray] = []
        for i in range(n):
            r = int(i // cols_i)
            c = int(i % cols_i)
            if r >= rows_i:
                return None
            tiles.append(arr[r * th : (r + 1) * th, c * tw : (c + 1) * tw, :])
        return tiles
    except Exception:
        return None
    

def steps_per_episode(env, env_cfg=None) -> int:
    """Derive episode horizon from env config: steps = episode_length_s / dt / decimation.
    
    Args:
        env: The environment instance
        env_cfg: Optional environment config. If provided, uses it directly. Otherwise tries env.cfg.
    
    Returns:
        Number of steps per episode
    """
    try:
        # Prefer explicitly passed config
        if env_cfg is None:
            env_cfg = getattr(env, "cfg", None)
        
        if env_cfg is None:
            print("[WARN] steps_per_episode: env.cfg is None, using fallback 480")
            return 480
        
        sim_cfg = getattr(env_cfg, "sim", None)
        episode_length_s = getattr(env_cfg, "episode_length_s", 8.0)
        dt = getattr(sim_cfg, "dt", 1 / 60) if sim_cfg is not None else 1 / 60
        decimation = getattr(env_cfg, "decimation", 1)
        
        steps = int(episode_length_s / dt / max(1, decimation))
        return steps
    except Exception as e:
        print(f"[WARN] steps_per_episode failed: {e}, using fallback 480")
        return 480


def _write_video(frames: list[np.ndarray], out_path: Path, fps: int = 15) -> bool:
    """Write frames (H,W,3 uint8) to an MP4. Returns True on success.

    Tries imageio (with ffmpeg plugin) first, then OpenCV as a fallback.
    """
    try:
        import imageio
        try:
            imageio.mimsave(out_path, frames, fps=fps)  # type: ignore[arg-type]
            return True
        except TypeError:
            # Some imageio versions require get_writer for mp4
            writer = imageio.get_writer(out_path, fps=fps)  # type: ignore[attr-defined]
            for fr in frames:
                writer.append_data(fr)
            writer.close()
            return True
    except Exception:
        pass
    # Fallback: OpenCV
    try:
        import cv2
        h, w, _ = frames[0].shape
        fourcc = 0
        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        except Exception:
            pass
        vw = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))
        for fr in frames:
            vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        vw.release()
        return True
    except Exception:
        return False


def hide_show_robot(env, make_invisible: bool, env_index: int = 0):
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


def capture_vlm_image(
    env,
    save_path: Path | None,
    step: int,
    hide_robot_flag: bool,
    tiled_rows: int | None = None,
    tiled_cols: int | None = None,
) -> np.ndarray | None:
    """Capture an RGB image for VLM.

    Behavior:
    - If a `TiledCamera` is present, returns a tiled image of shape (rows*H, cols*W, 3).
      This is required for splitting into per-env tiles when `num_envs>1`.
    - Otherwise, returns a single env0 RGB frame of shape (H, W, 3).
    """
    try:
        base_env = getattr(env, "unwrapped", env)
        scene = getattr(base_env, "scene", None)
        if scene is None or not hasattr(scene, "sensors"):
            return None
        sensor_name = "tiled_camera" if "tiled_camera" in scene.sensors else (
            "overhead_cam" if "overhead_cam" in scene.sensors else None
        )
        if sensor_name is None:
            return None
        cam = scene.sensors[sensor_name]

        prev_token = None
        if hide_robot_flag and sensor_name != "tiled_camera":
            prev_token = hide_show_robot(env, True, env_index=0)

        try:
            outputs = getattr(cam.data, "output", {})
            if (not outputs) or ("rgb" not in outputs and "rgba" not in outputs):
                if hasattr(cam, "_ALL_INDICES") and hasattr(cam, "_update_buffers_impl"):
                    cam._update_buffers_impl(cam._ALL_INDICES)  # type: ignore[attr-defined]
                    outputs = getattr(cam.data, "output", {})

            key = "rgb" if "rgb" in outputs else ("rgba" if "rgba" in outputs else None)
            if key is None:
                return None

            images = outputs[key]

            # TiledCamera stores per-env images as (N,H,W,C)

            if sensor_name == "tiled_camera":
                if hasattr(images, "detach"):
                    images_t = images
                else:
                    images_t = None

                if images_t is None:
                    if hasattr(images, "cpu"):
                        images_np = images.cpu().numpy()
                    else:
                        images_np = np.asarray(images)
                else:
                    images_np = images_t.detach().cpu().numpy()

                if images_np.ndim != 4 or images_np.shape[0] < 1:
                    return None

                # Force RGB
                if images_np.shape[-1] >= 3:
                    images_np = images_np[..., :3]
                else:
                    return None

                num = int(images_np.shape[0])
                tile_h = int(images_np.shape[1])
                tile_w = int(images_np.shape[2])

                if tiled_rows is not None and tiled_cols is not None and int(tiled_rows) > 0 and int(tiled_cols) > 0:
                    rows = int(tiled_rows)
                    cols = int(tiled_cols)
                else:
                    cols = int(math.ceil(math.sqrt(num)))
                    rows = int(math.ceil(num / max(1, cols)))
                atlas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=images_np.dtype)
                for i in range(num):
                    r = i // cols
                    c = i % cols
                    atlas[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w, :] = images_np[i]

                if save_path is not None:
                    try:
                        from PIL import Image

                        Image.fromarray(atlas).save(save_path / f"vlm_step_{step:08d}.png")
                    except Exception:
                        pass
                return atlas

            # Non-tiled camera path: return env0 image
            if hasattr(images, "cpu"):
                images = images.cpu().numpy()
            if getattr(images, "ndim", 0) == 4 and images.shape[0] >= 1:
                img = images[0]
            else:
                img = images
            if img.ndim == 3 and img.shape[-1] >= 3:
                img = img[..., :3]
            else:
                return None
            if save_path is not None:
                try:
                    from PIL import Image
                    Image.fromarray(img).save(save_path / f"vlm_step_{step:08d}.png")
                except Exception:
                    pass
            return img
        finally:
            if prev_token is not None:
                hide_show_robot(env, False, env_index=0)
    except Exception:
        return None


def _ensure_hwc_uint8(img: Any, target_size: int = 128) -> np.ndarray | None:
    """Ensure image is (H, W, 3) uint8 with specified resolution.
    
    Args:
        img: Input image (any format)
        target_size: Target square resolution (default 128 for backward compatibility)
    """
    if img is None:
        return None
    try:
        arr = np.asarray(img)
        if arr.ndim != 3 or arr.shape[-1] < 3:
            return None
        arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)
        if arr.shape[0] == target_size and arr.shape[1] == target_size:
            return arr
        # If this is a tiled atlas, env0 is top-left tile: crop first.
        if arr.shape[0] >= target_size and arr.shape[1] >= target_size:
            crop = arr[:target_size, :target_size, :]
            if crop.shape[0] == target_size and crop.shape[1] == target_size:
                return crop
        # Last resort: resize.
        try:
            from PIL import Image

            resized = np.array(Image.fromarray(arr).resize((target_size, target_size)))
            if resized.ndim == 3 and resized.shape[-1] >= 3:
                return resized[..., :3].astype(np.uint8, copy=False)
        except Exception:
            return None
    except Exception:
        return None
    return None


def _make_env_from_active_task(args: argparse.Namespace):
    task = get_active_task()
    cfg = task.cfg_factory()
    try:
        cfg.scene.num_envs = max(1, int(getattr(args, "num_envs", 1)))
    except Exception:
        pass
    try:
        cfg.scene.env_spacing = float(getattr(task, "env_spacing", 2.5))
    except Exception:
        pass
    # Optional initial scalar joint override
    try:
        init_abs = task.get_init_joint_abs(args)
        if init_abs is not None:
            entity_cfg = getattr(getattr(cfg, "scene"), str(task.scene_entity))
            entity_cfg.init_state.joint_pos[str(task.joint_name)] = float(init_abs)
    except Exception:
        pass
    try:
        if getattr(args, "device", None) is not None:
            cfg.sim.device = getattr(args, "device")
    except Exception:
        pass
    try:
        if getattr(args, "seed", None) is not None:
            cfg.seed = int(getattr(args, "seed"))
    except Exception:
        pass
    env = gym.make(str(task.gym_id), cfg=cfg, render_mode=None)
    return env, cfg


