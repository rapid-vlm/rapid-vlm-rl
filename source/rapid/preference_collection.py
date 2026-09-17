"""Image/state collection for offline VLM preference benchmarks."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from rapid.isaaclab_utils import _ensure_hwc_uint8, get_scalar_abs_vec_from_scene


def _to_numpy(value: Any) -> np.ndarray:
    try:
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        if hasattr(value, "cpu"):
            return value.cpu().numpy()
    except Exception:
        pass
    return np.asarray(value)


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except Exception:
        pass
    return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _root_state(env, entity_name: str, num_envs: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        base_env = getattr(env, "unwrapped", env)
        entity = base_env.scene[str(entity_name)]
        pos = _to_numpy(entity.data.root_pos_w)[: int(num_envs)]
        quat = _to_numpy(entity.data.root_quat_w)[: int(num_envs)]
        return pos.astype(np.float32, copy=False), quat.astype(np.float32, copy=False)
    except Exception:
        return None, None


def _quat_apply_inverse_wxyz(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float32)
    v = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.maximum(norm, 1e-8)
    qw = q[:, 0:1]
    qxyz = -q[:, 1:4]
    t = 2.0 * np.cross(qxyz, v)
    return v + qw * t + np.cross(qxyz, t)


class PreferenceEvalPoolWriter:
    """Write benchmark images plus state metadata during training."""

    def __init__(
        self,
        *,
        enabled: bool,
        log_dir: Path,
        output_dir: str,
        task_name: str,
        image_hw: int,
        goal_abs: float,
        success_thresh: float,
        interval: int,
        max_states: int,
        skip_done: bool,
    ) -> None:
        self.enabled = bool(enabled)
        self.task_name = str(task_name)
        self.image_hw = int(image_hw)
        self.goal_abs = float(goal_abs)
        self.success_thresh = float(success_thresh)
        self.interval = int(max(1, interval))
        self.max_states = int(max(0, max_states))
        self.skip_done = bool(skip_done)
        self.saved_states = 0
        self.next_env_steps = self.interval
        self._warned_no_images = False

        base_dir = Path(output_dir).expanduser() if str(output_dir).strip() else Path(log_dir) / "preference_eval_pool"
        self.output_dir = base_dir
        self.images_dir = self.output_dir / "images"
        self.metadata_path = self.output_dir / "metadata.jsonl"
        self._metadata_file = None

    @classmethod
    def from_args(
        cls,
        args,
        *,
        log_dir: Path,
        task_name: str,
        image_hw: int,
        goal_abs: float,
    ) -> "PreferenceEvalPoolWriter":
        return cls(
            enabled=bool(getattr(args, "save_preference_eval_pool", False)),
            log_dir=Path(log_dir),
            output_dir=str(getattr(args, "preference_eval_pool_dir", "") or ""),
            task_name=str(task_name),
            image_hw=int(image_hw),
            goal_abs=float(goal_abs),
            success_thresh=float(getattr(args, "success_thresh", 0.0)),
            interval=int(getattr(args, "preference_eval_pool_interval", 48550) or 48550),
            max_states=int(getattr(args, "preference_eval_pool_max_states", 0) or 0),
            skip_done=not bool(getattr(args, "preference_eval_pool_keep_done", False)),
        )

    def announce(self) -> None:
        if not self.enabled:
            return
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_file = self.metadata_path.open("a", buffering=1)
        cap_msg = "no cap" if self.max_states <= 0 else f"max_states={self.max_states}"
        print(
            f"[PREF-EVAL] Saving benchmark states every {self.interval} env_steps "
            f"to {self.output_dir} ({cap_msg})."
        )

    def maybe_collect(
        self,
        *,
        env,
        args,
        images: list[np.ndarray] | None,
        env_steps: int,
        loop_step: int,
        seed: int,
        episode: int,
        episode_step: int,
        active_envs: int,
        done_vec: np.ndarray | None = None,
    ) -> None:
        if not self.enabled:
            return
        if self.max_states > 0 and self.saved_states >= self.max_states:
            return
        if int(env_steps) < int(self.next_env_steps):
            return

        if not images:
            if not self._warned_no_images:
                self._warned_no_images = True
                print("[PREF-EVAL] No per-env images available; skipping benchmark state save.")
            return

        n_envs = int(active_envs)
        n_save = min(n_envs, len(images))
        if self.max_states > 0:
            n_save = min(n_save, self.max_states - self.saved_states)
        if n_save <= 0:
            return

        scalar_vec = get_scalar_abs_vec_from_scene(env, n_envs)
        if scalar_vec is None:
            scalar_vec = np.full((n_envs,), np.nan, dtype=np.float32)
        else:
            scalar_vec = np.asarray(scalar_vec, dtype=np.float32).reshape(-1)

        done_arr = None
        if done_vec is not None:
            done_arr = np.asarray(done_vec).reshape(-1)

        env_indices = list(range(n_save))
        if self.skip_done and done_arr is not None:
            env_indices = [
                idx for idx in env_indices
                if idx < done_arr.size and not bool(done_arr[idx])
            ]
            if not env_indices:
                return

        while int(env_steps) >= int(self.next_env_steps):
            self.next_env_steps += self.interval

        task_rows = self._task_rows(env, args, n_envs)

        try:
            from PIL import Image
        except Exception as exc:
            print(f"[PREF-EVAL] PIL is unavailable; cannot save benchmark images: {exc}")
            return

        saved_this_round = 0
        for env_idx in env_indices:
            img_u8 = _ensure_hwc_uint8(images[env_idx], self.image_hw)
            if img_u8 is None:
                continue

            file_name = (
                f"{self.task_name}_step{int(env_steps):08d}_env{env_idx:03d}_"
                f"{self.saved_states:06d}.png"
            )
            image_path = self.images_dir / file_name
            Image.fromarray(img_u8).save(image_path)

            scalar_abs = _finite_float(scalar_vec[env_idx]) if env_idx < scalar_vec.size else None
            dist_to_goal = (
                abs(float(scalar_abs) - self.goal_abs)
                if scalar_abs is not None
                else None
            )
            success_collected = (
                bool(dist_to_goal <= self.success_thresh)
                if dist_to_goal is not None
                else None
            )

            record: dict[str, Any] = {
                "task": self.task_name,
                "image": str(image_path.relative_to(self.output_dir)),
                "env_steps": int(env_steps),
                "loop_step": int(loop_step),
                "seed": int(seed),
                "episode": int(episode),
                "episode_step": int(episode_step),
                "env_index": int(env_idx),
                "scalar_abs": scalar_abs,
                "goal_abs": self.goal_abs,
                "dist_to_goal": dist_to_goal,
                "success_thresh": self.success_thresh,
                "success_collected": success_collected,
                "done": bool(done_arr[env_idx]) if done_arr is not None and env_idx < done_arr.size else None,
            }
            if env_idx < len(task_rows):
                record.update(task_rows[env_idx])
                if "sweep_success" in record:
                    record["success_collected"] = bool(record["sweep_success"])

            self._write_record(record)
            self.saved_states += 1
            saved_this_round += 1

        if saved_this_round > 0:
            print(f"[PREF-EVAL] Saved {saved_this_round} benchmark states at env_steps={env_steps}.")

    def _write_record(self, record: dict[str, Any]) -> None:
        if self._metadata_file is None:
            self.images_dir.mkdir(parents=True, exist_ok=True)
            self._metadata_file = self.metadata_path.open("a", buffering=1)
        self._metadata_file.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")

    def _task_rows(self, env, args, num_envs: int) -> list[dict[str, Any]]:
        if self.task_name == "soccer":
            return self._soccer_rows(env, args, num_envs)
        if self.task_name == "sweep_into":
            return self._sweep_into_rows(env, args, num_envs)
        return [{} for _ in range(int(num_envs))]

    def _soccer_rows(self, env, args, num_envs: int) -> list[dict[str, Any]]:
        ball_pos, ball_quat = _root_state(env, "ball", num_envs)
        goal_pos, goal_quat = _root_state(env, "goal", num_envs)
        rows: list[dict[str, Any]] = []
        half = list(getattr(args, "goal_half_extents", (0.055, 0.095, 0.075)))
        offset = list(getattr(args, "goal_offset", (0.0, 0.0, 0.0)))
        for idx in range(int(num_envs)):
            row: dict[str, Any] = {
                "goal_half_extents": [float(x) for x in half],
                "goal_offset": [float(x) for x in offset],
            }
            if ball_pos is not None and idx < ball_pos.shape[0]:
                row.update({
                    "ball_x": float(ball_pos[idx, 0]),
                    "ball_y": float(ball_pos[idx, 1]),
                    "ball_z": float(ball_pos[idx, 2]),
                })
            if ball_quat is not None and idx < ball_quat.shape[0]:
                row.update({
                    "ball_qw": float(ball_quat[idx, 0]),
                    "ball_qx": float(ball_quat[idx, 1]),
                    "ball_qy": float(ball_quat[idx, 2]),
                    "ball_qz": float(ball_quat[idx, 3]),
                })
            if goal_pos is not None and idx < goal_pos.shape[0]:
                row.update({
                    "goal_x": float(goal_pos[idx, 0]),
                    "goal_y": float(goal_pos[idx, 1]),
                    "goal_z": float(goal_pos[idx, 2]),
                })
            if goal_quat is not None and idx < goal_quat.shape[0]:
                row.update({
                    "goal_qw": float(goal_quat[idx, 0]),
                    "goal_qx": float(goal_quat[idx, 1]),
                    "goal_qy": float(goal_quat[idx, 2]),
                    "goal_qz": float(goal_quat[idx, 3]),
                })
            rows.append(row)
        return rows

    def _sweep_into_rows(self, env, args, num_envs: int) -> list[dict[str, Any]]:
        cube_pos, cube_quat = _root_state(env, "cube", num_envs)
        table_pos, table_quat = _root_state(env, "table_with_hole", num_envs)

        hole_center_x = float(getattr(args, "hole_center_local_x", 0.0))
        hole_center_y = float(getattr(args, "hole_center_local_y", 0.24))
        half_x = float(getattr(args, "hole_half_extent_x", 0.08))
        half_y = float(getattr(args, "hole_half_extent_y", 0.08))
        z_threshold = float(getattr(args, "hole_z_threshold", 0.967))
        success_mode = str(getattr(args, "sweep_success_mode", "inside_xy"))

        cube_table = None
        if cube_pos is not None and table_pos is not None and table_quat is not None:
            rel = cube_pos[: int(num_envs)] - table_pos[: int(num_envs)]
            cube_table = _quat_apply_inverse_wxyz(table_quat[: int(num_envs)], rel)

        rows: list[dict[str, Any]] = []
        for idx in range(int(num_envs)):
            row: dict[str, Any] = {
                "hole_center_local_x": hole_center_x,
                "hole_center_local_y": hole_center_y,
                "hole_half_extent_x": half_x,
                "hole_half_extent_y": half_y,
                "hole_z_threshold": z_threshold,
            }
            if cube_pos is not None and idx < cube_pos.shape[0]:
                row.update({
                    "cube_x": float(cube_pos[idx, 0]),
                    "cube_y": float(cube_pos[idx, 1]),
                    "cube_z": float(cube_pos[idx, 2]),
                })
            if cube_quat is not None and idx < cube_quat.shape[0]:
                row.update({
                    "cube_qw": float(cube_quat[idx, 0]),
                    "cube_qx": float(cube_quat[idx, 1]),
                    "cube_qy": float(cube_quat[idx, 2]),
                    "cube_qz": float(cube_quat[idx, 3]),
                })
            if table_pos is not None and idx < table_pos.shape[0]:
                row.update({
                    "table_x": float(table_pos[idx, 0]),
                    "table_y": float(table_pos[idx, 1]),
                    "table_z": float(table_pos[idx, 2]),
                })
            if table_quat is not None and idx < table_quat.shape[0]:
                row.update({
                    "table_qw": float(table_quat[idx, 0]),
                    "table_qx": float(table_quat[idx, 1]),
                    "table_qy": float(table_quat[idx, 2]),
                    "table_qz": float(table_quat[idx, 3]),
                })
            if cube_table is not None and idx < cube_table.shape[0]:
                local_x = float(cube_table[idx, 0])
                local_y = float(cube_table[idx, 1])
                local_z = float(cube_table[idx, 2])
                dx = abs(local_x - hole_center_x) - half_x
                dy = abs(local_y - hole_center_y) - half_y
                outside_x = max(dx, 0.0)
                outside_y = max(dy, 0.0)
                dist_region = math.sqrt(outside_x * outside_x + outside_y * outside_y)
                inside_xy = dx <= 0.0 and dy <= 0.0
                cube_z_world = (
                    float(cube_pos[idx, 2])
                    if cube_pos is not None and idx < cube_pos.shape[0]
                    else local_z
                )
                success_xy = bool(inside_xy)
                success_xy_and_z = bool(inside_xy and cube_z_world < z_threshold)
                row.update({
                    "cube_table_x": local_x,
                    "cube_table_y": local_y,
                    "cube_table_z": local_z,
                    "sweep_dist_to_hole_region": dist_region,
                    "sweep_inside_hole_xy": bool(inside_xy),
                    "sweep_success_xy": success_xy,
                    "sweep_success_xy_and_z": success_xy_and_z,
                    "sweep_success_mode": success_mode,
                    "sweep_success": success_xy_and_z if success_mode == "inside_xy_and_z" else success_xy,
                })
            rows.append(row)
        return rows
