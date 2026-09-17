"""VLM image-pool management for RAPID training."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from rapid.isaaclab_utils import _ensure_hwc_uint8
from rapid.reward_model import RewardModel


class VLMPoolManager:
    def __init__(
        self,
        *,
        reward_model: RewardModel,
        obs_dim: int,
        act_dim: int,
        image_hw: int,
        num_envs: int,
        log_dir: Path,
        pool_size: int,
        strategy: str,
        feature_res: int,
        feature_type: str,
        encoder_batch_size: int,
        candidate_cap: int,
        save_images: bool,
        refresh_every: int,
        save_snapshots_every: int,
    ) -> None:
        self.reward_model = reward_model
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.image_hw = int(image_hw)
        self.num_envs = int(num_envs)
        self.log_dir = Path(log_dir)
        self.pool_size = int(pool_size)
        self.strategy = str(strategy)
        self.feature_res = int(feature_res)
        self.feature_type = str(feature_type)
        if self.feature_type not in {"pixel", "reward_encoder"}:
            raise ValueError(
                "feature_type must be 'pixel' or 'reward_encoder'; "
                f"received {self.feature_type!r}"
            )
        self.encoder_batch_size = max(1, int(encoder_batch_size))
        self.candidate_cap = int(candidate_cap)
        self.save_images = bool(save_images)
        self.refresh_every = int(refresh_every)
        self.save_snapshots_every = int(save_snapshots_every)
        self.use_multi_env = (self.strategy != "fifo") or (self.num_envs > 1)

        self._pool_candidates: list[np.ndarray] = []
        self._pool_imgs: list[np.ndarray] = []
        self._last_pool_hashes: list[str] = []
        self._last_pool_refresh_step: int | None = None
        self.next_snapshot_step: int | None = (
            self.save_snapshots_every if self.save_snapshots_every > 0 else None
        )
        self._pool_dump_dir = self.log_dir / "vlm_pool" / "current"
        self.last_feature_seconds = 0.0
        self.last_fps_seconds = 0.0
        self.last_feature_dim = 0
        self.last_candidate_count = 0
        self.last_selected_count = 0

    def announce(self) -> None:
        if self.strategy == "representative":
            detail = (
                f"resolution={self.feature_res}x{self.feature_res}"
                if self.feature_type == "pixel"
                else f"ensemble=current, batch_size={self.encoder_batch_size}"
            )
            print(
                "[VLM] Representative FPS enabled: "
                f"feature_type={self.feature_type}, {detail}, "
                f"K={self.pool_size}, M={self.candidate_cap}"
            )

        if self.save_images:
            print(f"[VLM] Will dump current VLM pool images to: {self._pool_dump_dir}")
            if self.strategy == "representative" and self.refresh_every > 0:
                print(
                    f"[VLM] Representative pool refresh during rollout: every {self.refresh_every} env_steps"
                )

        if self.save_snapshots_every > 0:
            print(
                f"[VLM] Will save VLM pool snapshots every {self.save_snapshots_every} env_steps to: "
                f"{self.log_dir / 'vlm_pool' / 'step_XXXXX'}"
            )

    def update(self, new_imgs: list[np.ndarray], step_tag: int | None = None) -> None:
        if not new_imgs:
            return

        cleaned: list[np.ndarray] = []
        for img in new_imgs:
            img_u8 = _ensure_hwc_uint8(img, self.image_hw)
            if img_u8 is not None:
                cleaned.append(img_u8)
        if not cleaned:
            return

        self._pool_candidates.extend(cleaned)
        if len(self._pool_candidates) > self.candidate_cap:
            self._pool_candidates = self._pool_candidates[-self.candidate_cap :]

        if self.strategy == "fifo":
            self._pool_imgs = self._pool_candidates[-self.pool_size :]
            self._dump_current(step_tag=step_tag)
            return

        if not (self.save_images and self.refresh_every > 0 and step_tag is not None):
            return

        try:
            do_refresh = (
                self._last_pool_refresh_step is None
                or (int(step_tag) - int(self._last_pool_refresh_step) >= int(self.refresh_every))
            )
            if do_refresh:
                self._last_pool_refresh_step = int(step_tag)
                if self._pool_candidates:
                    self._pool_imgs = self._select_representative_imgs(
                        self._pool_candidates,
                        self.pool_size,
                        self.feature_res,
                    )
                else:
                    self._pool_imgs = []
                self._dump_current(step_tag=step_tag)
        except Exception:
            pass

    def sync_reward_model_dataset(self, step_tag: int | None = None) -> int:
        if self.strategy == "representative":
            if self._pool_candidates:
                self._pool_imgs = self._select_representative_imgs(
                    self._pool_candidates,
                    self.pool_size,
                    self.feature_res,
                )
            else:
                self._pool_imgs = []
        else:
            self._pool_imgs = self._pool_imgs[-self.pool_size :]

        self._dump_current(step_tag=step_tag)
        if not self._pool_imgs:
            return 0

        sample_dim = int(self.obs_dim + self.act_dim)
        self.reward_model.inputs = [
            np.zeros((1, sample_dim), dtype=np.float32) for _ in range(len(self._pool_imgs))
        ]
        self.reward_model.targets = [
            np.zeros((1, 1), dtype=np.float32) for _ in range(len(self._pool_imgs))
        ]
        self.reward_model.img_inputs = [
            img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
            for img in self._pool_imgs
        ]
        return len(self._pool_imgs)

    def maybe_save_snapshot(self, step_tag: int, reward_learning_active: bool) -> None:
        if self.next_snapshot_step is None:
            return

        if reward_learning_active and step_tag >= int(self.next_snapshot_step):
            try:
                if self.strategy == "representative" and self._pool_candidates:
                    self._pool_imgs = self._select_representative_imgs(
                        self._pool_candidates,
                        self.pool_size,
                        self.feature_res,
                    )
                self._save_snapshot(step_tag)
                self.next_snapshot_step = int(step_tag) + int(self.save_snapshots_every)
            except Exception as exc:
                print(f"[VLM] Error during pool snapshot: {exc}")
        elif not reward_learning_active:
            print(f"[VLM] Stopping pool snapshots after reward freeze at env_steps={step_tag}")
            self.next_snapshot_step = None

    def _dump_current(self, step_tag: int | None = None) -> None:
        if not self.save_images:
            return

        try:
            hashes: list[str] = []
            for img in self._pool_imgs:
                try:
                    img_u8 = _ensure_hwc_uint8(img, self.image_hw)
                    if img_u8 is None:
                        continue
                    hashes.append(hashlib.sha1(img_u8.tobytes()).hexdigest())
                except Exception:
                    continue

            if hashes == self._last_pool_hashes:
                return

            tmp_dir = self._pool_dump_dir.parent / ".tmp_current"
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
            tmp_dir.mkdir(parents=True, exist_ok=True)

            try:
                from PIL import Image

                for idx, img in enumerate(self._pool_imgs):
                    img_u8 = _ensure_hwc_uint8(img, self.image_hw)
                    if img_u8 is None:
                        continue
                    short_hash = hashlib.sha1(img_u8.tobytes()).hexdigest()[:12]
                    Image.fromarray(img_u8).save(tmp_dir / f"pool_{idx:03d}_{short_hash}.png")
            except Exception:
                pass

            manifest = {
                "step": int(step_tag) if step_tag is not None else None,
                "strategy": self.strategy,
                "feature_type": self.feature_type,
                "feature_dim": int(self.last_feature_dim),
                "pool_size": int(len(self._pool_imgs)),
                "hashes": [item[:12] for item in hashes],
            }
            try:
                (tmp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
            except Exception:
                pass

            try:
                shutil.rmtree(self._pool_dump_dir, ignore_errors=True)
            except Exception:
                pass
            try:
                tmp_dir.rename(self._pool_dump_dir)
            except Exception:
                try:
                    shutil.copytree(tmp_dir, self._pool_dump_dir, dirs_exist_ok=True)
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                except Exception:
                    pass

            self._last_pool_hashes = hashes
        except Exception:
            return

    def _save_snapshot(self, step_tag: int) -> None:
        if not self._pool_imgs:
            return
        try:
            snapshot_dir = self.log_dir / "vlm_pool" / f"step_{step_tag:07d}"
            snapshot_dir.mkdir(parents=True, exist_ok=True)

            from PIL import Image

            for idx, img in enumerate(self._pool_imgs):
                img_u8 = _ensure_hwc_uint8(img, self.image_hw)
                if img_u8 is None:
                    continue
                short_hash = hashlib.sha1(img_u8.tobytes()).hexdigest()[:12]
                Image.fromarray(img_u8).save(snapshot_dir / f"pool_{idx:03d}_{short_hash}.png")

            manifest = {
                "step": int(step_tag),
                "strategy": self.strategy,
                "feature_type": self.feature_type,
                "feature_dim": int(self.last_feature_dim),
                "pool_size": int(len(self._pool_imgs)),
                "timestamp": datetime.now().isoformat(),
            }
            (snapshot_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
            print(
                f"[VLM] Saved pool snapshot at step {step_tag} ({len(self._pool_imgs)} images) to {snapshot_dir}"
            )
        except Exception as exc:
            print(f"[VLM] Failed to save pool snapshot at step {step_tag}: {exc}")

    def _select_representative_imgs(
        self,
        imgs: list[np.ndarray],
        pool_size: int,
        feature_res: int,
    ) -> list[np.ndarray]:
        self.last_candidate_count = int(len(imgs))
        self.last_selected_count = int(min(max(pool_size, 0), len(imgs)))
        if pool_size <= 0 or len(imgs) <= pool_size:
            self.last_feature_seconds = 0.0
            self.last_fps_seconds = 0.0
            self.last_feature_dim = 0
            return list(imgs)

        feature_start = time.perf_counter()
        if self.feature_type == "reward_encoder":
            feats = self.reward_model.image_encoder_features(
                imgs,
                batch_size=self.encoder_batch_size,
            )
        else:
            feats = np.stack([self._img_feature(img, feature_res) for img in imgs], axis=0)
        self.last_feature_seconds = float(time.perf_counter() - feature_start)

        feats = np.ascontiguousarray(feats, dtype=np.float32)
        if feats.ndim != 2 or feats.shape[0] != len(imgs) or feats.shape[1] <= 0:
            raise RuntimeError(
                "Representative feature extraction returned an invalid matrix: "
                f"shape={getattr(feats, 'shape', None)}, expected rows={len(imgs)}"
            )
        if not np.isfinite(feats).all():
            raise RuntimeError("Representative features contain NaN or infinity")

        self.last_feature_dim = int(feats.shape[1])
        count = int(feats.shape[0])
        selected = [count // 2]
        fps_start = time.perf_counter()
        if self.feature_type == "pixel":
            # Preserve the exact distance computation used by the published pixel runs.
            dists = np.linalg.norm(feats - feats[selected[0]], axis=1)
            for _ in range(1, int(pool_size)):
                idx = int(np.argmax(dists))
                selected.append(idx)
                dists = np.minimum(dists, np.linalg.norm(feats - feats[idx], axis=1))
        else:
            # The 6,144-D ensemble features benefit from a BLAS-backed squared-distance
            # computation --> squaring does not change the FPS ordering
            feature_norm_sq = np.einsum("ij,ij->i", feats, feats)

            def squared_distances(index: int) -> np.ndarray:
                distances = feature_norm_sq + feature_norm_sq[index] - 2.0 * (feats @ feats[index])
                return np.maximum(distances, 0.0)

            dists = squared_distances(selected[0])
            for _ in range(1, int(pool_size)):
                idx = int(np.argmax(dists))
                selected.append(idx)
                dists = np.minimum(dists, squared_distances(idx))
        self.last_fps_seconds = float(time.perf_counter() - fps_start)

        print(
            "[VLM] Representative FPS selection: "
            f"feature_type={self.feature_type}, candidates={count}, selected={len(selected)}, "
            f"feature_dim={self.last_feature_dim}, feature_seconds={self.last_feature_seconds:.3f}, "
            f"fps_seconds={self.last_fps_seconds:.3f}"
        )
        return [imgs[idx] for idx in selected]

    def _img_feature(self, img_u8: np.ndarray, feature_res: int) -> np.ndarray:
        try:
            from PIL import Image

            image = Image.fromarray(np.asarray(img_u8)).convert("L").resize(
                (feature_res, feature_res),
                getattr(getattr(Image, "Resampling", Image), "BILINEAR"),
            )
            arr = (np.asarray(image).astype(np.float32) / 255.0).reshape(-1)
        except Exception:
            arr = np.asarray(img_u8).astype(np.float32).reshape(-1)

        norm = float(np.linalg.norm(arr) + 1e-8)
        return (arr / norm).astype(np.float32, copy=False)
