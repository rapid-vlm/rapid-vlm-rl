"""VLM-based reward labeling and stability-based freeze logic for RAPID."""
from __future__ import annotations

import math

import numpy as np

from rapid.replay_buffer import ReplayBuffer
from rapid.reward_model import RewardModel


##
# VLM labeling
##

def _iter_chunks(total: int, chunk: int, desc: str, show_progress: bool):
    """Yield chunk indices, optionally wrapped in a tqdm progress bar."""
    n = max(1, math.ceil(total / max(1, chunk)))
    if show_progress:
        try:
            from tqdm.auto import tqdm  # type: ignore
            return tqdm(range(n), total=n, desc=desc, leave=False)
        except Exception:
            pass
    return range(n)


def run_vlm_labeling(reward_model: RewardModel, target: int, chunk: int,
                     desc: str, show_progress: bool) -> int:
    """Run chunked VLM labeling and return the number of accepted pairs."""
    orig_mb = reward_model.mb_size
    chunk = max(1, min(chunk, target))
    acc_sum = 0
    done_labels = 0
    for _ in _iter_chunks(target, chunk, desc, show_progress):
        this_batch = min(chunk, target - done_labels)
        reward_model.set_batch(this_batch)
        acc_sum += reward_model.uniform_sampling()
        done_labels += this_batch
    reward_model.set_batch(orig_mb)
    return int(acc_sum)


def train_reward_epochs(reward_model: RewardModel, n_epochs: int) -> float:
    """Train the reward model for up to *n_epochs*, early-stopping at 97 % accuracy."""
    total_acc = 0.0
    for _ in range(n_epochs):
        reward_model.train()
        if getattr(reward_model, "label_margin", 0) > 0 or getattr(reward_model, "teacher_eps_equal", 0) > 0:
            train_acc = reward_model.train_soft_reward()
        else:
            train_acc = reward_model.train_reward()
        try:
            total_acc = float(np.mean(train_acc))
        except Exception:
            total_acc = 0.0
        if total_acc > 0.97:
            break
    return total_acc


##
# Stability-based freeze
##

def build_stability_anchors(replay_buffer: ReplayBuffer, reward_model: RewardModel,
                            anchor_size: int):
    """Sample anchor images from the replay buffer and compute initial predictions.

    Returns:
        (anchor_inputs, last_anchor_pred) — both ``np.ndarray``, or ``(None, None)``
        on failure.
    """
    if not getattr(replay_buffer, 'store_image', False):
        return None, None
    n = min(anchor_size, int(replay_buffer.idx))
    if n <= 0:
        return None, None
    idxs = np.linspace(0, replay_buffer.idx - 1, num=n, dtype=int)
    imgs = replay_buffer.images[idxs]
    anchor_inputs = np.transpose(imgs, (0, 3, 1, 2)).astype(np.float32) / 255.0
    reward_model.eval()
    last_anchor_pred = np.asarray(reward_model.r_hat_batch(anchor_inputs)).reshape(-1)
    return anchor_inputs, last_anchor_pred


def check_stability_freeze(
    reward_model: RewardModel,
    anchor_inputs: np.ndarray,
    last_anchor_pred: np.ndarray,
    args,
    acc_round: int,
    stability_good_count: int,
    low_accept_count: int,
    env_steps: int,
    step: int,
    logger,
):
    """Check reward prediction stability on the anchor set.

    Returns:
        (should_freeze, new_pred, updated_good_count, updated_low_count)
    """
    reward_model.eval()
    curr_pred = np.asarray(reward_model.r_hat_batch(anchor_inputs)).reshape(-1)

    should_freeze = False
    if last_anchor_pred is not None and curr_pred.size == last_anchor_pred.size:
        corr = float(np.corrcoef(curr_pred, last_anchor_pred)[0, 1]) if curr_pred.size > 1 else 1.0
        delta = float(np.mean(np.abs(curr_pred - last_anchor_pred)))

        logger.log('train/reward_stability_corr', corr, env_steps)
        logger.log('train/reward_stability_delta', delta, env_steps)

        stable = (
            corr >= float(getattr(args, 'stability_corr_thresh', 0.99))
            and delta <= float(getattr(args, 'stability_delta_eps', 0.025))
        )
        stability_good_count = stability_good_count + 1 if stable else 0

        low_accept = int(acc_round) <= int(getattr(args, 'min_vlm_accept_per_round', 5))
        low_accept_count = (low_accept_count + 1) if low_accept else 0

        should_freeze = (
            stability_good_count >= int(getattr(args, 'stability_patience', 2))
            or low_accept_count >= int(getattr(args, 'min_vlm_accept_patience', 3))
        )
        if should_freeze:
            print(
                f"[REWARD] Stability freeze engaged at env_steps={env_steps} "
                f"(loop_step={step}, corr={corr:.4f}, delta={delta:.4f}, "
                f"acc_round={int(acc_round)})."
            )
            logger.log('train/reward_frozen', 1.0, env_steps)
            logger.log('train/reward_freeze_step', float(env_steps), env_steps)

    logger.dump(env_steps)
    return should_freeze, curr_pred, stability_good_count, low_accept_count


##
# Reward inference
##

def infer_reward_hat(reward_model: RewardModel, img: np.ndarray,
                     per_env_imgs: list | None, active_envs: int):
    """Compute learned reward from images.

    Returns:
        (reward_hat0, per_env_reward_hat)   where ``per_env_reward_hat`` is a
        1-D ``np.ndarray`` when batch inference succeeded, else ``None``.
    """
    if active_envs > 1 and per_env_imgs is not None:
        batch = np.stack(
            [np.asarray(f).transpose(2, 0, 1).astype(np.float32) / 255.0
             for f in per_env_imgs],
            axis=0,
        )
        reward_model.eval()
        per_env = np.asarray(reward_model.r_hat_batch(batch)).reshape(-1)
        reward_model.train()
        r0 = float(per_env[0]) if per_env.size > 0 else 0.0
        return r0, per_env

    # Single-image path
    im = np.asarray(img).transpose(2, 0, 1).astype(np.float32) / 255.0
    im = im.reshape(1, 3, im.shape[1], im.shape[2])
    reward_model.eval()
    r0 = float(reward_model.r_hat(im))
    reward_model.train()
    return r0, None
