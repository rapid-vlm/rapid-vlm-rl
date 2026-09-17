#!/usr/bin/env python3
"""
Shared RAPID training runner.
"""
from __future__ import annotations

import math
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher

from rapid.logger import Logger
from rapid.preference_collection import PreferenceEvalPoolWriter
from rapid.vlm_pool import VLMPoolManager

from rapid.isaaclab_utils import IsaacLabTaskSpec  # noqa: F401

from rapid.isaaclab_utils import (
    _ensure_hwc_uint8,
    _make_env_from_active_task,
    _split_tiled_rgb,
    _to_np,
    capture_vlm_image,
    eval_mode,
    extract_scalar_rel_from_obs,
    flatten_obs_batch,
    get_active_task,
    get_scalar_abs_from_scene,
    get_scalar_default_abs,
    get_scalar_slice_from_env,
    hide_show_robot,
    seed_everywhere,
    set_active_task,
    steps_per_episode,
)
from rapid.argparser import build_argparser
from rapid.setup import (
    create_replay_buffer,
    create_reward_model,
    create_sac_agent,
    detect_camera_resolution,
    detect_obs_action_dims,
    load_checkpoint,
)
from rapid.reward_labeling import (
    build_stability_anchors,
    check_stability_freeze,
    infer_reward_hat,
    run_vlm_labeling,
    train_reward_epochs,
)
from train_eval import evaluate_policy_on_env, evaluate_policy_on_env_parallel

_LAST_PARSED_ARGS = None


def _log_vlm_pool_selection(logger: Logger, manager: VLMPoolManager, step: int) -> None:
    """Record the selector configuration and the latest FPS timing."""
    logger.log(
        'train/vlm_pool_reward_encoder',
        1.0 if manager.feature_type == 'reward_encoder' else 0.0,
        step,
    )
    logger.log('train/vlm_pool_feature_dim', float(manager.last_feature_dim), step)
    logger.log('train/vlm_pool_candidate_count', float(manager.last_candidate_count), step)
    logger.log('train/vlm_pool_selected_count', float(manager.last_selected_count), step)
    logger.log('train/vlm_pool_feature_seconds', float(manager.last_feature_seconds), step)
    logger.log('train/vlm_pool_fps_seconds', float(manager.last_fps_seconds), step)

def main(task: IsaacLabTaskSpec | None = None):
    global _LAST_PARSED_ARGS

    if task is not None:
        set_active_task(task)
    parser = build_argparser()
    raw_argv = list(sys.argv[1:])
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        parser.error(f"unrecognized arguments: {' '.join(unknown_args)}")
    _LAST_PARSED_ARGS = args # make available to task wrappers

    if task is not None:
        task.post_parse_args(args)

    # Cameras are needed for VLM images: set via AppLauncher standard flag name if present
    if hasattr(args, "enable_cameras"):
        setattr(args, "enable_cameras", True)
    
    if int(getattr(args, "num_envs", 1)) < 1:
        args.num_envs = 1
    num_envs = int(getattr(args, "num_envs", 1))

    # Prepare Isaac app
    sys.argv = [sys.argv[0]]
    app = AppLauncher(args)
    simulation_app = app.app

    # Respect requested VLM model id
    if getattr(args, 'vlm_model', None):
        os.environ["OPENROUTER_MODEL"] = str(args.vlm_model)

    if not (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        print("[WARN] Missing OPENROUTER_API_KEY. VLM labeling will likely accept 0 pairs.")

    # Seed
    seed_everywhere(args.seed)

    # Log directory
    time_tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(args.log_root) / args.exp_name / time_tag
    (log_dir / "models").mkdir(parents=True, exist_ok=True)
    (log_dir / "tb").mkdir(parents=True, exist_ok=True)
    (log_dir / "command.txt").write_text(" ".join(sys.orig_argv if hasattr(sys, 'orig_argv') else raw_argv))

    
    env, env_cfg = _make_env_from_active_task(args) # build env

    if args.hide_robot: # optionally hide robot for the entire training (visuals only)
        hide_show_robot(env, True, env_index=0)
        print("[INFO] --hide_robot: robot visuals hidden for the whole run.")

    horizon = steps_per_episode(env, env_cfg) # episode horizon

    # Goal
    goal_abs = float(get_active_task().get_goal_abs(args))

    # --- Agent, replay buffer, reward model ---
    obs_dim, obs_shape, act_dim, action_range = detect_obs_action_dims(env, num_envs)
    agent = create_sac_agent(args, obs_dim, act_dim, action_range)
    expected_act_dim = int(act_dim)
    image_hw = detect_camera_resolution(env)
    replay_buffer = create_replay_buffer(args, obs_shape, act_dim, image_hw, agent.device)
    vlm_env_name = str(get_active_task().get_vlm_env_name(args))
    reward_model = create_reward_model(args, obs_dim, act_dim, image_hw, log_dir, vlm_env_name)
    resume_from_step = load_checkpoint(args, agent, reward_model)

    # ----- Multi-env VLM sampling pool (optional) -----
    vlm_pool_manager = VLMPoolManager(
        reward_model=reward_model,
        obs_dim=obs_dim,
        act_dim=act_dim,
        image_hw=image_hw,
        num_envs=num_envs,
        log_dir=log_dir,
        pool_size=int(max(1, int(getattr(args, 'vlm_pool_size', 100) or 100))),
        strategy=str(getattr(args, 'vlm_pool_strategy', 'fifo')),
        feature_res=int(max(8, int(getattr(args, 'vlm_pool_feature_res', 32) or 32))),
        feature_type=str(getattr(args, 'vlm_pool_feature_type', 'pixel')),
        encoder_batch_size=int(max(1, int(getattr(args, 'vlm_pool_encoder_batch_size', 128) or 128))),
        candidate_cap=int(
            max(
                int(max(1, int(getattr(args, 'vlm_pool_size', 100) or 100))),
                int(getattr(args, 'vlm_pool_candidate_cap', 2000) or 2000),
            )
        ),
        save_images=bool(getattr(args, 'save_vlm_pool_images', False)),
        refresh_every=int(max(0, int(getattr(args, 'vlm_pool_refresh_every', 0) or 0))),
        save_snapshots_every=int(max(0, int(getattr(args, 'save_vlm_pool_snapshots_every', 0) or 0))),
    )
    vlm_pool_manager.announce()
    use_multi_env_vlm_pool = vlm_pool_manager.use_multi_env
    preference_eval_writer = PreferenceEvalPoolWriter.from_args(
        args,
        log_dir=log_dir,
        task_name=str(get_active_task().name),
        image_hw=image_hw,
        goal_abs=goal_abs,
    )
    preference_eval_writer.announce()

    # --- Logger and training state ---
    logger = Logger(str(log_dir), save_tb=True,
                    save_csv=getattr(args, 'save_csv', False),
                    log_frequency=1000, agent="sac")
    best_success = -1.0
    best_min_dist = float("inf")
    freeze_after_step_dynamic: int | None = None
    try:
        logger.log('train/reward_frozen', 0.0, 0)
        logger.log(
            'train/vlm_pool_reward_encoder',
            1.0 if vlm_pool_manager.feature_type == 'reward_encoder' else 0.0,
            0,
        )
        logger.log(
            'train/vlm_pool_pixel_feature_res',
            float(vlm_pool_manager.feature_res if vlm_pool_manager.feature_type == 'pixel' else 0),
            0,
        )
        logger.dump(0)
    except Exception:
        pass

    # Reward-stability state
    anchor_inputs: np.ndarray | None = None
    last_anchor_pred: np.ndarray | None = None
    stability_good_count = 0
    low_accept_count = 0
    update_budget = 0.0   # fractional update accumulator

    # --- Training loop ---
    obs, _ = env.reset()
    scalar_obs_slice = get_scalar_slice_from_env(env)
    obs_batch = flatten_obs_batch(obs, num_envs)
    obs = obs_batch[0]

    episode = 0
    episode_reward = 0.0
    true_episode_reward = 0.0
    episode_step = 0
    train_min_dist = float("inf")
    total_feedback = 0
    labeled_feedback = 0
    step = 0
    env_steps = resume_from_step
    last_env_steps_for_update = resume_from_step

    # Config shortcuts
    update_scale_cap = int(getattr(args, 'update_scale_cap', 16))
    update_scale_mode = str(getattr(args, 'update_scale_mode', 'cap_active_envs'))
    debug_tiled_split = bool(getattr(args, 'debug_tiled_split', False))
    printed_tiled_split = False
    last_logged_update_regime = False
    replay_images_dropped = False

    # Scheduling (all in env_steps)
    seed_end_steps = int(max(0, int(getattr(args, 'num_seed_steps', 0) or 0)))
    unsup_end_steps = seed_end_steps + int(max(0, int(getattr(args, 'num_unsup_steps', 0) or 0)))
    initial_label_done = resume_from_step > unsup_end_steps
    base_interact = int(max(1, int(getattr(args, 'num_interact', 1) or 1)))
    if resume_from_step > unsup_end_steps:
        next_label_step = unsup_end_steps + ((((resume_from_step - unsup_end_steps) // base_interact) + 1) * base_interact)
    else:
        next_label_step = unsup_end_steps + base_interact

    if getattr(args, 'eval_every', 0) and args.eval_every > 0:
        base_eval = int(args.eval_every)
        next_eval_step = ((resume_from_step // base_eval) + 1) * base_eval if resume_from_step > 0 else base_eval
    else:
        next_eval_step = None

    if getattr(args, 'save_interval', 0) and args.save_interval > 0:
        base_ckpt = int(args.save_interval)
        next_ckpt_step = ((resume_from_step // base_ckpt) + 1) * base_ckpt if resume_from_step > 0 else base_ckpt
    else:
        next_ckpt_step = None

    def _fix_act(a):
        """Reshape/clip action to (expected_act_dim,)."""
        aa = np.asarray(a, dtype=np.float32).reshape(-1)
        if aa.size < expected_act_dim:
            raise ValueError(f"Action too small: {aa.size} vs {expected_act_dim}")
        return aa[:expected_act_dim] if aa.size > expected_act_dim else aa

    while env_steps < args.total_steps:
        active_envs = num_envs

        # Compute effective freeze boundary
        effective_freeze_after = None
        if int(getattr(args, 'freeze_reward_after', 0)) > 0:
            effective_freeze_after = int(args.freeze_reward_after)
        if freeze_after_step_dynamic is not None:
            effective_freeze_after = (
                min(effective_freeze_after, freeze_after_step_dynamic)
                if effective_freeze_after is not None
                else freeze_after_step_dynamic
            )

        # Drop replay images after freeze to save RAM
        if (
            not replay_images_dropped
            and bool(getattr(args, 'drop_replay_images_after_freeze', False))
            and effective_freeze_after is not None
            and env_steps >= int(effective_freeze_after)
        ):
            replay_buffer.store_image = False
            if hasattr(replay_buffer, 'images'):
                delattr(replay_buffer, 'images')
            replay_images_dropped = True
            print(f"[INFO] Dropped replay buffer images at env_steps={env_steps}.")
            logger.log('train/replay_images_dropped', 1.0, env_steps)

        reward_learning_active = (effective_freeze_after is None) or (env_steps < int(effective_freeze_after))

        # Select actions
        action_mat = np.zeros((num_envs, expected_act_dim), dtype=np.float32)
        if env_steps < seed_end_steps:
            action_mat[:] = _fix_act(env.action_space.sample())
        else:
            with eval_mode(agent):
                if active_envs <= 1:
                    action_mat[:] = _fix_act(agent.act(obs, sample=True))
                else:
                    a0 = _fix_act(agent.act(obs_batch[0], sample=True))
                    action_mat[:] = a0
                    for i in range(active_envs):
                        action_mat[i] = _fix_act(agent.act(obs_batch[i], sample=True))

        action_env = torch.as_tensor(action_mat, dtype=torch.float32)
        try:
            action_env = action_env.to(getattr(env, "device", action_env.device))
        except Exception:
            pass

        # Step environment
        step_out = env.step(action_env)
        if len(step_out) == 5:
            next_obs, reward, terminated, truncated, info = step_out
            done_vec = np.asarray(_to_np(terminated)) | np.asarray(_to_np(truncated))
        else:
            next_obs, reward, done_raw, info = step_out
            done_vec = np.asarray(_to_np(done_raw))
        done = bool(np.asarray(done_vec).reshape(-1)[0])
        next_obs_batch = flatten_obs_batch(next_obs, num_envs)
        next_obs0 = next_obs_batch[0]

        # Capture camera image
        rows_any = int(getattr(args, 'tiled_camera_rows', 0) or 0)
        cols_any = int(getattr(args, 'tiled_camera_cols', 0) or 0)
        img = capture_vlm_image(
            env,
            save_path=(log_dir / "vlm_images"
                        if args.save_camera_images and (step % max(1, args.camera_interval) == 0)
                        else None),
            step=step,
            hide_robot_flag=(not args.hide_robot),
            tiled_rows=(rows_any if rows_any > 0 else None),
            tiled_cols=(cols_any if cols_any > 0 else None),
        )
        if img is None:
            try:
                rendered = env.render()
                img = rendered.get("rgb_array", None) if isinstance(rendered, dict) else rendered
            except Exception:
                pass

        # Compute learned reward
        reward0 = float(np.asarray(_to_np(reward)).reshape(-1)[0])
        reward_hat0 = reward0
        per_env_imgs: list[np.ndarray] | None = None
        per_env_reward_hat: np.ndarray | None = None
        if img is not None:
            try:
                per_env_imgs = _split_tiled_rgb(
                    np.asarray(img), num_envs,
                    rows=(rows_any if rows_any > 0 else None),
                    cols=(cols_any if cols_any > 0 else None),
                )
            except Exception:
                per_env_imgs = None

            if active_envs > 1 and debug_tiled_split and not printed_tiled_split:
                printed_tiled_split = True
                try:
                    arr0 = np.asarray(img)
                    ci = int(cols_any) if cols_any > 0 else int(math.ceil(math.sqrt(num_envs)))
                    ri = int(rows_any) if rows_any > 0 else int(math.ceil(num_envs / ci))
                    print(f"[TILED] full={arr0.shape} envs={num_envs} rows={ri} cols={ci} "
                          f"tile={arr0.shape[0]//ri}x{arr0.shape[1]//ci}")
                except Exception:
                    pass

            try:
                reward_hat0, per_env_reward_hat = infer_reward_hat(
                    reward_model, img, per_env_imgs, active_envs,
                )
            except Exception:
                per_env_imgs = None
                per_env_reward_hat = None
                reward_hat0 = reward0

        # Store transitions
        actions_np = _to_np(action_env)
        if actions_np.ndim == 1:
            actions_np = actions_np.reshape(1, -1)
        done_vec = np.asarray(done_vec).reshape(-1)
        per_env_reward = per_env_reward_hat.astype(np.float32, copy=False) if (active_envs > 1 and per_env_reward_hat is not None) else None

        if active_envs <= 1:
            img0 = (per_env_imgs[0] if (per_env_imgs and len(per_env_imgs) > 0) else img)
            replay_buffer.add(
                obs,
                np.asarray(actions_np[0], dtype=np.float32).reshape(-1),
                np.asarray([reward_hat0], dtype=np.float32),
                next_obs0,
                float(done_vec[0]),
                0.0 if (episode_step + 1 == horizon) else float(done_vec[0]),
                image=_ensure_hwc_uint8(img0, image_hw),
            )
        else:
            for i in range(active_envs):
                ri = float(per_env_reward[i]) if (per_env_reward is not None and i < per_env_reward.size) else reward_hat0
                img_i = per_env_imgs[i] if (per_env_imgs and i < len(per_env_imgs)) else None
                replay_buffer.add(
                    obs_batch[i],
                    np.asarray(actions_np[i], dtype=np.float32).reshape(-1),
                    np.asarray([ri], dtype=np.float32),
                    next_obs_batch[i],
                    float(done_vec[i]),
                    0.0 if (episode_step + 1 == horizon) else float(done_vec[i]),
                    image=_ensure_hwc_uint8(img_i, image_hw),
                )

        # Update VLM sampling pool
        if use_multi_env_vlm_pool and reward_learning_active and img is not None:
            try:
                if per_env_imgs and len(per_env_imgs) >= active_envs:
                    vlm_pool_manager.update([per_env_imgs[i] for i in range(active_envs)], step_tag=int(env_steps))
                else:
                    vlm_pool_manager.update([np.asarray(img)], step_tag=int(env_steps))
            except Exception:
                pass
        elif not use_multi_env_vlm_pool and active_envs <= 1:
            try:
                img_for_data = per_env_imgs[0] if (per_env_imgs and len(per_env_imgs) > 0) else img
                reward_model.add_data(
                    obs,
                    np.asarray(actions_np[0], dtype=np.float32).reshape(-1),
                    reward0, float(done_vec[0]),
                    img=_ensure_hwc_uint8(img_for_data, image_hw),
                )
            except Exception:
                pass

        # Bookkeeping
        episode_reward += reward_hat0
        true_episode_reward += reward0
        try:
            scalar_abs = get_scalar_abs_from_scene(env)
            if scalar_abs is None:
                scalar_rel = extract_scalar_rel_from_obs(next_obs, scalar_obs_slice)
                if scalar_rel is not None:
                    default_abs = get_scalar_default_abs(env)
                    if default_abs is not None and np.isfinite(default_abs):
                        scalar_abs = float(scalar_rel + float(default_abs))
            if scalar_abs is not None and np.isfinite(scalar_abs):
                train_min_dist = min(train_min_dist, abs(float(scalar_abs) - goal_abs))
        except Exception:
            pass
        obs_batch = next_obs_batch
        obs = next_obs0
        episode_step += 1
        step += 1
        env_steps += int(active_envs)

        if bool(getattr(args, 'save_preference_eval_pool', False)):
            try:
                collect_imgs = None
                if per_env_imgs and len(per_env_imgs) >= active_envs:
                    collect_imgs = [per_env_imgs[i] for i in range(active_envs)]
                elif active_envs <= 1 and img is not None:
                    collect_imgs = [np.asarray(img)]
                preference_eval_writer.maybe_collect(
                    env=env,
                    args=args,
                    images=collect_imgs,
                    env_steps=int(env_steps),
                    loop_step=int(step),
                    seed=int(args.seed),
                    episode=int(episode),
                    episode_step=int(episode_step),
                    active_envs=int(active_envs),
                    done_vec=done_vec,
                )
            except Exception as exc:
                print(f"[PREF-EVAL] Failed to save benchmark state at env_steps={env_steps}: {exc}")

        # One-time update-regime logging
        if not last_logged_update_regime and step > (args.num_seed_steps + args.num_unsup_steps):
            logger.log('train/num_envs', float(num_envs), env_steps)
            logger.log('train/active_envs', float(active_envs), env_steps)
            logger.log('train/batch_size', float(args.batch_size), env_steps)
            logger.log('train/gradient_update_base', float(args.gradient_update), env_steps)
            logger.log('train/updates_per_env_step', float(getattr(args, 'updates_per_env_step', 0.0) or 0.0), env_steps)
            logger.log('train/update_scale_cap', float(update_scale_cap), env_steps)
            logger.log('train/update_scale_mode_fixed_one', 1.0 if update_scale_mode == 'fixed_one' else 0.0, env_steps)
            logger.log('train/actor_lr', float(args.actor_lr), env_steps)
            logger.log('train/critic_lr', float(args.critic_lr), env_steps)
            last_logged_update_regime = True

        # Learning schedule
        if (not initial_label_done) and (env_steps >= unsup_end_steps):
            initial_label_done = True
            reward_model.change_batch(1.0)
            if use_multi_env_vlm_pool:
                try:
                    n_pool = vlm_pool_manager.sync_reward_model_dataset(step_tag=int(env_steps))
                    print(f"[VLM] Synced representative pool: n_imgs={n_pool}")
                    _log_vlm_pool_selection(logger, vlm_pool_manager, int(env_steps))
                except Exception as exc:
                    print(f"[VLM] Representative pool selection failed: {type(exc).__name__}: {exc}")
                    raise

            target = reward_model.mb_size
            acc_sum = run_vlm_labeling(
                reward_model, target,
                max(1, min(int(args.vlm_chunk), target)),
                "VLM labeling (initial)", bool(args.vlm_progress),
            )
            labeled_feedback += acc_sum
            total_feedback += target
            print(f"[VLM] Initial labeling: accepted {acc_sum}/{target} pairs")

            reward_model.eval()
            replay_buffer.relabel_with_predictor(reward_model)
            reward_model.train()
            agent.reset_critic()
            agent.update_after_reset(
                replay_buffer, logger, env_steps,
                gradient_update=args.gradient_update, policy_update=True,
            )

            try:
                logger.log('train/vlm_label_acc', float(getattr(reward_model, 'vlm_label_acc', np.nan)), env_steps)
            except Exception:
                pass
            logger.log('train/feedback_total', float(total_feedback), env_steps)
            logger.log('train/feedback_labeled', float(labeled_feedback), env_steps)
            logger.log('train/vlm_accepted_round', float(acc_sum), env_steps)
            if target > 0:
                logger.log('train/vlm_accept_rate_round', float(acc_sum) / float(target), env_steps)

            # Build stability anchors
            if bool(getattr(args, 'freeze_on_reward_stability', False)):
                try:
                    anchor_inputs, last_anchor_pred = build_stability_anchors(
                        replay_buffer, reward_model,
                        int(getattr(args, 'stability_anchor_size', 256)),
                    )
                    if anchor_inputs is not None:
                        logger.log('train/reward_stability_anchor', float(anchor_inputs.shape[0]), env_steps)
                        logger.dump(env_steps)
                except Exception:
                    pass

        elif initial_label_done:
            # Gradient updates
            scale = 1 if update_scale_mode == 'fixed_one' else max(1, min(active_envs, update_scale_cap))
            logger.log('train/active_envs', float(active_envs), env_steps)

            ups = float(getattr(args, 'updates_per_env_step', 0.0) or 0.0)
            if ups > 0.0:
                delta = max(0, env_steps - last_env_steps_for_update)
                max_upd = max(1, args.gradient_update * update_scale_cap)
                update_budget += ups * delta
                update_budget = min(update_budget, max_upd + 1.0)
                n_updates = min(int(update_budget), max_upd)
                logger.log('train/gradient_updates_this_iter', float(n_updates), env_steps)
                if delta > 0:
                    logger.log('train/updates_per_env_step_effective', float(n_updates) / delta, env_steps)
                for _ in range(n_updates):
                    agent.update(replay_buffer, logger, env_steps, gradient_update=1)
                update_budget -= n_updates
                last_env_steps_for_update = env_steps
            else:
                gu = max(1, args.gradient_update * scale)
                logger.log('train/gradient_updates_this_iter', float(gu), env_steps)
                agent.update(replay_buffer, logger, env_steps, gradient_update=gu)

        # Checkpointing
        vlm_pool_manager.maybe_save_snapshot(int(env_steps), reward_learning_active)
        if next_ckpt_step is not None and env_steps >= next_ckpt_step:
            while next_ckpt_step is not None and env_steps >= next_ckpt_step:
                try:
                    agent.save(str(log_dir / "models"), int(next_ckpt_step))
                    print(f"[CKPT] Saved at env_steps={next_ckpt_step}")
                except Exception:
                    pass
                next_ckpt_step += int(args.save_interval)

        # Episode end
        if done or (episode_step >= horizon):
            logger.log('train/loop_step', float(step), env_steps)
            logger.log('train/episode_reward', episode_reward, env_steps)
            logger.log('train/true_episode_reward', true_episode_reward, env_steps)
            logger.log('train/episode', episode, env_steps)
            logger.log('train/episode_length', episode_step, env_steps)
            try:
                task_obj = get_active_task()
                val = float(train_min_dist) if np.isfinite(train_min_dist) else float('nan')
                logger.log(f"train/{task_obj.train_min_log_key}", val, env_steps)
            except Exception:
                pass
            logger.dump(env_steps)

            # Evaluation at episode boundaries
            if next_eval_step is not None and env_steps >= next_eval_step:
                try:
                    record_dir = None
                    want_videos = (
                        bool(getattr(args, 'save_eval_videos', False))
                        and not bool(getattr(args, 'eval_disable_video_capture', False))
                    )
                    if want_videos:
                        k = int(round(env_steps / 10000))
                        record_dir = log_dir / "eval" / f"eval_videos_{k*10}K"

                    use_serial = bool(getattr(args, 'eval_serial', False))
                    if (not use_serial) and num_envs > 1:
                        metrics = evaluate_policy_on_env_parallel(
                            env, agent, args.eval_episodes, horizon,
                            args.success_thresh,
                            record_dir=record_dir,
                            record_fps=int(getattr(args, 'eval_video_fps', 15)),
                            goal_abs=goal_abs,
                            disable_video_capture=bool(getattr(args, 'eval_disable_video_capture', False)),
                        )
                    else:
                        metrics = evaluate_policy_on_env(
                            env, agent, args.eval_episodes, horizon,
                            args.success_thresh,
                            record_dir=record_dir,
                            record_fps=int(getattr(args, 'eval_video_fps', 15)),
                            goal_abs=goal_abs,
                            disable_video_capture=bool(getattr(args, 'eval_disable_video_capture', False)),
                            image_hw=image_hw,
                        )

                    task_obj = get_active_task()
                    logger.log('eval/success_rate', metrics.get('success_rate', 0.0), env_steps)
                    logger.log(f"eval/{task_obj.eval_mean_min_key}",
                               metrics.get(str(task_obj.eval_mean_min_key), 0.0), env_steps)
                    if 'mean_episode_length' in metrics:
                        logger.log('eval/episode_length', metrics['mean_episode_length'], env_steps)
                    logger.dump(env_steps, ty='eval')
                    print(
                        f"[EVAL] env_steps={env_steps} success={metrics.get('success_rate', 0.0):.3f} "
                        f"{task_obj.eval_print_label}={metrics.get(str(task_obj.eval_mean_min_key), float('nan')):.4f} "
                        f"ep_len={metrics.get('mean_episode_length', float('nan')):.1f}"
                    )

                    # Save best checkpoints
                    if metrics.get('success_rate', 0.0) > best_success:
                        best_success = metrics['success_rate']
                        agent.save(str(log_dir / "models"), f"best_success_{step}")
                        print(f"[CKPT] BEST success at step {step} ({best_success:.2f})")
                    mean_min = metrics.get(str(task_obj.eval_mean_min_key), float('nan'))
                    if np.isfinite(mean_min) and mean_min < best_min_dist:
                        best_min_dist = float(mean_min)
                        agent.save(str(log_dir / "models"), f"{task_obj.best_min_ckpt_prefix}_{step}")
                        print(f"[CKPT] BEST {task_obj.eval_print_label} at step {step} ({best_min_dist:.4f})")
                except Exception as e:
                    import traceback
                    print(f"[EVAL][ERROR] env_steps={env_steps}: {type(e).__name__}: {e}")
                    traceback.print_exc()
                while env_steps >= next_eval_step:
                    next_eval_step += int(args.eval_every)

            obs, _ = env.reset()
            obs_batch = flatten_obs_batch(obs, num_envs)
            obs = obs_batch[0]
            episode_reward = 0.0
            true_episode_reward = 0.0
            episode_step = 0
            train_min_dist = float("inf")
            episode += 1

        # Periodic VLM relabeling
        if (
            initial_label_done
            and env_steps >= next_label_step
            and total_feedback < args.max_feedback
            and (effective_freeze_after is None or env_steps < effective_freeze_after)
        ):
            while env_steps >= next_label_step:
                next_label_step += base_interact
            if reward_model.mb_size + total_feedback > args.max_feedback:
                reward_model.set_batch(args.max_feedback - total_feedback)

            if use_multi_env_vlm_pool:
                try:
                    vlm_pool_manager.sync_reward_model_dataset(step_tag=int(env_steps))
                    _log_vlm_pool_selection(logger, vlm_pool_manager, int(env_steps))
                except Exception as exc:
                    print(f"[VLM] Representative pool selection failed: {type(exc).__name__}: {exc}")
                    raise

            target = reward_model.mb_size
            acc_sum = run_vlm_labeling(
                reward_model, target,
                max(1, min(int(args.vlm_chunk), target)),
                "VLM labeling", bool(args.vlm_progress),
            )
            total_feedback += target
            labeled_feedback += acc_sum
            print(f"[VLM] Labeling round: accepted {acc_sum}/{target} pairs")

            total_acc = train_reward_epochs(reward_model, args.reward_update)
            print(f"Reward function updated — ACC: {total_acc:.3f}")

            try:
                logger.log('train/vlm_label_acc', float(getattr(reward_model, 'vlm_label_acc', np.nan)), env_steps)
            except Exception:
                pass
            logger.log('train/feedback_total', float(total_feedback), env_steps)
            logger.log('train/feedback_labeled', float(labeled_feedback), env_steps)
            logger.log('train/vlm_accepted_round', float(acc_sum), env_steps)
            if target > 0:
                logger.log('train/vlm_accept_rate_round', float(acc_sum) / float(target), env_steps)

            # Relabel replay buffer
            if effective_freeze_after is None or env_steps < effective_freeze_after:
                reward_model.eval()
                replay_buffer.relabel_with_predictor(reward_model)
                reward_model.train()
            else:
                print("[REWARD] Skipped relabeling (reward model frozen).")

            # Stability-based dynamic freeze
            if bool(getattr(args, 'freeze_on_reward_stability', False)) and anchor_inputs is not None:
                try:
                    should_freeze, last_anchor_pred, stability_good_count, low_accept_count = (
                        check_stability_freeze(
                            reward_model, anchor_inputs, last_anchor_pred,
                            args, acc_sum,
                            stability_good_count, low_accept_count,
                            env_steps, step, logger,
                        )
                    )
                    if should_freeze and freeze_after_step_dynamic is None:
                        freeze_after_step_dynamic = int(env_steps)
                except Exception:
                    pass

    # Final checkpoint
    try:
        agent.save(str(log_dir / "models"), min(env_steps, args.total_steps))
        env.close()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise RuntimeError(
        "Do not run train_isaaclab_core.py directly. Run a task wrapper (e.g. "
        "train_isaaclab_drawer_open.py / train_isaaclab_soccer.py) "
    )
