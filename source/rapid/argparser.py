"""Argument parser for the RAPID shared training runner."""
from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

from rapid.isaaclab_utils import get_active_task

def build_argparser() -> argparse.ArgumentParser:
    task = get_active_task()
    parser = argparse.ArgumentParser(description=str(task.description))
    # Isaac app flags
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
    parser.add_argument("--hide_robot", action="store_true", help="Hide robot visuals during training")

    # Task-specific flags (targets, modes, etc.)
    task.add_task_args(parser)
    # Training config (subset of RAPID knobs)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_steps", type=int, default=1_000_000)
    # Resume from checkpoint
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint directory")
    parser.add_argument("--resume_step", type=int, default=None, help="Step number to resume from (e.g., 800000)")

    parser.add_argument("--num_seed_steps", type=int, default=1000)
    parser.add_argument("--num_unsup_steps", type=int, default=9000)
    parser.add_argument("--num_interact", type=int, default=4000)
    parser.add_argument("--max_feedback", type=int, default=20000)
    parser.add_argument("--reward_update", type=int, default=10)
    parser.add_argument("--reward_batch", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument(
        "--replay_buffer_capacity",
        type=int,
        default=200_000,
        help=(
            "Max transitions stored in the replay buffer."
            "Important for RAM: with image-based rewards, replay images dominate memory."
        ),
    )
    parser.add_argument(
        "--drop_replay_images_after_freeze",
        action="store_true",
        help=(
            "After the reward model is frozen (no more relabeling), free replay buffer image storage to reduce RAM. "
            "Safe for typical runs where learned reward is frozen and no further relabeling is needed."
        ),
    )
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--hidden_depth", type=int, default=3)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--gradient_update", type=int, default=1)
    parser.add_argument(
        "--updates_per_env_step",
        type=float,
        default=0.0,
        help=(
            "If >0, drive SAC updates by env_steps instead of loop_step: run roughly "
            "updates_per_env_step * (env_steps_delta) gradient updates each iteration (capped by update_scale_cap). "
            "This keeps update/data ratio consistent across different num_envs."
        ),
    )
    # VLM & images
    parser.add_argument("--vlm_model", type=str, default="google/gemma-3-12b-it")
    parser.add_argument("--one_prompt", action="store_true", help="Use single-prompt VLM labeling (one request per pair) instead of two-stage (analysis+summary)")
    parser.add_argument("--save_camera_images", action="store_true")
    parser.add_argument("--camera_interval", type=int, default=1000)
    parser.add_argument(
        "--tiled_camera_rows",
        type=int,
        default=0,
        help="If >0, explicitly set tiled camera rows for splitting frames into per-env images.",
    )
    parser.add_argument(
        "--tiled_camera_cols",
        type=int,
        default=0,
        help="If >0, explicitly set tiled camera cols for splitting frames into per-env images.",
    )
    parser.add_argument(
        "--update_scale_cap",
        type=int,
        default=16,
        help="Cap on scaling SAC gradient updates by active_envs (keeps update/data ratio bounded).",
    )
    parser.add_argument(
        "--update_scale_mode",
        type=str,
        default="cap_active_envs",
        choices=["cap_active_envs", "fixed_one"],
        help="How to scale gradient updates post-freeze: cap_active_envs multiplies by min(active_envs, cap); fixed_one disables scaling.",
    )
    parser.add_argument(
        "--debug_tiled_split",
        action="store_true",
        help="Print a one-time summary of tiled image splitting (rows/cols, per-tile resolution).",
    )
    # Logging
    parser.add_argument("--log_root", type=str, default=str(Path("logs/rapid").absolute()))
    parser.add_argument("--exp_name", type=str, default=str(task.default_exp_name))
    parser.add_argument("--save_interval", type=int, default=20000, help="Save model every N steps (0 to disable)")
    # VLM progress display
    parser.add_argument("--vlm_progress", action="store_true", help="Show a progress bar during VLM labeling")
    parser.add_argument("--vlm_chunk", type=int, default=10, help="Labeling chunk size per VLM request loop (<= mb_size)")
    parser.add_argument(
        "--save_vlm_pool_images",
        action="store_true",
        help=(
            "If set, maintain a 'vlm_pool/current' folder inside the run log directory that mirrors the current VLM sampling pool. "
            "The folder is refreshed whenever the pool changes (FIFO: on new images; representative: when recomputed before labeling)."
        ),
    )
    parser.add_argument(
        "--vlm_pool_refresh_every",
        type=int,
        default=0,
        help=(
            "Only used when --save_vlm_pool_images and --vlm_pool_strategy=representative. "
            "If >0, recompute and dump the representative pool every N env_steps during rollout (best-effort). "
            "If 0, representative pool images are dumped only when the pool is recomputed for VLM labeling."
        ),
    )
    parser.add_argument(
        "--save_vlm_pool_snapshots_every",
        type=int,
        default=0,
        help=(
            "If >0, save a timestamped snapshot of the VLM pool every N env_steps to vlm_pool/step_XXXXX/. "
            "This is independent of --save_vlm_pool_images and allows tracking pool evolution over training. "
            "Each snapshot is saved to a separate folder for that step."
        ),
    )
    parser.add_argument(
        "--save_preference_eval_pool",
        action="store_true",
        help=(
            "Save per-env camera images and simulator state metadata for an offline "
            "VLM preference/oracle benchmark. This does not add extra VLM calls."
        ),
    )
    parser.add_argument(
        "--preference_eval_pool_interval",
        type=int,
        default=48550,
        help=(
            "Save one batch of per-env benchmark states every N env_steps when "
            "--save_preference_eval_pool is enabled. With 50 envs and 1M steps, "
            "48550 saves about 1000 images while avoiding common episode-boundary alignment."
        ),
    )
    parser.add_argument(
        "--preference_eval_pool_max_states",
        type=int,
        default=0,
        help="Maximum benchmark states/images to save for this run (0 means no cap).",
    )
    parser.add_argument(
        "--preference_eval_pool_dir",
        type=str,
        default="",
        help=(
            "Optional output directory for the benchmark pool. If empty, writes to "
            "<run_log_dir>/preference_eval_pool."
        ),
    )
    parser.add_argument(
        "--preference_eval_pool_keep_done",
        action="store_true",
        help=(
            "Also save states whose done flag is true. By default done states are skipped "
            "because synchronized vectorized runs can otherwise collect mostly reset/terminal frames."
        ),
    )
    parser.add_argument(
        "--vlm_pool_size",
        type=int,
        default=50,
        help=(
            "Size of the image pool used to sample VLM preference pairs (conceptually the RewardModel trajectory FIFO when size_segment=1). "
            "When --vlm_pool_strategy=representative, this is the number of representative images retained across all envs."
        ),
    )
    parser.add_argument(
        "--vlm_pool_strategy",
        type=str,
        default="fifo",
        choices=["fifo", "representative"],
        help=(
            "How to maintain the VLM sampling pool before reward freeze. "
        ),
    )
    parser.add_argument(
        "--vlm_pool_feature_res",
        type=int,
        default=16,
        help=(
            "Feature resolution (square) used when --vlm_pool_feature_type=pixel "
            "(lower is faster). Ignored for reward_encoder."
        ),
    )
    parser.add_argument(
        "--vlm_pool_feature_type",
        type=str,
        default="pixel",
        choices=["pixel", "reward_encoder"],
        help=(
            "Representation used by representative farthest-point sampling. "
            "'pixel' uses fixed downsampled grayscale pixels; 'reward_encoder' "
            "uses the current learned CNN features from all reward-ensemble members."
        ),
    )
    parser.add_argument(
        "--vlm_pool_encoder_batch_size",
        type=int,
        default=128,
        help="GPU batch size for reward-encoder feature extraction during representative sampling.",
    )
    parser.add_argument(
        "--vlm_pool_candidate_cap",
        type=int,
        default=5000,
        help=(
            "Max number of recent candidate images retained for representative selection before each VLM labeling round. "
            "Higher improves diversity but uses more CPU/memory during selection."
        ),
    )
    # Evaluation hooks
    parser.add_argument("--eval_every", type=int, default=10000, help="Evaluate policy every N steps (0 to disable)")
    parser.add_argument("--eval_episodes", type=int, default=3, help="Number of eval episodes per evaluation")
    parser.add_argument(
        "--eval_serial",
        action="store_true",
        help="Force legacy serial eval (env0-only, episodes run one-after-another). By default eval is parallelized when num_envs>1.",
    )
    parser.add_argument(
        "--success_thresh",
        type=float,
        default=0.04,
        help="Distance-to-goal threshold in the task's scalar units (e.g., joint units for drawer/button; meters for soccer distance-to-goal-region).",
    )
    # Freeze reward model
    parser.add_argument("--freeze_reward_after", type=int, default=0, help="After this many env steps, stop updating the reward model and relabeling to reduce reward drift (0 disables)")
    # Logging
    parser.add_argument("--save_csv", action="store_true", help="Save train.csv and eval.csv logs in addition to TensorBoard (default: off)")
    # Evaluation video saving
    parser.add_argument("--save_eval_videos", action="store_true", help="Save MP4 videos for each evaluated episode into logs/.../eval/eval_videos_* folders")
    parser.add_argument("--eval_video_fps", type=int, default=15, help="Frames per second for saved evaluation videos")
    parser.add_argument(
        "--eval_disable_video_capture",
        action="store_true",
        help="Disable eval frame capture/video writing (more robust; still logs success metrics).",
    )

    # Reward-model stabilization
    parser.add_argument("--freeze_on_reward_stability", action="store_true", help="Freeze reward updates automatically when the learned reward stabilizes on an anchor set (corr>=threshold for patience rounds)")
    parser.add_argument("--stability_anchor_size", type=int, default=512, help="Number of anchor images sampled from the replay buffer to monitor reward predictor stability")
    parser.add_argument("--stability_corr_thresh", type=float, default=0.99, help="Minimum Pearson correlation between consecutive anchor predictions to count as stable")
    parser.add_argument("--stability_delta_eps", type=float, default=0.025, help="Additionally require mean absolute change of anchor predictions to be below this value")
    parser.add_argument("--stability_patience", type=int, default=2, help="Freeze after this many consecutive stable rounds")
    parser.add_argument("--min_vlm_accept_per_round", type=int, default=-1, help="Also freeze if accepted VLM labels per round drop to or below this value for the specified patience (-1 disables)")
    parser.add_argument("--min_vlm_accept_patience", type=int, default=3, help="Number of consecutive low-acceptance rounds to trigger freeze")

    # Let AppLauncher add its args (enable_cameras will be forced)
    AppLauncher.add_app_launcher_args(parser)
    return parser
