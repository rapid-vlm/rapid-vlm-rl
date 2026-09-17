# Training Commands

## Open Drawer

```bash
./isaaclab.sh -p /path/to/rapid-vlm-rl/scripts/train_drawer_open.py \
  --headless --hide_robot --success_thresh 0.04 --total_steps 1000000 \
  --seed 0 --vlm_model google/gemma-3-12b-it --one_prompt \
  --eval_every 10000 --eval_episodes 10 \
  --freeze_on_reward_stability --stability_anchor_size 512 \
  --stability_corr_thresh 0.99 --stability_delta_eps 0.025 \
  --stability_patience 2 \
  --vlm_progress --vlm_chunk 40 \
  --num_envs 50 \
  --update_scale_mode cap_active_envs --updates_per_env_step 0.5 --update_scale_cap 50 \
  --vlm_pool_strategy representative --vlm_pool_size 50 \
  --vlm_pool_feature_res 16 --vlm_pool_candidate_cap 5000 \
  --exp_name isaaclab_drawer_open \
  --reward_batch 40 --reward_update 5 --drop_replay_images_after_freeze \
  --replay_buffer_capacity 500000 --num_seed_steps 10000 --num_unsup_steps 0 \
  --vlm_pool_refresh_every 500 --max_feedback 20000
```

## Open Window

```bash
./isaaclab.sh -p /path/to/rapid-vlm-rl/scripts/train_window_open.py \
  --headless --hide_robot --close_target 0.0 --open_target 0.2 --success_thresh 0.05 \
  --total_steps 1000000 --seed 0 --vlm_model google/gemma-3-12b-it --one_prompt \
  --eval_every 10000 --eval_episodes 10 \
  --freeze_on_reward_stability --stability_anchor_size 512 \
  --stability_corr_thresh 0.99 --stability_delta_eps 0.025 \
  --stability_patience 2 \
  --vlm_progress --vlm_chunk 40 \
  --num_envs 50 \
  --update_scale_mode cap_active_envs --updates_per_env_step 0.5 --update_scale_cap 50 \
  --vlm_pool_strategy representative --vlm_pool_size 50 \
  --vlm_pool_feature_res 16 --vlm_pool_candidate_cap 5000 \
  --exp_name isaaclab_window_open \
  --reward_batch 40 --reward_update 5 --drop_replay_images_after_freeze \
  --replay_buffer_capacity 500000 --num_seed_steps 10000 --num_unsup_steps 0 \
  --vlm_pool_refresh_every 500 --max_feedback 20000
```

## Soccer

```bash
./isaaclab.sh -p /path/to/rapid-vlm-rl/scripts/train_soccer.py \
  --headless --hide_robot --success_thresh 0.0 --total_steps 1000000 \
  --seed 0 --vlm_model google/gemma-3-12b-it --one_prompt \
  --eval_every 10000 --eval_episodes 10 \
  --freeze_on_reward_stability --stability_anchor_size 512 \
  --stability_corr_thresh 0.99 --stability_delta_eps 0.025 \
  --stability_patience 2 \
  --vlm_progress --vlm_chunk 40 \
  --num_envs 50 \
  --update_scale_mode cap_active_envs --updates_per_env_step 0.5 --update_scale_cap 50 \
  --vlm_pool_strategy representative --vlm_pool_size 50 \
  --vlm_pool_feature_res 16 --vlm_pool_candidate_cap 5000 \
  --exp_name isaaclab_soccer \
  --reward_batch 40 --reward_update 5 --drop_replay_images_after_freeze \
  --replay_buffer_capacity 500000 --num_seed_steps 10000 --num_unsup_steps 0 \
  --vlm_pool_refresh_every 500 --max_feedback 20000
```

## Push Button

```bash
./isaaclab.sh -p /path/to/rapid-vlm-rl/scripts/train_button_push.py \
  --headless --hide_robot --start_pos 0.0 --push_target -0.06 --success_thresh 0.01 \
  --total_steps 1000000 --seed 0 --vlm_model google/gemma-3-12b-it --one_prompt \
  --eval_every 10000 --eval_episodes 10 \
  --freeze_on_reward_stability --stability_anchor_size 512 \
  --stability_corr_thresh 0.99 --stability_delta_eps 0.025 \
  --stability_patience 2 \
  --vlm_progress --vlm_chunk 40 \
  --num_envs 50 \
  --update_scale_mode cap_active_envs --updates_per_env_step 0.5 --update_scale_cap 50 \
  --vlm_pool_strategy representative --vlm_pool_size 50 \
  --vlm_pool_feature_res 16 --vlm_pool_candidate_cap 5000 \
  --exp_name isaaclab_button_push \
  --reward_batch 40 --reward_update 5 --drop_replay_images_after_freeze \
  --replay_buffer_capacity 500000 --num_seed_steps 10000 --num_unsup_steps 0 \
  --vlm_pool_refresh_every 500 --max_feedback 20000
```

## Sweep Into

```bash
./isaaclab.sh -p /path/to/rapid-vlm-rl/scripts/train_sweep_into.py \
  --headless --hide_robot --success_thresh 0.0 --total_steps 1000000 \
  --seed 0 --vlm_model google/gemma-3-12b-it --one_prompt \
  --eval_every 10000 --eval_episodes 10 \
  --freeze_on_reward_stability --stability_anchor_size 512 \
  --stability_corr_thresh 0.99 --stability_delta_eps 0.025 \
  --stability_patience 2 \
  --vlm_progress --vlm_chunk 40 \
  --num_envs 50 \
  --update_scale_mode cap_active_envs --updates_per_env_step 0.5 --update_scale_cap 50 \
  --vlm_pool_strategy representative --vlm_pool_size 50 \
  --vlm_pool_feature_res 16 --vlm_pool_candidate_cap 5000 \
  --exp_name isaaclab_sweep_into \
  --reward_batch 40 --reward_update 5 --drop_replay_images_after_freeze \
  --replay_buffer_capacity 500000 --num_seed_steps 10000 --num_unsup_steps 0 \
  --vlm_pool_refresh_every 500 --hole_z_threshold 0.96 \
  --max_feedback 20000
```
