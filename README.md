# RAPID: Scaling Vision-Language Reward Learning for Robot Manipulation in Parallel Simulation

Code for our paper: *Scaling Vision-Language Reward Learning for Robot Manipulation in Parallel Simulation*.

## Installation

### Prerequisites

1. **Isaac Sim**: follow the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html#installing-isaac-sim)
2. **IsaacLab**: follow the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html#installing-isaac-lab)

### Install

```bash
git clone https://github.com/rapid-vlm/rapid-vlm-rl.git
cd /path/to/IsaacLab

# install into IsaacLab's Python environment
./isaaclab.sh -p -m pip install -e /path/to/rapid-vlm-rl
```

## Setup OpenRouter

VLM preference queries go through [OpenRouter](https://openrouter.ai/). Create an account and
get your API key at [openrouter.ai/keys](https://openrouter.ai/keys).

## Training

1. Go to the Isaac Lab folder:
    ```bash
    cd /path/to/IsaacLab
    ```
2. Set the key in your environment:
    ```bash
    export OPENROUTER_API_KEY=<your_key>
    ```
3. To speed up labeling, enable parallel VLM requests. We send up to 40 queries concurrently:
    ```bash
    export VLM_PARALLEL_REQUESTS=40
    ```
4. Run a training script through IsaacLab's launcher. The task scripts are in
   `scripts/`; see [training.md](docs/training.md) for the full command list.

## Evaluation

To visualize a policy or record a video/GIF from a checkpoint, use
`scripts/eval_record_policy.py`.

For your own training logs, point `--checkpoint_dir` to a run's `models/`
directory:

```bash
# Play interactively (GUI, no output saved):
./isaaclab.sh -p /path/to/rapid-vlm-rl/scripts/eval_record_policy.py \
  --device cuda:0 \
  --task <task_name> \
  --checkpoint_dir /path/to/logs/rapid/<exp_name>/models \
  --step 1000000 \
  --episodes 5 \
  --seed 0 \
  --out_dir ""

# Record to MP4 and GIF (headless):
./isaaclab.sh -p /path/to/rapid-vlm-rl/scripts/eval_record_policy.py \
  --device cuda:0 \
  --headless \
  --task <task_name> \
  --checkpoint_dir /path/to/logs/rapid/<exp_name>/models \
  --step 1000000 \
  --episodes 5 \
  --seed 0 \
  --out_dir <path_to_output_dir> \
  --save_gif --gif_fps 15 --mp4_fps 30
```

## Adding a New Task

If you want to add a task beyond the 5 provided, see [new-tasks.md](docs/new-tasks.md).


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project builds on [RL-VLM-F](https://github.com/yufeiwang63/RL-VLM-F) and [IsaacLab](https://github.com/isaac-sim/IsaacLab). We thank the authors for open-sourcing their code.
