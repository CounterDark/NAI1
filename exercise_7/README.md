## Exercise 7 — Reinforcement Learning (Atari / ALE)

## Authors

- Mateusz Anikiej
- Aleksander Kunkowski

## Prerequisites

1. Python 3.13 installed
2. `uv` installed (`https://docs.astral.sh/uv/getting-started/installation/`)
3. Atari ROMs installed (one-time, license acceptance required):

```bash
AutoROM --accept-license
```

## Setup

Use `uv` to create a virtual environment and install dependencies.

Option A: using Makefile

```bash
make setup
```

Option B: using uv directly

```bash
uv sync
```

## Running the program

This exercise supports two modes: `train` and `run`.

Option A: using Makefile

```bash
make exe7 ARGS="train"
make exe7 ARGS="run"
```

Option B: using uv directly

```bash
cd exercise_7
uv run python src/main.py train
uv run python src/main.py run
```

## Configuration

All settings are in `exercise_7/src/main.py` inside the `Config` dataclass:

- **Game**: change `env_id` (e.g. `ALE/Adventure-v5`)
- **Device**: `device` (`auto`, `mps`, `cpu`)
- **Training budget**: `total_timesteps`
- **Evaluation cadence**: `eval_every_steps`, `eval_episodes`
- **Checkpoint cadence**: `checkpoint_every_steps` (saves a resume checkpoint + replay buffer)

### Config entries descriptions

You can adjust each entry in `Config` to control training.

- **env_id**: Which ALE Atari game to train on (e.g. `ALE/Seaquest-v5`).
- **seed**: Random seed for environment initialization and training reproducibility.
- **device**: Where the neural network runs (`mps` on Apple Silicon GPU, `cpu`, or `auto`).
- **n_envs**: Number of parallel environments to collect experience faster (higher = more throughput, more CPU/RAM).
- **total_timesteps**: Total number of environment steps to train for; bigger is usually better for Atari.
- **runs_dir**: Directory for TensorBoard logs written during training.
- **models_dir**: Directory where models/checkpoints are saved.
- **run_name**: Name used for the TensorBoard run (so multiple runs don’t overwrite each other).
- **checkpoint_every_steps**: How often (in timesteps) to save a resume checkpoint (`checkpoint_latest.zip`) and replay buffer.
- **eval_every_steps**: How often (in timesteps) to evaluate the current policy during training.
- **eval_episodes**: Number of episodes used for each evaluation (more = smoother metric, slower eval).
- **learning_rate**: DQN optimizer step size; too high can destabilize learning, too low learns slowly.
- **buffer_size**: Size of the replay buffer storing past transitions for off-policy learning.
- **learning_starts**: Timesteps of pure data collection before training updates start (helps stabilize early learning).
- **batch_size**: Number of samples per gradient update.
- **train_freq**: How often (in env steps) to do a training update.
- **target_update_interval**: How often to update the target network (stabilizes Q-learning).
- **exploration_fraction**: Fraction of training over which epsilon decays from 1.0 to `exploration_final_eps`.
- **exploration_final_eps**: Final epsilon for epsilon-greedy exploration (minimum random action rate).
- **run_episodes**: Number of episodes to render in `run` mode.
- **run_deterministic**: If `True`, uses greedy actions (best guess); if `False`, adds randomness.

## Outputs

Saved under `exercise_7/models/`:

- `best_model.zip`: best-performing model from periodic evaluation
- `final_model.zip`: final model after training finishes
- `checkpoint_latest.zip`: latest checkpoint (for resuming training)
- `checkpoint_latest_replay.pkl`: replay buffer (required for good DQN resume)

Training logs are saved under `exercise_7/runs/` (TensorBoard).

## Project Description

This exercise implements reinforcement learning for an Atari game in ALE using the Gymnasium interface and Stable-Baselines3.

- The agent learns directly from **image observations** (frames) using a CNN-based policy.
- Training periodically evaluates the current policy and saves the best model.
- Training checkpoints are saved to allow resuming long runs.
