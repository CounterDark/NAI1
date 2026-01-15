"""

Run:
  make exe7 ARGS="train"
  make exe7 ARGS="run"

  or
  uv run python src/main.py train
  uv run python src/main.py run
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import ale_py  # noqa: F401
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv, VecFrameStack, VecTransposeImage

# Apple Silicon: fallback to CPU when an op isn't supported on MPS
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


@dataclass(frozen=True)
class Config:
    # Environment
    env_id: str = "ALE/Seaquest-v5"
    seed: int = 0
    device: Literal["auto", "mps", "cpu"] = "mps"

    # Throughput
    n_envs: int = 4

    # Training budget
    total_timesteps: int = 5_000_000

    # Saving / logging
    runs_dir: str = "runs"
    models_dir: str = "models"
    run_name: str = "dqn_seaquest"

    # Checkpointing (for resume)
    checkpoint_every_steps: int = 200_000

    # Evaluation (for "best model")
    eval_every_steps: int = 200_000
    eval_episodes: int = 5

    # DQN (sane Atari-ish defaults)
    learning_rate: float = 1e-4
    buffer_size: int = 1_000_000
    learning_starts: int = 80_000
    batch_size: int = 32
    train_freq: int = 4
    target_update_interval: int = 10_000
    exploration_fraction: float = 0.10
    exploration_final_eps: float = 0.01

    # Demo
    run_episodes: int = 1
    run_deterministic: bool = True


CFG = Config()


def paths(cfg: Config) -> dict[str, Path]:
    models = Path(cfg.models_dir)
    runs = Path(cfg.runs_dir)
    models.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)

    return {
        "models_dir": models,
        "runs_dir": runs,
        "best_model": models / "best_model.zip",
        "final_model": models / "final_model.zip",
        # "latest" checkpoint for resuming
        "latest_model": models / "checkpoint_latest.zip",
        "latest_replay": models / "checkpoint_latest_replay.pkl",
    }


def make_env(
    *, cfg: Config, render_mode: Literal["human", "rgb_array"] | None, n_envs: int
) -> VecEnv:
    """
    make_atari_env already applies common Atari wrappers.
    Then we:
      - stack 4 frames (motion information)
      - transpose to channel-first for PyTorch CNNs
    """
    env = make_atari_env(
        cfg.env_id,
        n_envs=int(n_envs),
        seed=int(cfg.seed),
        env_kwargs={"render_mode": render_mode},
    )
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    return env


class SeaquestTrainer:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.p = paths(cfg)

    def _build_model(self, env: VecEnv) -> DQN:
        return DQN(
            policy="CnnPolicy",
            env=env,
            device=str(self.cfg.device),
            verbose=1,
            tensorboard_log=str(self.p["runs_dir"]),
            learning_rate=float(self.cfg.learning_rate),
            buffer_size=int(self.cfg.buffer_size),
            learning_starts=int(self.cfg.learning_starts),
            batch_size=int(self.cfg.batch_size),
            train_freq=int(self.cfg.train_freq),
            target_update_interval=int(self.cfg.target_update_interval),
            exploration_fraction=float(self.cfg.exploration_fraction),
            exploration_final_eps=float(self.cfg.exploration_final_eps),
            seed=int(self.cfg.seed),
        )

    def _save_latest_checkpoint(self, model: DQN) -> None:
        """
        Save both:
        - model weights/state
        - replay buffer (very important for DQN resume)
        """
        model.save(str(self.p["latest_model"]))
        model.save_replay_buffer(str(self.p["latest_replay"]))
        print(f"[checkpoint] saved -> {self.p['latest_model']} (+ replay buffer)")

    def _try_resume(self, env: VecEnv) -> DQN:
        """
        If a latest checkpoint exists, load it (and replay buffer if present).
        Otherwise start from scratch.
        """
        if self.p["latest_model"].exists():
            model = DQN.load(
                str(self.p["latest_model"]), env=env, device=str(self.cfg.device)
            )
            print(f"[resume] loaded model -> {self.p['latest_model']}")

            if self.p["latest_replay"].exists():
                model.load_replay_buffer(str(self.p["latest_replay"]))
                print(f"[resume] loaded replay -> {self.p['latest_replay']}")
            else:
                print(
                    "[resume] replay buffer not found (will still resume, but less stable)."
                )

            return model

        print("[resume] no checkpoint found, starting new model.")
        return self._build_model(env)

    def train(self) -> None:
        train_env = make_env(cfg=self.cfg, render_mode=None, n_envs=self.cfg.n_envs)
        eval_env = make_env(cfg=self.cfg, render_mode=None, n_envs=1)

        model = self._try_resume(train_env)

        best_mean_reward = float("-inf")
        trained = int(model.num_timesteps)

        # We train in chunks so we can checkpoint/evaluate periodically.
        chunk = max(
            10_000, min(self.cfg.checkpoint_every_steps, self.cfg.eval_every_steps)
        )

        try:
            while trained < int(self.cfg.total_timesteps):
                step = min(int(chunk), int(self.cfg.total_timesteps) - trained)

                model.learn(
                    total_timesteps=step,
                    reset_num_timesteps=False,
                    tb_log_name=str(self.cfg.run_name),
                )
                trained = int(model.num_timesteps)

                # Periodic checkpoint for resume
                if trained % int(self.cfg.checkpoint_every_steps) < step:
                    self._save_latest_checkpoint(model)

                # Periodic evaluation for "best model"
                if trained % int(self.cfg.eval_every_steps) < step:
                    mean_reward, std_reward = evaluate_policy(
                        model,
                        eval_env,
                        n_eval_episodes=int(self.cfg.eval_episodes),
                        deterministic=True,
                        render=False,
                    )
                    print(
                        f"[eval] steps={trained:,} mean_reward={mean_reward:.2f} +/- {std_reward:.2f} "
                        f"best={best_mean_reward:.2f}"
                    )
                    if mean_reward > best_mean_reward:
                        best_mean_reward = mean_reward
                        model.save(str(self.p["best_model"]))
                        print(f"[save] new best -> {self.p['best_model']}")

        except KeyboardInterrupt:
            print("\n[interrupt] Ctrl+C detected. Saving checkpoint for resume...")
            self._save_latest_checkpoint(model)
            train_env.close()
            eval_env.close()
            return

        # Save final model
        model.save(str(self.p["final_model"]))
        print(f"[save] final -> {self.p['final_model']}")

        train_env.close()
        eval_env.close()


def run(cfg: Config) -> None:
    p = paths(cfg)
    print(f"Checking for models in {p['models_dir']}")
    print(f"Best model: {p['best_model']}")
    print(f"Final model: {p['final_model']}")
    print(f"Latest model: {p['latest_model']}")

    # For demo, prefer best model, else final, else latest
    if p["best_model"].exists():
        model_path = p["best_model"]
    elif p["final_model"].exists():
        model_path = p["final_model"]
    elif p["latest_model"].exists():
        model_path = p["latest_model"]
    else:
        raise FileNotFoundError("No saved model found. Run training first.")

    env = make_env(cfg=cfg, render_mode="human", n_envs=1)
    model = DQN.load(str(model_path), env=env, device=str(cfg.device))

    obs = env.reset()
    episodes_done = 0
    ep_reward = 0.0

    while episodes_done < int(cfg.run_episodes):
        action, _ = model.predict(obs, deterministic=bool(cfg.run_deterministic))
        obs, reward, dones, _infos = env.step(action)

        ep_reward += float(reward[0])
        if bool(dones[0]):
            episodes_done += 1
            print(f"[run] episode={episodes_done} reward={ep_reward:.2f}")
            ep_reward = 0.0
            obs = env.reset()

    env.close()


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"train", "run"}:
        raise ValueError("Usage: python seaquest_dqn.py [train|run]")

    if sys.argv[1] == "train":
        SeaquestTrainer(CFG).train()
    else:
        run(CFG)


if __name__ == "__main__":
    main()
