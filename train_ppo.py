"""Train MaskablePPO agent for YouBlewIt environments."""

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

import gym_env  # noqa: F401 - registers environments

if TYPE_CHECKING:
    from gym_env.you_blew_it import YouBlewItEnv
    from gym_env.you_blew_it_v2 import YouBlewItV2Env


def mask_fn(env: gym.Env["NDArray[np.int64]", int]) -> "NDArray[np.bool_]":
    """Extract action mask from environment."""
    unwrapped: YouBlewItEnv | YouBlewItV2Env = env.unwrapped  # type: ignore[assignment]
    return unwrapped.action_masks()


def make_env(env_id: str) -> gym.Env["NDArray[np.int64]", int]:
    """Create and wrap environment with action masking."""
    env = gym.make(env_id)
    return ActionMasker(env, mask_fn)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MaskablePPO on YouBlewIt")
    parser.add_argument(
        "--env",
        type=str,
        default="YouBlewIt-v1",
        choices=["YouBlewIt-v1", "YouBlewIt-v2"],
        help="Environment to train on",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Total timesteps to train",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu, mps)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models",
        help="Directory to save models",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10_000,
        help="Evaluation frequency in timesteps",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=50_000,
        help="Checkpoint frequency in timesteps",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable WandB logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="youblewitgym",
        help="WandB project name",
    )
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    env_name = args.env.replace("-", "_").lower()

    # Create training environment
    train_env = DummyVecEnv([lambda: make_env(args.env)])

    # Create evaluation environment
    eval_env = DummyVecEnv([lambda: make_env(args.env)])

    # Set up callbacks
    callbacks = []

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir / f"best_{env_name}"),
        log_path=str(save_dir / f"logs_{env_name}"),
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_callback)

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(save_dir / f"checkpoints_{env_name}"),
        name_prefix=f"ppo_{env_name}",
    )
    callbacks.append(checkpoint_callback)

    # Initialize WandB if requested
    if args.wandb:
        import wandb
        from wandb.integration.sb3 import WandbCallback

        run = wandb.init(
            project=args.wandb_project,
            config={
                "env": args.env,
                "total_timesteps": args.total_timesteps,
                "algorithm": "MaskablePPO",
            },
            sync_tensorboard=True,
        )
        callbacks.append(WandbCallback(verbose=2))

    # Create model with hyperparameters tuned for this game
    tensorboard_log = str(save_dir / "tensorboard") if args.wandb else None
    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        device=args.device,
        tensorboard_log=tensorboard_log,
    )

    # Check if progress bar dependencies are available
    try:
        import rich  # noqa: F401  # type: ignore[import-not-found]
        import tqdm  # noqa: F401

        progress_bar = True
    except ImportError:
        progress_bar = False

    print(f"Training MaskablePPO on {args.env} for {args.total_timesteps} timesteps")
    print(f"Device: {model.device}")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=progress_bar,
    )

    # Save final model
    final_path = save_dir / f"final_{env_name}"
    model.save(str(final_path))
    print(f"Saved final model to {final_path}")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
