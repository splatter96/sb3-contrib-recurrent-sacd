#!/usr/bin/env python3
"""Benchmark RecurrentSACD on POPGym RepeatFirst.

RepeatFirst presents one initial card suit (Discrete(4)). The agent must keep
outputting that initial suit for the rest of the episode while observations
change to newly dealt cards.

Difficulty is controlled by num_decks:
    num_decks=1   ->  51 decisions per episode
    num_decks=8   -> 415 decisions per episode
    num_decks=16  -> 831 decisions per episode

These correspond to POPGym's Easy, Medium, and Hard variants.

The environment normalizes episode return so that:
    perfect policy: +1.0
    random 4-action policy: approximately -0.5

This script compares:
  1. a random policy,
  2. RecurrentSACD with recurrent state carried normally,
  3. the same trained policy with its LSTM state reset every step.

Examples
--------
Easy task:
    python evaluate_recurrent_sacd_repeat_first.py --num-decks 1

Easy/Medium/Hard sweep:
    python evaluate_recurrent_sacd_repeat_first.py --num-decks 1 8 16

Longer training for the harder tasks:
    python evaluate_recurrent_sacd_repeat_first.py \
        --num-decks 8 16 --timesteps 500000 --eval-episodes 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import popgym

from sb3_contrib.sacd_recurrent.sacd_recurrent import RecurrentSACD


def make_env(num_decks: int):
    """Create POPGym RepeatFirst directly."""
    return popgym.envs.RepeatFirst(num_decks=num_decks)


def episode_length(num_decks: int) -> int:
    return 52 * num_decks - 1


def return_to_accuracy(episode_return: float) -> float:
    """Convert normalized RepeatFirst return to prediction accuracy."""
    return float(np.clip((episode_return + 1.0) / 2.0, 0.0, 1.0))


def evaluate_random(
    num_decks: int,
    n_episodes: int,
    seed: int,
) -> np.ndarray:
    env = make_env(num_decks)
    returns: list[float] = []

    for episode in range(n_episodes):
        episode_seed = seed + episode
        _, _ = env.reset(seed=episode_seed)
        env.action_space.seed(episode_seed)

        terminated = False
        truncated = False
        episode_return = 0.0

        while not (terminated or truncated):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)

        returns.append(episode_return)

    env.close()
    return np.asarray(returns, dtype=np.float32)


def evaluate_model(
    model: RecurrentSACD,
    num_decks: int,
    n_episodes: int,
    seed: int,
    reset_memory_every_step: bool = False,
) -> np.ndarray:
    """Evaluate with explicit recurrent-state handling."""
    env = make_env(num_decks)
    returns: list[float] = []

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)

        lstm_state = None
        episode_start = np.ones((1,), dtype=bool)

        terminated = False
        truncated = False
        episode_return = 0.0

        while not (terminated or truncated):
            if reset_memory_every_step:
                lstm_state = None
                episode_start[:] = True

            action, lstm_state = model.predict(
                obs,
                state=lstm_state,
                episode_start=episode_start,
                deterministic=True,
            )

            obs, reward, terminated, truncated, _ = env.step(int(action))
            episode_return += float(reward)

            episode_start[:] = False

        returns.append(episode_return)

    env.close()
    return np.asarray(returns, dtype=np.float32)


def print_stats(name: str, returns: np.ndarray) -> None:
    accuracies = np.asarray([return_to_accuracy(r) for r in returns])

    print(
        f"{name:28s} | "
        f"return mean = {returns.mean():+7.3f} | "
        f"std = {returns.std():6.3f} | "
        f"min = {returns.min():+6.3f} | "
        f"max = {returns.max():+6.3f} | "
        f"accuracy = {100.0 * accuracies.mean():6.2f}%"
    )


def train_model(
    num_decks: int,
    args: argparse.Namespace,
) -> RecurrentSACD:
    train_env = make_env(num_decks)

    model = RecurrentSACD(
        "MlpLstmPolicy",
        train_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto_0.1",
        target_entropy=0.0,
        sequence_length=args.sequence_length,
        burn_in_length=args.burn_in_length,
        policy_kwargs=dict(
            lstm_hidden_size=args.lstm_hidden_size,
            n_lstm_layers=1,
            net_arch=dict(
                pi=[128, 128],
                qf=[128, 128],
            ),
        ),
        seed=args.seed,
        verbose=args.verbose,
        device=args.device,
    )

    print(
        f"\nTraining RepeatFirst(num_decks={num_decks}) "
        f"[episode length={episode_length(num_decks)}] "
        f"for {args.timesteps:,} environment steps..."
    )

    model.learn(
        total_timesteps=args.timesteps,
        progress_bar=args.progress_bar,
    )

    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / f"recurrent_sacd_repeat_first_decks{num_decks}"
    model.save(model_path)
    print(f"Saved model to: {model_path}.zip")

    train_env.close()
    return model


def run_one_setting(num_decks: int, args: argparse.Namespace) -> dict[str, float]:
    model = train_model(num_decks, args)
    eval_seed = args.seed + 10_000

    random_returns = evaluate_random(
        num_decks=num_decks,
        n_episodes=args.eval_episodes,
        seed=eval_seed,
    )

    recurrent_returns = evaluate_model(
        model=model,
        num_decks=num_decks,
        n_episodes=args.eval_episodes,
        seed=eval_seed,
        reset_memory_every_step=False,
    )

    reset_returns = evaluate_model(
        model=model,
        num_decks=num_decks,
        n_episodes=args.eval_episodes,
        seed=eval_seed,
        reset_memory_every_step=True,
    )

    print(
        f"\nEvaluation: RepeatFirst(num_decks={num_decks}, "
        f"episode_length={episode_length(num_decks)})"
    )
    print("-" * 100)
    print_stats("Random policy", random_returns)
    print_stats("RecurrentSACD", recurrent_returns)
    print_stats("Same policy, LSTM reset", reset_returns)

    recurrent_accuracy = np.mean([return_to_accuracy(r) for r in recurrent_returns])
    reset_accuracy = np.mean([return_to_accuracy(r) for r in reset_returns])
    memory_advantage = recurrent_returns.mean() - reset_returns.mean()

    print(
        f"\nMemory advantage: {memory_advantage:+.3f} mean return "
        f"({100.0 * (recurrent_accuracy - reset_accuracy):+.2f} "
        f"percentage points accuracy)"
    )

    return {
        "num_decks": float(num_decks),
        "episode_length": float(episode_length(num_decks)),
        "random_return": float(random_returns.mean()),
        "recurrent_return": float(recurrent_returns.mean()),
        "reset_return": float(reset_returns.mean()),
        "random_accuracy": float(
            np.mean([return_to_accuracy(r) for r in random_returns])
        ),
        "recurrent_accuracy": float(recurrent_accuracy),
        "reset_accuracy": float(reset_accuracy),
        "memory_advantage": float(memory_advantage),
    }


def print_summary(results: list[dict[str, float]]) -> None:
    if len(results) <= 1:
        return

    print("\n\nRetention-horizon summary")
    print("=" * 112)
    print(
        f"{'decks':>5s} | {'episode':>7s} | "
        f"{'random ret.':>11s} | {'RSACD ret.':>11s} | {'reset ret.':>11s} | "
        f"{'RSACD acc.':>10s} | {'reset acc.':>10s} | {'memory adv.':>11s}"
    )
    print("-" * 112)

    for row in results:
        print(
            f"{int(row['num_decks']):5d} | "
            f"{int(row['episode_length']):7d} | "
            f"{row['random_return']:+11.3f} | "
            f"{row['recurrent_return']:+11.3f} | "
            f"{row['reset_return']:+11.3f} | "
            f"{100.0 * row['recurrent_accuracy']:9.2f}% | "
            f"{100.0 * row['reset_accuracy']:9.2f}% | "
            f"{row['memory_advantage']:+11.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark RecurrentSACD on POPGym RepeatFirst."
    )

    parser.add_argument(
        "--num-decks",
        type=int,
        nargs="+",
        default=[1],
        help=(
            "Number of decks / retention horizon. "
            "POPGym uses 1=Easy, 8=Medium, 16=Hard. "
            "Example: --num-decks 1 8 16"
        ),
    )
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--burn-in-length", type=int, default=32)
    parser.add_argument("--lstm-hidden-size", type=int, default=128)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument(
        "--progress-bar",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("repeat_first_models"),
    )

    args = parser.parse_args()

    for n in args.num_decks:
        if n < 1:
            parser.error("All --num-decks values must be >= 1.")

    print("POPGym RepeatFirst benchmark")
    print("============================")
    print(f"num_decks:          {args.num_decks}")
    print(
        "episode lengths:    "
        + str([episode_length(n) for n in args.num_decks])
    )
    print(f"training steps:     {args.timesteps:,} per setting")
    print(f"evaluation eps:     {args.eval_episodes}")
    print(f"sequence length:    {args.sequence_length}")
    print(f"burn-in length:     {args.burn_in_length}")
    print(f"sequence batch:     {args.batch_size}")
    print("target entropy:     0.0")

    results = [run_one_setting(n, args) for n in args.num_decks]
    print_summary(results)


if __name__ == "__main__":
    main()
