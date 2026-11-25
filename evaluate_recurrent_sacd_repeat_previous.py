#!/usr/bin/env python3
"""Benchmark RecurrentSACD on POPGym RepeatPrevious.

RepeatPrevious presents a sequence of card suits (Discrete(4)). The agent must
output the suit from k positions back in the environment's history.

Important POPGym convention:
    k=1  -> current observed card (no memory required)
    k=2  -> 1-step memory
    k=4  -> 3-step memory
    ...
    k=16 -> 15-step memory

The environment normalizes episode return so that:
    perfect policy: +1.0
    random policy:  approximately -0.5

This script compares:
  1. a random policy,
  2. RecurrentSACD with recurrent state carried normally,
  3. the same trained policy with its LSTM state reset at every step.

For k > 1, a successful recurrent implementation should clearly outperform
the reset-memory ablation.

Examples
--------
Single task:
    python evaluate_recurrent_sacd_repeat_previous.py --k 4

Memory-horizon sweep:
    python evaluate_recurrent_sacd_repeat_previous.py --k 1 2 4 8 16

Longer training:
    python evaluate_recurrent_sacd_repeat_previous.py \
        --k 4 8 16 --timesteps 300000 --eval-episodes 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import popgym

from sb3_contrib.sacd_recurrent.sacd_recurrent import RecurrentSACD


def make_env(k: int, num_decks: int = 1):
    """Create POPGym RepeatPrevious directly.

    The environment already has Discrete(4) observations and Discrete(4)
    actions, so no flattening/action wrapper is necessary for SAC-Discrete.
    """
    return popgym.envs.RepeatPrevious(num_decks=num_decks, k=k)


def return_to_accuracy(episode_return: float) -> float:
    """Convert RepeatPrevious's normalized signed return to prediction accuracy.

    Each scored decision contributes +1/N when correct and -1/N when wrong, so

        return = correct_fraction - wrong_fraction
               = 2 * accuracy - 1.

    Thus accuracy = (return + 1) / 2.
    """
    return float(np.clip((episode_return + 1.0) / 2.0, 0.0, 1.0))


def evaluate_random(
    k: int,
    n_episodes: int,
    seed: int,
    num_decks: int = 1,
) -> np.ndarray:
    env = make_env(k=k, num_decks=num_decks)
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
    k: int,
    n_episodes: int,
    seed: int,
    reset_memory_every_step: bool = False,
    num_decks: int = 1,
) -> np.ndarray:
    """Evaluate with explicit recurrent-state handling."""
    env = make_env(k=k, num_decks=num_decks)
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
                # Deliberately remove temporal memory while preserving the exact
                # same trained weights.
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

            # Still inside the same episode on the next iteration.
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
    k: int,
    args: argparse.Namespace,
) -> RecurrentSACD:
    train_env = make_env(k=k, num_decks=args.num_decks)

    model = RecurrentSACD(
        "MlpLstmPolicy",
        train_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,

        # Number of replay sequences per gradient step.
        batch_size=args.batch_size,

        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,

        # The high 0.98*log(|A|) target was too stochastic for the earlier
        # CartPole experiment. Keep the corrected setting here.
        ent_coef="auto_0.1",
        target_entropy=0.0,

        # Sequence settings that worked substantially better in the CartPole
        # POMDP experiment.
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
        f"\nTraining k={k} "
        f"(effective memory delay = {max(0, k - 1)} step(s), "
        f"{args.timesteps:,} environment steps)..."
    )

    model.learn(
        total_timesteps=args.timesteps,
        progress_bar=args.progress_bar,
    )

    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / f"recurrent_sacd_repeat_previous_k{k}"
    model.save(model_path)
    print(f"Saved model to: {model_path}.zip")

    train_env.close()
    return model


def run_one_k(k: int, args: argparse.Namespace) -> dict[str, float]:
    model = train_model(k, args)

    eval_seed = args.seed + 10_000

    random_returns = evaluate_random(
        k=k,
        n_episodes=args.eval_episodes,
        seed=eval_seed,
        num_decks=args.num_decks,
    )

    recurrent_returns = evaluate_model(
        model=model,
        k=k,
        n_episodes=args.eval_episodes,
        seed=eval_seed,
        reset_memory_every_step=False,
        num_decks=args.num_decks,
    )

    reset_returns = evaluate_model(
        model=model,
        k=k,
        n_episodes=args.eval_episodes,
        seed=eval_seed,
        reset_memory_every_step=True,
        num_decks=args.num_decks,
    )

    print(f"\nEvaluation: RepeatPrevious(k={k})")
    print("-" * 96)
    print_stats("Random policy", random_returns)
    print_stats("RecurrentSACD", recurrent_returns)
    print_stats("Same policy, LSTM reset", reset_returns)

    memory_advantage = recurrent_returns.mean() - reset_returns.mean()
    recurrent_accuracy = np.mean(
        [return_to_accuracy(r) for r in recurrent_returns]
    )
    reset_accuracy = np.mean(
        [return_to_accuracy(r) for r in reset_returns]
    )

    print(
        f"\nMemory advantage: {memory_advantage:+.3f} mean return "
        f"({100.0 * (recurrent_accuracy - reset_accuracy):+.2f} percentage points accuracy)"
    )

    if k > 1 and recurrent_returns.mean() <= reset_returns.mean() + 0.05:
        print(
            "WARNING: recurrent state provides little advantage for this k. "
            "Consider more training, a longer burn-in/sequence, or inspecting "
            "the replay-state reconstruction."
        )

    return {
        "k": float(k),
        "delay": float(max(0, k - 1)),
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

    print("\n\nMemory-horizon summary")
    print("=" * 105)
    print(
        f"{'k':>4s} | {'delay':>5s} | "
        f"{'random ret.':>11s} | {'RSACD ret.':>11s} | {'reset ret.':>11s} | "
        f"{'RSACD acc.':>10s} | {'reset acc.':>10s} | {'memory adv.':>11s}"
    )
    print("-" * 105)

    for row in results:
        print(
            f"{int(row['k']):4d} | "
            f"{int(row['delay']):5d} | "
            f"{row['random_return']:+11.3f} | "
            f"{row['recurrent_return']:+11.3f} | "
            f"{row['reset_return']:+11.3f} | "
            f"{100.0 * row['recurrent_accuracy']:9.2f}% | "
            f"{100.0 * row['reset_accuracy']:9.2f}% | "
            f"{row['memory_advantage']:+11.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark RecurrentSACD on POPGym RepeatPrevious."
    )

    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[4],
        help=(
            "POPGym RepeatPrevious k value(s). "
            "Examples: --k 4 or --k 1 2 4 8 16. "
            "Effective memory delay is k-1."
        ),
    )
    parser.add_argument("--num-decks", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=5_000)

    # These are deliberately the improved recurrent settings from the
    # partially observable CartPole experiment.
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
        default=Path("repeat_previous_models"),
    )

    args = parser.parse_args()

    for k in args.k:
        if k < 1:
            parser.error("All --k values must be >= 1.")
        if 52 * args.num_decks <= k:
            parser.error(
                f"k={k} is too large for num_decks={args.num_decks}; "
                f"it must be less than {52 * args.num_decks}."
            )

    print("POPGym RepeatPrevious benchmark")
    print("===============================")
    print(f"k values:          {args.k}")
    print(f"training steps:    {args.timesteps:,} per k")
    print(f"evaluation eps:    {args.eval_episodes}")
    print(f"sequence length:   {args.sequence_length}")
    print(f"burn-in length:    {args.burn_in_length}")
    print(f"sequence batch:    {args.batch_size}")
    print(f"target entropy:    0.0")

    results = [run_one_k(k, args) for k in args.k]
    print_summary(results)


if __name__ == "__main__":
    main()
