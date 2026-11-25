#!/usr/bin/env python3
"""Train/evaluate RecurrentSACD on a fixed-start MiniGrid Memory task.

This is a controlled variant of MiniGrid's MemoryEnv.

Standard MemoryEnv randomizes the agent's initial x position along the hallway.
That means larger maps test not only longer memory, but also whether the agent
actively returns to the beginning to acquire the cue.

FixedStartMemoryEnv keeps the original map generation, cue identity, and target
randomization, but always starts the agent:

    position  = (1, height // 2)
    direction = 3  # north

The cue is located directly in front of the agent at:

    (1, height // 2 - 1)

Therefore the cue is visible immediately at episode start. Increasing `size`
primarily increases the time/distance between cue observation and the final
memory-dependent branch choice.

The script compares:
  1. random policy
  2. RecurrentSACD
  3. the same trained policy with its LSTM reset every step

Examples
--------
S13:
    python evaluate_recurrent_sacd_minigrid_fixed_start.py \
        --size 13 \
        --timesteps 500000 \
        --eval-episodes 200 \
        --burn-in-length 128

Multiple sizes:
    python evaluate_recurrent_sacd_minigrid_fixed_start.py \
        --size 7 11 13 \
        --timesteps 500000 \
        --eval-episodes 200

Render one trained episode:
    python evaluate_recurrent_sacd_minigrid_fixed_start.py \
        --size 13 \
        --render
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from minigrid.envs.memory import MemoryEnv
from minigrid.wrappers import ImgObsWrapper, OneHotPartialObsWrapper
from stable_baselines3.common.monitor import Monitor

from sb3_contrib.sacd_recurrent.sacd_recurrent import RecurrentSACD


class FixedStartMemoryEnv(MemoryEnv):
    """MemoryEnv with a deterministic cue-visible starting state."""

    def _gen_grid(self, width: int, height: int) -> None:
        # Generate the normal MemoryEnv:
        #   - random cue object (key/ball)
        #   - random final object arrangement
        #   - normal success/failure positions
        super()._gen_grid(width, height)

        # The cue is placed by MemoryEnv at:
        #     (1, height // 2 - 1)
        #
        # Start immediately below it.
        self.agent_pos = np.array((1, height // 2))

        # MiniGrid directions:
        #   0 = east
        #   1 = south
        #   2 = west
        #   3 = north
        #
        # Face the cue so it is visible in the very first observation.
        self.agent_dir = 3


class ReducedMiniGridActionWrapper(gym.ActionWrapper):
    """Restrict MiniGrid to left, right, and forward."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = spaces.Discrete(3)

    def action(self, action):
        # MiniGrid already maps 0,1,2 to left,right,forward.
        return int(action)


def make_env(
    size: int,
    reduced_actions: bool,
    render_mode: str | None = None,
) -> gym.Env:
    env = FixedStartMemoryEnv(
        size=size,
        random_length=False,
        render_mode=render_mode,
    )

    if reduced_actions:
        env = ReducedMiniGridActionWrapper(env)

    # Keep only the partially observable symbolic image.
    env = OneHotPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    env = gym.wrappers.FlattenObservation(env)

    # Monitor keeps standard SB3 episode statistics.
    env = Monitor(env)
    return env


def evaluate_random(
    size: int,
    reduced_actions: bool,
    n_episodes: int,
    seed: int,
):
    env = make_env(size, reduced_actions)

    returns = []
    successes = []
    lengths = []

    for episode in range(n_episodes):
        episode_seed = seed + episode
        _, _ = env.reset(seed=episode_seed)
        env.action_space.seed(episode_seed)

        terminated = False
        truncated = False
        episode_return = 0.0
        steps = 0

        while not (terminated or truncated):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)

            episode_return += float(reward)
            steps += 1

        returns.append(episode_return)
        successes.append(float(episode_return > 0.0))
        lengths.append(steps)

    env.close()

    return (
        np.asarray(returns, dtype=np.float32),
        np.asarray(successes, dtype=np.float32),
        np.asarray(lengths, dtype=np.float32),
    )


def evaluate_model(
    model: RecurrentSACD,
    size: int,
    reduced_actions: bool,
    n_episodes: int,
    seed: int,
    reset_memory_every_step: bool = False,
):
    env = make_env(size, reduced_actions)

    returns = []
    successes = []
    lengths = []

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)

        lstm_state = None
        episode_start = np.ones((1,), dtype=bool)

        terminated = False
        truncated = False
        episode_return = 0.0
        steps = 0

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
            steps += 1
            episode_start[:] = False

        returns.append(episode_return)
        successes.append(float(episode_return > 0.0))
        lengths.append(steps)

    env.close()

    return (
        np.asarray(returns, dtype=np.float32),
        np.asarray(successes, dtype=np.float32),
        np.asarray(lengths, dtype=np.float32),
    )


def print_stats(name, returns, successes, lengths):
    print(
        f"{name:28s} | "
        f"return = {returns.mean():6.3f} ± {returns.std():5.3f} | "
        f"success = {100.0 * successes.mean():6.2f}% | "
        f"steps = {lengths.mean():7.2f} ± {lengths.std():6.2f}"
    )


def train_model(size: int, args: argparse.Namespace) -> RecurrentSACD:
    env = make_env(
        size=size,
        reduced_actions=args.reduced_actions,
    )

    model = RecurrentSACD(
        "MlpLstmPolicy",
        env,

        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,

        # Number of replay sequences per gradient update.
        batch_size=args.batch_size,

        tau=0.005,
        gamma=args.gamma,
        train_freq=1,
        gradient_steps=args.gradient_steps,

        ent_coef="auto_0.1",
        target_entropy=args.target_entropy,

        sequence_length=args.sequence_length,
        burn_in_length=args.burn_in_length,

        policy_kwargs=dict(
            lstm_hidden_size=args.lstm_hidden_size,
            n_lstm_layers=1,
            net_arch=dict(
                pi=[256, 256],
                qf=[256, 256],
            ),
        ),

        seed=args.seed,
        verbose=args.verbose,
        device=args.device,
    )

    print(
        f"\nTraining FixedStartMemoryEnv(size={size}) "
        f"for {args.timesteps:,} environment steps..."
    )
    print(f"Observation shape: {env.observation_space.shape}")
    print(f"Action count:      {env.action_space.n}")
    print(f"Max episode steps: {5 * size * size}")

    # Sanity check: verify fixed starting state on one reset.
    env.reset(seed=args.seed)
    base_env = env.unwrapped
    print(f"Initial position:  {tuple(base_env.agent_pos)}")
    print(f"Initial direction: {base_env.agent_dir} (3 = north)")

    model.learn(
        total_timesteps=args.timesteps,
        progress_bar=args.progress_bar,
    )

    args.model_dir.mkdir(parents=True, exist_ok=True)

    action_suffix = "_a3" if args.reduced_actions else "_a7"
    model_path = (
        args.model_dir
        / f"recurrent_sacd_minigrid_fixed_start_s{size}{action_suffix}"
    )
    model.save(model_path)

    print(f"Saved model to: {model_path}.zip")

    env.close()
    return model


def run_one_size(size: int, args: argparse.Namespace):
    model = train_model(size, args)

    eval_seed = args.seed + 10_000

    random_returns, random_successes, random_lengths = evaluate_random(
        size=size,
        reduced_actions=args.reduced_actions,
        n_episodes=args.eval_episodes,
        seed=eval_seed,
    )

    recurrent_returns, recurrent_successes, recurrent_lengths = evaluate_model(
        model=model,
        size=size,
        reduced_actions=args.reduced_actions,
        n_episodes=args.eval_episodes,
        seed=eval_seed,
        reset_memory_every_step=False,
    )

    reset_returns, reset_successes, reset_lengths = evaluate_model(
        model=model,
        size=size,
        reduced_actions=args.reduced_actions,
        n_episodes=args.eval_episodes,
        seed=eval_seed,
        reset_memory_every_step=True,
    )

    print(f"\nEvaluation: FixedStart MiniGrid Memory(size={size})")
    print("-" * 112)

    print_stats(
        "Random policy",
        random_returns,
        random_successes,
        random_lengths,
    )
    print_stats(
        "RecurrentSACD",
        recurrent_returns,
        recurrent_successes,
        recurrent_lengths,
    )
    print_stats(
        "Same policy, LSTM reset",
        reset_returns,
        reset_successes,
        reset_lengths,
    )

    success_advantage = (
        recurrent_successes.mean() - reset_successes.mean()
    )
    return_advantage = (
        recurrent_returns.mean() - reset_returns.mean()
    )

    print(
        f"\nMemory advantage: "
        f"{100.0 * success_advantage:+.2f} percentage points success, "
        f"{return_advantage:+.3f} mean return"
    )

    return {
        "size": size,
        "random_success": float(random_successes.mean()),
        "recurrent_success": float(recurrent_successes.mean()),
        "reset_success": float(reset_successes.mean()),
        "random_return": float(random_returns.mean()),
        "recurrent_return": float(recurrent_returns.mean()),
        "reset_return": float(reset_returns.mean()),
        "recurrent_steps": float(recurrent_lengths.mean()),
        "reset_steps": float(reset_lengths.mean()),
        "memory_advantage": float(success_advantage),
        "model": model,
    }


def print_summary(results):
    if len(results) <= 1:
        return

    print("\n\nFixed-start MiniGrid Memory summary")
    print("=" * 106)
    print(
        f"{'size':>5s} | "
        f"{'random succ.':>12s} | "
        f"{'RSACD succ.':>12s} | "
        f"{'reset succ.':>12s} | "
        f"{'RSACD ret.':>11s} | "
        f"{'reset ret.':>11s} | "
        f"{'memory adv.':>11s}"
    )
    print("-" * 106)

    for r in results:
        print(
            f"{r['size']:5d} | "
            f"{100.0 * r['random_success']:11.2f}% | "
            f"{100.0 * r['recurrent_success']:11.2f}% | "
            f"{100.0 * r['reset_success']:11.2f}% | "
            f"{r['recurrent_return']:+11.3f} | "
            f"{r['reset_return']:+11.3f} | "
            f"{100.0 * r['memory_advantage']:+10.2f}pp"
        )


def render_episode(
    model: RecurrentSACD,
    size: int,
    args: argparse.Namespace,
):
    env = make_env(
        size=size,
        reduced_actions=args.reduced_actions,
        render_mode="human",
    )

    obs, _ = env.reset(seed=args.seed + 20_000)

    lstm_state = None
    episode_start = np.ones((1,), dtype=bool)

    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, lstm_state = model.predict(
            obs,
            state=lstm_state,
            episode_start=episode_start,
            deterministic=True,
        )

        obs, _, terminated, truncated, _ = env.step(int(action))
        episode_start[:] = False

    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RecurrentSACD on fixed-start MiniGrid Memory."
    )

    parser.add_argument(
        "--size",
        type=int,
        nargs="+",
        default=[13],
        help="Odd MiniGrid map size(s), e.g. --size 7 11 13.",
    )

    parser.add_argument(
        "--reduced-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use only left/right/forward. "
            "Use --no-reduced-actions for all original MiniGrid actions."
        ),
    )

    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--burn-in-length", type=int, default=128)

    parser.add_argument("--lstm-hidden-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gradient-steps", type=int, default=1)

    parser.add_argument(
        "--target-entropy",
        type=float,
        default=0.25,
        help="Target categorical entropy. Try 0.0 as an ablation.",
    )

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
        default=Path("minigrid_fixed_start_models"),
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="Render one episode using the first trained size.",
    )

    args = parser.parse_args()

    for size in args.size:
        if size < 7 or size % 2 == 0:
            parser.error(
                "MiniGrid Memory sizes must be odd integers >= 7."
            )

    print("Fixed-start MiniGrid Memory benchmark")
    print("=====================================")
    print(f"sizes:             {args.size}")
    print(f"training steps:    {args.timesteps:,} per size")
    print(f"evaluation eps:    {args.eval_episodes}")
    print(f"burn-in length:    {args.burn_in_length}")
    print(f"sequence length:   {args.sequence_length}")
    print(f"sequence batch:    {args.batch_size}")
    print(f"target entropy:    {args.target_entropy}")
    print(f"reduced actions:   {args.reduced_actions}")

    results = []

    for size in args.size:
        results.append(run_one_size(size, args))

    print_summary(results)

    if args.render and results:
        render_episode(
            model=results[0]["model"],
            size=results[0]["size"],
            args=args,
        )


if __name__ == "__main__":
    main()
