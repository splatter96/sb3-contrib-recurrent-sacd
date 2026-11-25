#!/usr/bin/env python3
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


class ReducedMiniGridActionWrapper(gym.ActionWrapper):
    """Keep only left, right and forward actions (0, 1, 2)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = spaces.Discrete(3)

    def action(self, action):
        return int(action)


def make_env(size, random_length, reduced_actions, render_mode=None):
    env = MemoryEnv(size=size, random_length=random_length, render_mode=render_mode)

    if reduced_actions:
        env = ReducedMiniGridActionWrapper(env)

    # Keep only the partially observable symbolic view, one-hot encode it,
    # then flatten to a vector for MlpLstmPolicy.
    env = OneHotPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    env = gym.wrappers.FlattenObservation(env)
    env = Monitor(env)
    return env


def evaluate_random(size, random_length, reduced_actions, n_episodes, seed):
    env = make_env(size, random_length, reduced_actions)
    returns, successes, lengths = [], [], []

    for ep in range(n_episodes):
        s = seed + ep
        _, _ = env.reset(seed=s)
        env.action_space.seed(s)

        terminated = truncated = False
        ret = 0.0
        steps = 0

        while not (terminated or truncated):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            ret += float(reward)
            steps += 1

        returns.append(ret)
        successes.append(float(ret > 0.0))
        lengths.append(steps)

    env.close()
    return (
        np.asarray(returns, dtype=np.float32),
        np.asarray(successes, dtype=np.float32),
        np.asarray(lengths, dtype=np.float32),
    )


def evaluate_model(
    model,
    size,
    random_length,
    reduced_actions,
    n_episodes,
    seed,
    reset_memory_every_step=False,
):
    env = make_env(size, random_length, reduced_actions)
    returns, successes, lengths = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)

        state = None
        episode_start = np.ones((1,), dtype=bool)
        terminated = truncated = False
        ret = 0.0
        steps = 0

        while not (terminated or truncated):
            if reset_memory_every_step:
                state = None
                episode_start[:] = True

            action, state = model.predict(
                obs,
                state=state,
                episode_start=episode_start,
                deterministic=True,
            )

            obs, reward, terminated, truncated, _ = env.step(int(action))
            ret += float(reward)
            steps += 1
            episode_start[:] = False

        returns.append(ret)
        successes.append(float(ret > 0.0))
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


def train_model(size, args):
    env = make_env(size, args.random_length, args.reduced_actions)

    model = RecurrentSACD(
        "MlpLstmPolicy",
        env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
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
        f"\nTraining MemoryEnv(size={size}, random_length={args.random_length}) "
        f"for {args.timesteps:,} steps"
    )
    print(f"Observation shape: {env.observation_space.shape}")
    print(f"Actions: {env.action_space.n}")
    print(f"Max episode steps: {5 * size * size}")

    model.learn(total_timesteps=args.timesteps, progress_bar=args.progress_bar)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_random" if args.random_length else ""
    action_suffix = "_a3" if args.reduced_actions else "_a7"
    path = args.model_dir / f"recurrent_sacd_minigrid_memory_s{size}{suffix}{action_suffix}"
    model.save(path)
    print(f"Saved model to: {path}.zip")

    env.close()
    return model


def render_episode(model, size, args):
    env = make_env(
        size,
        args.random_length,
        args.reduced_actions,
        render_mode="human",
    )
    obs, _ = env.reset(seed=args.seed + 20000)
    state = None
    episode_start = np.ones((1,), dtype=bool)
    terminated = truncated = False

    while not (terminated or truncated):
        action, state = model.predict(
            obs,
            state=state,
            episode_start=episode_start,
            deterministic=True,
        )
        obs, _, terminated, truncated, _ = env.step(int(action))
        episode_start[:] = False

    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RecurrentSACD on MiniGrid Memory."
    )
    parser.add_argument("--size", type=int, nargs="+", default=[7])
    parser.add_argument(
        "--random-length",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--reduced-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use only left/right/forward. Use --no-reduced-actions for all 7 actions.",
    )
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--burn-in-length", type=int, default=64)
    parser.add_argument("--lstm-hidden-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument(
        "--target-entropy",
        type=float,
        default=0.25,
        help="Try 0.0 as an ablation; modest positive entropy helps sparse exploration.",
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
        default=Path("minigrid_memory_models"),
    )
    parser.add_argument("--render", action="store_true")

    args = parser.parse_args()

    for size in args.size:
        if size < 7 or size % 2 == 0:
            parser.error("MiniGrid Memory sizes must be odd integers >= 7.")

    print("MiniGrid Memory benchmark")
    print("=========================")
    print(f"sizes:            {args.size}")
    print(f"random_length:    {args.random_length}")
    print(f"reduced_actions:  {args.reduced_actions}")
    print(f"timesteps:        {args.timesteps:,} per size")
    print(f"eval episodes:    {args.eval_episodes}")
    print(f"sequence length:  {args.sequence_length}")
    print(f"burn-in length:   {args.burn_in_length}")
    print(f"target entropy:   {args.target_entropy}")

    results = []
    first_model = None

    for i, size in enumerate(args.size):
        model = train_model(size, args)
        if i == 0:
            first_model = model

        eval_seed = args.seed + 10000

        rr, rs, rl = evaluate_random(
            size, args.random_length, args.reduced_actions,
            args.eval_episodes, eval_seed
        )
        mr, ms, ml = evaluate_model(
            model, size, args.random_length, args.reduced_actions,
            args.eval_episodes, eval_seed, False
        )
        zr, zs, zl = evaluate_model(
            model, size, args.random_length, args.reduced_actions,
            args.eval_episodes, eval_seed, True
        )

        print(f"\nEvaluation: MiniGrid Memory(size={size}, random_length={args.random_length})")
        print("-" * 112)
        print_stats("Random policy", rr, rs, rl)
        print_stats("RecurrentSACD", mr, ms, ml)
        print_stats("Same policy, LSTM reset", zr, zs, zl)

        success_adv = ms.mean() - zs.mean()
        return_adv = mr.mean() - zr.mean()

        print(
            f"\nMemory advantage: "
            f"{100.0 * success_adv:+.2f} percentage points success, "
            f"{return_adv:+.3f} mean return"
        )

        results.append(
            (size, rs.mean(), ms.mean(), zs.mean(), mr.mean(), zr.mean(), success_adv)
        )

    if len(results) > 1:
        print("\n\nMiniGrid Memory summary")
        print("=" * 100)
        print(
            f"{'size':>5s} | {'random succ.':>12s} | {'RSACD succ.':>12s} | "
            f"{'reset succ.':>12s} | {'RSACD ret.':>11s} | {'reset ret.':>11s}"
        )
        print("-" * 100)
        for size, r_s, m_s, z_s, m_r, z_r, _ in results:
            print(
                f"{size:5d} | {100*r_s:11.2f}% | {100*m_s:11.2f}% | "
                f"{100*z_s:11.2f}% | {m_r:+11.3f} | {z_r:+11.3f}"
            )

    if args.render and first_model is not None:
        render_episode(first_model, args.size[0], args)


if __name__ == "__main__":
    main()
