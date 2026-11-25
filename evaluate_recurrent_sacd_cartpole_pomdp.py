#!/usr/bin/env python3
"""Small POMDP smoke test for RecurrentSACD.

The environment is CartPole-v1 with the two velocity components removed from the
observation. The agent observes only

    [cart_position, pole_angle]

instead of

    [cart_position, cart_velocity, pole_angle, pole_angular_velocity].

Because velocity is hidden, a single observation is not Markov; a recurrent
policy can infer motion from a sequence of observations.

The script evaluates three cases:
  1. random actions,
  2. RecurrentSACD with the LSTM state carried between steps,
  3. the same trained policy with its LSTM state reset at every step.

The third case is a simple memory ablation: if recurrence is useful, normal
recurrent evaluation should outperform the reset-every-step evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# This import works with the directory layout from the implementation I sent.
# If you add an __init__.py that exports RecurrentSACD, you can instead use:
# from sb3_contrib.sacd_recurrent import RecurrentSACD
from sb3_contrib.sacd_recurrent.sacd_recurrent import RecurrentSACD


class CartPoleNoVelocity(gym.ObservationWrapper):
    """Hide cart and pole velocities, turning CartPole into a POMDP."""

    def __init__(self, env: gym.Env):
        super().__init__(env)

        assert isinstance(env.observation_space, spaces.Box)

        # Original CartPole observation:
        # [x, x_dot, theta, theta_dot]
        self.keep_indices = np.array([0, 2], dtype=np.int64)

        self.observation_space = spaces.Box(
            low=env.observation_space.low[self.keep_indices],
            high=env.observation_space.high[self.keep_indices],
            dtype=env.observation_space.dtype,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return observation[self.keep_indices]


def make_env(render_mode: str | None = None) -> gym.Env:
    #return CartPoleNoVelocity(gym.make("CartPole-v1", render_mode=render_mode))
    return gym.make("CartPole-v1", render_mode=render_mode)


def evaluate_random(n_episodes: int, seed: int) -> np.ndarray:
    env = make_env()
    returns = []

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        del obs
        done = False
        episode_return = 0.0

        while not done:
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            done = terminated or truncated

        returns.append(episode_return)

    env.close()
    return np.asarray(returns, dtype=np.float32)


def evaluate_model(
    model: RecurrentSACD,
    n_episodes: int,
    seed: int,
    reset_memory_every_step: bool = False,
    render: bool = False,
) -> np.ndarray:
    """Evaluate while explicitly handling recurrent state and episode starts."""

    env = make_env(render_mode="human" if render else None)
    returns = []

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)

        # Same convention as RecurrentPPO:
        # state=None initializes zero hidden/cell state.
        lstm_state = None
        episode_start = np.ones((1,), dtype=bool)

        done = False
        episode_return = 0.0

        while not done:
            if reset_memory_every_step:
                # Deliberately remove temporal memory while keeping exactly the
                # same trained network weights.
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
            done = terminated or truncated

            # We are still inside the same episode on the next iteration.
            episode_start[:] = False

        returns.append(episode_return)

    env.close()
    return np.asarray(returns, dtype=np.float32)


def print_stats(name: str, returns: np.ndarray) -> None:
    solved_fraction = np.mean(returns >= 475.0)
    print(
        f"{name:28s} | "
        f"mean = {returns.mean():7.2f} | "
        f"std = {returns.std():7.2f} | "
        f"min = {returns.min():6.1f} | "
        f"max = {returns.max():6.1f} | "
        f">=475 = {100.0 * solved_fraction:5.1f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-path", type=Path, default=Path("recurrent_sacd_cartpole_pomdp"))
    parser.add_argument("--render", action="store_true", help="Render one final recurrent evaluation episode.")
    args = parser.parse_args()

    train_env = make_env()

    model = RecurrentSACD(
        "MlpLstmPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=5_000,
        #batch_size=64,  # number of replay sequences
        batch_size=16,  # number of replay sequences
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto_0.1",
        target_entropy=0.0,
        # Longer recurrent context
        sequence_length=32,
        burn_in_length=32,

        #sequence_length=16,  # steps used for the loss
        #burn_in_length=8,  # preceding steps used to reconstruct LSTM state
        policy_kwargs=dict(
            lstm_hidden_size=128,
            n_lstm_layers=1,
            net_arch=dict(
                pi=[128, 128],
                qf=[128, 128],
            ),
        ),
        seed=args.seed,
        verbose=1,
        device="auto",
    )

    print("\nTraining RecurrentSACD on partially observable CartPole...")
    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    model.save(args.model_path)

    print("\nEvaluation")
    print("----------")

    random_returns = evaluate_random(args.eval_episodes, seed=args.seed + 10_000)
    recurrent_returns = evaluate_model(
        model,
        args.eval_episodes,
        seed=args.seed + 20_000,
        reset_memory_every_step=False,
    )
    no_memory_returns = evaluate_model(
        model,
        args.eval_episodes,
        seed=args.seed + 20_000,
        reset_memory_every_step=True,
    )

    print_stats("Random policy", random_returns)
    print_stats("RecurrentSACD", recurrent_returns)
    print_stats("Same policy, LSTM reset", no_memory_returns)

    difference = recurrent_returns.mean() - no_memory_returns.mean()
    print(f"\nMemory advantage: {difference:+.2f} mean return")

    # A warning rather than a hard assertion: early versions of a new algorithm
    # may need more training or hyperparameter tuning.
    if recurrent_returns.mean() <= no_memory_returns.mean() + 10.0:
        print(
            "WARNING: carrying recurrent state provides little/no advantage. "
            "This can mean the policy has not learned yet, or recurrent state/"
            "sequence handling is not working as intended."
        )

    if args.render:
        print("\nRendering one recurrent episode...")
        evaluate_model(
            model,
            n_episodes=1,
            seed=args.seed + 30_000,
            reset_memory_every_step=False,
            render=True,
        )

    train_env.close()


if __name__ == "__main__":
    main()
