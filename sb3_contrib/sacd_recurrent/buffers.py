from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize

from sb3_contrib.sacd_recurrent.type_aliases import RecurrentReplayBufferSamples


class _SequenceReplayMixin:
    """Sequence sampling shared by ndarray and Dict replay buffers.

    The buffer intentionally does not store LSTM states. Stored hidden states become
    stale as soon as the recurrent networks are updated. Instead, sampled sequences
    use truncated BPTT with an optional burn-in prefix.
    """

    sequence_length: int
    burn_in_length: int

    def _sequence_init(self, sequence_length: int, burn_in_length: int) -> None:
        if sequence_length < 1:
            raise ValueError("sequence_length must be >= 1")
        if burn_in_length < 0:
            raise ValueError("burn_in_length must be >= 0")
        self.sequence_length = sequence_length
        self.burn_in_length = burn_in_length

    @property
    def total_sequence_length(self) -> int:
        return self.burn_in_length + self.sequence_length

    def _chronological_indices(self) -> np.ndarray:
        """Return valid replay indices in insertion order (oldest -> newest)."""
        size = self.size()
        if size == 0:
            return np.empty((0,), dtype=np.int64)
        if not self.full:
            return np.arange(size, dtype=np.int64)
        return np.concatenate(
            (
                np.arange(self.pos, self.buffer_size, dtype=np.int64),
                np.arange(0, self.pos, dtype=np.int64),
            )
        )

    def _sample_sequence_indices(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        chronological = self._chronological_indices()
        total_len = self.total_sequence_length
        if len(chronological) < total_len:
            raise ValueError(
                f"Not enough replay data for one recurrent sequence: have {len(chronological)} transitions, "
                f"need at least burn_in_length + sequence_length = {total_len}."
            )

        max_start = len(chronological) - total_len
        starts = np.random.randint(0, max_start + 1, size=batch_size)
        offsets = np.arange(total_len, dtype=np.int64)[None, :]
        replay_indices = chronological[starts[:, None] + offsets]
        env_indices = np.random.randint(0, self.n_envs, size=batch_size, dtype=np.int64)
        return replay_indices, env_indices

    @staticmethod
    def _flatten_bt(array: np.ndarray) -> np.ndarray:
        """[batch, time, ...] -> [batch*time, ...]."""
        return array.reshape((-1, *array.shape[2:]))

    def _make_episode_starts(self, raw_dones: np.ndarray) -> np.ndarray:
        """Create start flags for the T+1 observations of each sampled sequence.

        We deliberately reset the recurrent state at the beginning of each sampled
        sequence (truncated BPTT). Internal episode boundaries are reconstructed from
        the raw VecEnv done signal.
        """
        batch_size, total_len = raw_dones.shape
        episode_starts = np.zeros((batch_size, total_len + 1), dtype=np.float32)
        episode_starts[:, 0] = 1.0
        episode_starts[:, 1:] = raw_dones.astype(np.float32)
        return episode_starts


class RecurrentReplayBuffer(_SequenceReplayMixin, ReplayBuffer):
    """Replay buffer that samples contiguous transition sequences for recurrent SAC-D.

    Notes
    -----
    ``optimize_memory_usage=True`` is intentionally unsupported: recurrent training
    needs an explicit final next-observation for each sampled sequence.

    Time-limit terminations are treated as terminal for the Bellman target. This is
    deliberate because the sequence representation uses the next replay observation
    after an environment reset at an internal episode boundary. Supporting SB3's
    usual timeout bootstrapping requires carrying the terminal observation through
    the recurrent target state separately.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = False,
        sequence_length: int = 16,
        burn_in_length: int = 8,
    ):
        if optimize_memory_usage:
            raise ValueError("RecurrentReplayBuffer does not support optimize_memory_usage=True")
        self._sequence_init(sequence_length, burn_in_length)
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=False,
            handle_timeout_termination=handle_timeout_termination,
        )

    def sample(self, batch_size: int, env: VecNormalize | None = None) -> RecurrentReplayBufferSamples:  # type: ignore[override]
        indices, env_indices = self._sample_sequence_indices(batch_size)
        batch_rows = np.arange(batch_size)[:, None]
        env_rows = env_indices[:, None]

        obs = self.observations[indices, env_rows]
        last_next_obs = self.next_observations[indices[:, -1], env_indices]
        obs_plus_one = np.concatenate((obs, last_next_obs[:, None, ...]), axis=1)

        actions = self.actions[indices, env_rows]
        rewards = self.rewards[indices, env_rows]
        raw_dones = self.dones[indices, env_rows]
        episode_starts = self._make_episode_starts(raw_dones)

        # See class docstring: time-limit resets are treated as terminal here.
        dones = raw_dones.astype(np.float32)

        obs_flat = self._flatten_bt(obs_plus_one)
        actions_flat = self._flatten_bt(actions)
        rewards_flat = rewards.reshape(-1, 1)
        dones_flat = dones.reshape(-1, 1)
        episode_starts_flat = episode_starts.reshape(-1)

        obs_flat = self._normalize_obs(obs_flat, env)
        rewards_flat = self._normalize_reward(rewards_flat, env)

        return RecurrentReplayBufferSamples(
            observations=self.to_torch(obs_flat),  # type: ignore[arg-type]
            actions=self.to_torch(actions_flat),
            rewards=self.to_torch(rewards_flat),
            dones=self.to_torch(dones_flat),
            episode_starts=self.to_torch(episode_starts_flat),
            batch_size=batch_size,
            sequence_length=self.sequence_length,
            burn_in_length=self.burn_in_length,
        )


class RecurrentDictReplayBuffer(_SequenceReplayMixin, DictReplayBuffer):
    """Dict-observation equivalent of :class:`RecurrentReplayBuffer`."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = False,
        sequence_length: int = 16,
        burn_in_length: int = 8,
    ):
        if optimize_memory_usage:
            raise ValueError("RecurrentDictReplayBuffer does not support optimize_memory_usage=True")
        self._sequence_init(sequence_length, burn_in_length)
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=False,
            handle_timeout_termination=handle_timeout_termination,
        )

    def sample(self, batch_size: int, env: VecNormalize | None = None) -> RecurrentReplayBufferSamples:  # type: ignore[override]
        indices, env_indices = self._sample_sequence_indices(batch_size)
        env_rows = env_indices[:, None]

        obs_plus_one: dict[str, np.ndarray] = {}
        for key, obs_storage in self.observations.items():
            obs = obs_storage[indices, env_rows]
            last_next_obs = self.next_observations[key][indices[:, -1], env_indices]
            obs_plus_one[key] = np.concatenate((obs, last_next_obs[:, None, ...]), axis=1)

        actions = self.actions[indices, env_rows]
        rewards = self.rewards[indices, env_rows]
        raw_dones = self.dones[indices, env_rows]
        episode_starts = self._make_episode_starts(raw_dones)
        dones = raw_dones.astype(np.float32)

        obs_flat = {key: self._flatten_bt(value) for key, value in obs_plus_one.items()}
        obs_flat = self._normalize_obs(obs_flat, env)
        assert isinstance(obs_flat, dict)

        rewards_flat = self._normalize_reward(rewards.reshape(-1, 1), env)

        return RecurrentReplayBufferSamples(
            observations={key: self.to_torch(value) for key, value in obs_flat.items()},
            actions=self.to_torch(self._flatten_bt(actions)),
            rewards=self.to_torch(rewards_flat),
            dones=self.to_torch(dones.reshape(-1, 1)),
            episode_starts=self.to_torch(episode_starts.reshape(-1)),
            batch_size=batch_size,
            sequence_length=self.sequence_length,
            burn_in_length=self.burn_in_length,
        )
