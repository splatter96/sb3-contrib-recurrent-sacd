from __future__ import annotations

from copy import deepcopy
from typing import Any, ClassVar, TypeVar

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import (
    GymEnv,
    MaybeCallback,
    RolloutReturn,
    Schedule,
    TrainFreq,
    TrainFrequencyUnit,
)
from stable_baselines3.common.utils import get_parameters_by_name, obs_as_tensor, polyak_update, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from torch.nn import functional as F

from sb3_contrib.sacd_recurrent.buffers import RecurrentDictReplayBuffer, RecurrentReplayBuffer
from sb3_contrib.sacd_recurrent.policies import (
    CnnLstmPolicy,
    LSTMState,
    MlpLstmPolicy,
    MultiInputLstmPolicy,
    RecurrentSACDPolicy,
)
from sb3_contrib.sacd_recurrent.type_aliases import RecurrentReplayBufferSamples

SelfRecurrentSACD = TypeVar("SelfRecurrentSACD", bound="RecurrentSACD")


def _slice_flat_obs(obs, batch_size: int, total_steps: int, start: int, end: int):
    """Slice a batch-major flattened sequence and flatten it again."""
    if isinstance(obs, dict):
        return {
            key: value.reshape(batch_size, total_steps, *value.shape[1:])[:, start:end].reshape(
                batch_size * (end - start), *value.shape[1:]
            )
            for key, value in obs.items()
        }
    return obs.reshape(batch_size, total_steps, *obs.shape[1:])[:, start:end].reshape(
        batch_size * (end - start), *obs.shape[1:]
    )


def _detach_state(state: LSTMState) -> LSTMState:
    return state[0].detach(), state[1].detach()


class RecurrentSACD(OffPolicyAlgorithm):
    """Discrete Soft Actor-Critic with recurrent actor and recurrent twin-Q critic.

    This follows SB3-Contrib RecurrentPPO's recurrent-state interface during data
    collection, but uses sequence replay + burn-in for off-policy training.
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpLstmPolicy": MlpLstmPolicy,
        "CnnLstmPolicy": CnnLstmPolicy,
        "MultiInputLstmPolicy": MultiInputLstmPolicy,
    }

    policy: RecurrentSACDPolicy

    def __init__(
        self,
        policy: str | type[RecurrentSACDPolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 1_000,
        batch_size: int = 64,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int | tuple[int, str] = 1,
        gradient_steps: int = 1,
        replay_buffer_class: type[ReplayBuffer] | None = None,
        replay_buffer_kwargs: dict[str, Any] | None = None,
        optimize_memory_usage: bool = False,
        ent_coef: str | float = "auto",
        target_update_interval: int = 1,
        target_entropy: str | float = "auto",
        sequence_length: int = 16,
        burn_in_length: int = 8,
        stats_window_size: int = 100,
        max_grad_norm: float = 5.0,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
    ):
        if optimize_memory_usage:
            raise ValueError("RecurrentSACD does not support optimize_memory_usage=True")

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=None,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=False,
            n_steps=1,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            support_multi_env=True,
            seed=seed,
            use_sde=False,
            sde_support=False,
            supported_action_spaces=(spaces.Discrete,),
        )

        if replay_buffer_class is None:
            self.replay_buffer_class = (
                RecurrentDictReplayBuffer if isinstance(self.observation_space, spaces.Dict) else RecurrentReplayBuffer
            )
        self.replay_buffer_kwargs = self.replay_buffer_kwargs.copy()
        self.replay_buffer_kwargs.setdefault("sequence_length", sequence_length)
        self.replay_buffer_kwargs.setdefault("burn_in_length", burn_in_length)
        # See recurrent buffer docstring for the reason this is False.
        self.replay_buffer_kwargs.setdefault("handle_timeout_termination", False)

        self.sequence_length = sequence_length
        self.burn_in_length = burn_in_length
        self.ent_coef = ent_coef
        self.target_entropy = target_entropy
        self.target_update_interval = target_update_interval
        self.max_grad_norm = max_grad_norm

        self.log_ent_coef: th.Tensor | None = None
        self.ent_coef_tensor: th.Tensor | None = None
        self.ent_coef_optimizer: th.optim.Optimizer | None = None
        self._last_lstm_states: LSTMState | None = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        if not isinstance(self.policy, RecurrentSACDPolicy):
            raise ValueError("Policy must subclass RecurrentSACDPolicy")
        if not isinstance(self.replay_buffer, (RecurrentReplayBuffer, RecurrentDictReplayBuffer)):
            raise ValueError("RecurrentSACD requires a recurrent sequence replay buffer")

        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target
        self._last_lstm_states = self.policy.initial_state(self.n_envs)

        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])

        if self.target_entropy == "auto":
            # Positive entropy target for a categorical distribution.
            self.target_entropy = 0.98 * float(np.log(self.action_space.n))
        else:
            self.target_entropy = float(self.target_entropy)

        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_", 1)[1])
                if init_value <= 0.0:
                    raise ValueError("The initial entropy coefficient must be > 0")
            self.log_ent_coef = th.log(th.tensor([init_value], device=self.device)).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
            self.ent_coef_tensor = self.log_ent_coef.detach().exp()
        else:
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "RecurrentSACD",
        progress_bar: bool = False,
    ):
        result = super()._setup_learn(total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar)
        if reset_num_timesteps or self._last_lstm_states is None:
            self._last_lstm_states = self.policy.initial_state(self.n_envs)
        return result

    def _burn_in_states(
        self,
        replay_data: RecurrentReplayBufferSamples,
    ) -> tuple[LSTMState, LSTMState, LSTMState]:
        batch_size = replay_data.batch_size
        burn = replay_data.burn_in_length
        total_obs_steps = burn + replay_data.sequence_length + 1

        actor_state = self.actor.initial_state(batch_size, self.device)
        critic_state = self.critic.initial_state(batch_size, self.device)
        target_critic_state = self.critic_target.initial_state(batch_size, self.device)

        if burn == 0:
            return actor_state, critic_state, target_critic_state

        burn_obs = _slice_flat_obs(replay_data.observations, batch_size, total_obs_steps, 0, burn)
        burn_starts = replay_data.episode_starts.reshape(batch_size, total_obs_steps)[:, :burn].reshape(-1)

        with th.no_grad():
            _, _, actor_state = self.actor.action_log_prob(burn_obs, actor_state, burn_starts)
            _, critic_state = self.critic(burn_obs, critic_state, burn_starts)
            _, target_critic_state = self.critic_target(burn_obs, target_critic_state, burn_starts)

        return _detach_state(actor_state), _detach_state(critic_state), _detach_state(target_critic_state)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers: list[th.optim.Optimizer] = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)

        actor_losses: list[float] = []
        critic_losses: list[float] = []
        ent_coef_losses: list[float] = []
        ent_coefs: list[float] = []
        entropies: list[float] = []

        for gradient_step in range(gradient_steps):
            assert isinstance(self.replay_buffer, (RecurrentReplayBuffer, RecurrentDictReplayBuffer))
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            batch = replay_data.batch_size
            seq = replay_data.sequence_length
            burn = replay_data.burn_in_length
            total_obs_steps = burn + seq + 1

            actor_state, critic_state, target_critic_state = self._burn_in_states(replay_data)

            train_obs_plus = _slice_flat_obs(
                replay_data.observations,
                batch,
                total_obs_steps,
                burn,
                burn + seq + 1,
            )
            train_obs = _slice_flat_obs(
                replay_data.observations,
                batch,
                total_obs_steps,
                burn,
                burn + seq,
            )
            episode_starts_plus = replay_data.episode_starts.reshape(batch, total_obs_steps)[:, burn : burn + seq + 1].reshape(-1)
            episode_starts = episode_starts_plus.reshape(batch, seq + 1)[:, :-1].reshape(-1)

            actions = replay_data.actions.reshape(batch, burn + seq, -1)[:, burn:].reshape(batch * seq, -1).long()
            rewards = replay_data.rewards.reshape(batch, burn + seq, 1)[:, burn:].reshape(batch * seq, 1)
            dones = replay_data.dones.reshape(batch, burn + seq, 1)[:, burn:].reshape(batch * seq, 1)

            assert self.ent_coef_tensor is not None
            alpha = self.ent_coef_tensor.detach()

            # Actor probabilities for current and next recurrent states.  The final
            # state is only used by the target and is detached there.
            action_prob_plus, log_action_prob_plus, _ = self.actor.action_log_prob(
                train_obs_plus, actor_state, episode_starts_plus
            )
            action_prob_plus = action_prob_plus.reshape(batch, seq + 1, -1)
            log_action_prob_plus = log_action_prob_plus.reshape(batch, seq + 1, -1)

            current_action_prob = action_prob_plus[:, :-1].reshape(batch * seq, -1)
            current_log_action_prob = log_action_prob_plus[:, :-1].reshape(batch * seq, -1)
            next_action_prob = action_prob_plus[:, 1:].reshape(batch * seq, -1).detach()
            next_log_action_prob = log_action_prob_plus[:, 1:].reshape(batch * seq, -1).detach()

            # Critic target.
            with th.no_grad():
                target_q_plus, _ = self.critic_target(train_obs_plus, target_critic_state, episode_starts_plus)
                target_q_stack = th.stack(
                    [q.reshape(batch, seq + 1, -1)[:, 1:].reshape(batch * seq, -1) for q in target_q_plus],
                    dim=0,
                )
                next_min_q = target_q_stack.min(dim=0).values
                next_v = (next_action_prob * (next_min_q - alpha * next_log_action_prob)).sum(dim=-1, keepdim=True)
                target_q = rewards + (1.0 - dones) * self.gamma * next_v

            current_q_values, _ = self.critic(train_obs, critic_state, episode_starts)
            action_indices = actions[:, :1]
            critic_loss = 0.5 * sum(
                F.mse_loss(q.gather(1, action_indices), target_q) for q in current_q_values
            )
            critic_losses.append(float(critic_loss.item()))

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            th.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic.optimizer.step()

            # Actor objective. Q-values are constants with respect to the actor in
            # discrete SAC, so detach them to avoid unnecessary critic gradients.
            with th.no_grad():
                q_for_actor, _ = self.critic(train_obs, critic_state, episode_starts)
                min_q_for_actor = th.stack(q_for_actor, dim=0).min(dim=0).values

            actor_loss = (
                current_action_prob * (alpha * current_log_action_prob - min_q_for_actor)
            ).sum(dim=-1).mean()
            actor_losses.append(float(actor_loss.item()))

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            th.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor.optimizer.step()

            entropy = -(current_action_prob.detach() * current_log_action_prob.detach()).sum(dim=-1)
            entropies.append(float(entropy.mean().item()))

            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Positive target entropy: increase alpha if H(pi) < H_target.
                ent_coef_loss = (self.log_ent_coef * (entropy - self.target_entropy).detach()).mean()
                ent_coef_losses.append(float(ent_coef_loss.item()))
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
                self.ent_coef_tensor = self.log_ent_coef.detach().exp()

            ent_coefs.append(float(self.ent_coef_tensor.item()))

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/entropy", np.mean(entropies))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if ent_coef_losses:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise=None,
        learning_starts: int = 0,
        log_interval: int | None = None,
    ) -> RolloutReturn:
        del action_noise
        self.policy.set_training_mode(False)
        assert isinstance(env, VecEnv)
        assert train_freq.frequency > 0
        assert isinstance(replay_buffer, (RecurrentReplayBuffer, RecurrentDictReplayBuffer))
        assert self._last_obs is not None
        assert self._last_episode_starts is not None
        assert self._last_lstm_states is not None

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "Use step-based train_freq with multiple environments"

        callback.on_rollout_start()
        num_collected_steps = 0
        num_collected_episodes = 0
        continue_training = True
        lstm_states = deepcopy(self._last_lstm_states)

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                episode_starts = th.as_tensor(self._last_episode_starts, dtype=th.float32, device=self.device)
                policy_actions, new_lstm_states = self.policy(
                    obs_tensor, lstm_states, episode_starts, deterministic=False
                )

            if self.num_timesteps < learning_starts:
                actions = np.array([self.action_space.sample() for _ in range(env.num_envs)])
            else:
                actions = policy_actions.cpu().numpy()
            buffer_actions = actions

            new_obs, rewards, dones, infos = env.step(actions)
            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            callback.update_locals(locals())
            if callback.on_step() is False:
                return RolloutReturn(
                    num_collected_steps * env.num_envs,
                    num_collected_episodes,
                    continue_training=False,
                )

            self._update_info_buffer(infos, dones)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

            self._last_episode_starts = dones
            lstm_states = new_lstm_states
            self._last_lstm_states = lstm_states

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)
            self._on_step()

            for done in dones:
                if done:
                    num_collected_episodes += 1
                    self._episode_num += 1
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self.dump_logs()

        callback.on_rollout_end()
        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def learn(
        self: SelfRecurrentSACD,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "RecurrentSACD",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfRecurrentSACD:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["actor", "critic", "critic_target", "_last_lstm_states"]

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            state_dicts.append("ent_coef_optimizer")
            saved_pytorch_variables = ["log_ent_coef"]
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables
