from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import BaseModel, BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from torch import nn
from torch.distributions import Categorical

LSTMState = tuple[th.Tensor, th.Tensor]


def _process_sequence(
    features: th.Tensor,
    lstm_states: LSTMState,
    episode_starts: th.Tensor,
    lstm: nn.LSTM,
) -> tuple[th.Tensor, LSTMState]:
    """SB3-Contrib RecurrentPPO-style sequence processing."""
    n_seq = lstm_states[0].shape[1]
    features_sequence = features.reshape((n_seq, -1, lstm.input_size)).swapaxes(0, 1)
    episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)

    if th.all(episode_starts == 0.0):
        lstm_output, lstm_states = lstm(features_sequence, lstm_states)
        lstm_output = th.flatten(lstm_output.transpose(0, 1), start_dim=0, end_dim=1)
        return lstm_output, lstm_states

    outputs: list[th.Tensor] = []
    for feature_t, episode_start_t in zip(features_sequence, episode_starts, strict=True):
        reset_mask = (1.0 - episode_start_t).view(1, n_seq, 1)
        output, lstm_states = lstm(
            feature_t.unsqueeze(0),
            (reset_mask * lstm_states[0], reset_mask * lstm_states[1]),
        )
        outputs.append(output)

    lstm_output = th.flatten(th.cat(outputs).transpose(0, 1), start_dim=0, end_dim=1)
    return lstm_output, lstm_states


class RecurrentActor(BasePolicy):
    """Categorical recurrent actor for discrete SAC."""

    action_space: spaces.Discrete

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        net_arch: list[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        lstm_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=False,
        )
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.lstm_hidden_size = lstm_hidden_size
        self.n_lstm_layers = n_lstm_layers
        self.lstm_kwargs = lstm_kwargs or {}

        self.lstm = nn.LSTM(
            features_dim,
            lstm_hidden_size,
            num_layers=n_lstm_layers,
            **self.lstm_kwargs,
        )
        self.lstm_hidden_state_shape = (n_lstm_layers, 1, lstm_hidden_size)

        num_actions = int(action_space.n)
        self.policy_net = nn.Sequential(*create_mlp(lstm_hidden_size, num_actions, net_arch, activation_fn))

    def initial_state(self, n_seq: int, device: th.device | None = None) -> LSTMState:
        device = self.device if device is None else device
        shape = (self.n_lstm_layers, n_seq, self.lstm_hidden_size)
        return th.zeros(shape, device=device), th.zeros(shape, device=device)

    def get_logits(
        self,
        obs: PyTorchObs,
        lstm_states: LSTMState,
        episode_starts: th.Tensor,
    ) -> tuple[th.Tensor, LSTMState]:
        features = self.extract_features(obs, self.features_extractor)
        latent, new_states = _process_sequence(features, lstm_states, episode_starts, self.lstm)
        return self.policy_net(latent), new_states

    def action_log_prob(
        self,
        obs: PyTorchObs,
        lstm_states: LSTMState,
        episode_starts: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, LSTMState]:
        logits, new_states = self.get_logits(obs, lstm_states, episode_starts)
        log_probs = th.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        return probs, log_probs, new_states

    def forward(
        self,
        obs: PyTorchObs,
        lstm_states: LSTMState,
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> tuple[th.Tensor, LSTMState]:
        logits, new_states = self.get_logits(obs, lstm_states, episode_starts)
        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = Categorical(logits=logits).sample()
        return actions, new_states

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        if isinstance(observation, dict):
            n_envs = next(iter(observation.values())).shape[0]
        else:
            n_envs = observation.shape[0]
        states = self.initial_state(n_envs)
        episode_starts = th.zeros(n_envs, device=self.device)
        actions, _ = self(observation, states, episode_starts, deterministic)
        return actions


class RecurrentDiscreteCritic(BaseModel):
    """Twin discrete Q-functions behind a recurrent state encoder."""

    action_space: spaces.Discrete

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        net_arch: list[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        lstm_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.n_critics = n_critics
        self.lstm_hidden_size = lstm_hidden_size
        self.n_lstm_layers = n_lstm_layers
        self.lstm_kwargs = lstm_kwargs or {}

        self.lstm = nn.LSTM(
            features_dim,
            lstm_hidden_size,
            num_layers=n_lstm_layers,
            **self.lstm_kwargs,
        )

        num_actions = int(action_space.n)
        self.q_networks = nn.ModuleList(
            [nn.Sequential(*create_mlp(lstm_hidden_size, num_actions, net_arch, activation_fn)) for _ in range(n_critics)]
        )

    def initial_state(self, n_seq: int, device: th.device | None = None) -> LSTMState:
        device = self.device if device is None else device
        shape = (self.n_lstm_layers, n_seq, self.lstm_hidden_size)
        return th.zeros(shape, device=device), th.zeros(shape, device=device)

    def forward(
        self,
        obs: PyTorchObs,
        lstm_states: LSTMState,
        episode_starts: th.Tensor,
    ) -> tuple[tuple[th.Tensor, ...], LSTMState]:
        features = self.extract_features(obs, self.features_extractor)
        latent, new_states = _process_sequence(features, lstm_states, episode_starts, self.lstm)
        return tuple(q_net(latent) for q_net in self.q_networks), new_states


class RecurrentSACDPolicy(BasePolicy):
    """SB3-style recurrent SAC-Discrete policy.

    The actor, critic and target critic each own a separate feature extractor and
    recurrent state. Keeping the recurrent encoders separate avoids optimizer
    coupling between actor and critic updates.
    """

    actor: RecurrentActor
    critic: RecurrentDiscreteCritic
    critic_target: RecurrentDiscreteCritic

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: dict[str, Any] | None = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        n_critics: int = 2,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        lstm_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=False,
            normalize_images=normalize_images,
        )
        if not isinstance(action_space, spaces.Discrete):
            raise ValueError("RecurrentSACDPolicy only supports gymnasium.spaces.Discrete")

        if net_arch is None:
            net_arch = [256, 256]
        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.n_critics = n_critics
        self.lstm_hidden_size = lstm_hidden_size
        self.n_lstm_layers = n_lstm_layers
        self.lstm_kwargs = lstm_kwargs or {}

        base_kwargs = dict(
            observation_space=observation_space,
            action_space=action_space,
            activation_fn=activation_fn,
            normalize_images=normalize_images,
            lstm_hidden_size=lstm_hidden_size,
            n_lstm_layers=n_lstm_layers,
            lstm_kwargs=self.lstm_kwargs,
        )
        self.actor_kwargs = dict(base_kwargs, net_arch=actor_arch)
        self.critic_kwargs = dict(base_kwargs, net_arch=critic_arch, n_critics=n_critics)

        self._build(lr_schedule)

    def make_actor(self) -> RecurrentActor:
        kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor=None)
        return RecurrentActor(**kwargs).to(self.device)

    def make_critic(self) -> RecurrentDiscreteCritic:
        kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor=None)
        return RecurrentDiscreteCritic(**kwargs).to(self.device)

    def _build(self, lr_schedule: Schedule) -> None:
        self.actor = self.make_actor()
        self.critic = self.make_critic()
        self.critic_target = self.make_critic()
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.set_training_mode(False)

        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )
        self.critic.optimizer = self.optimizer_class(
            self.critic.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def initial_state(self, n_envs: int) -> LSTMState:
        return self.actor.initial_state(n_envs, self.device)

    def forward(
        self,
        obs: PyTorchObs,
        lstm_states: LSTMState,
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> tuple[th.Tensor, LSTMState]:
        return self.actor(obs, lstm_states, episode_starts, deterministic)

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self.actor._predict(observation, deterministic)

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, np.ndarray] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        self.set_training_mode(False)
        obs_tensor, vectorized_env = self.obs_to_tensor(observation)
        if isinstance(obs_tensor, dict):
            n_envs = next(iter(obs_tensor.values())).shape[0]
        else:
            n_envs = obs_tensor.shape[0]

        if state is None:
            state_t = self.actor.initial_state(n_envs, self.device)
        else:
            state_t = (
                th.as_tensor(state[0], dtype=th.float32, device=self.device),
                th.as_tensor(state[1], dtype=th.float32, device=self.device),
            )
        if episode_start is None:
            episode_start = np.zeros(n_envs, dtype=bool)
        episode_starts = th.as_tensor(episode_start, dtype=th.float32, device=self.device)

        with th.no_grad():
            actions, new_state = self.actor(obs_tensor, state_t, episode_starts, deterministic)

        actions_np = actions.cpu().numpy()
        if not vectorized_env:
            actions_np = actions_np.squeeze(axis=0)
        return actions_np, (new_state[0].cpu().numpy(), new_state[1].cpu().numpy())

    def set_training_mode(self, mode: bool) -> None:
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        # Target critic remains in eval mode.
        self.critic_target.set_training_mode(False)
        self.training = mode

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                normalize_images=self.normalize_images,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                n_critics=self.n_critics,
                lstm_hidden_size=self.lstm_hidden_size,
                n_lstm_layers=self.n_lstm_layers,
                lstm_kwargs=self.lstm_kwargs,
                lr_schedule=self._dummy_schedule,
            )
        )
        return data


MlpLstmPolicy = RecurrentSACDPolicy


class CnnLstmPolicy(RecurrentSACDPolicy):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("features_extractor_class", NatureCNN)
        super().__init__(*args, **kwargs)


class MultiInputLstmPolicy(RecurrentSACDPolicy):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("features_extractor_class", CombinedExtractor)
        super().__init__(*args, **kwargs)
