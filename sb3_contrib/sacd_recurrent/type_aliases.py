from typing import NamedTuple

import torch as th
from stable_baselines3.common.type_aliases import PyTorchObs


class RecurrentReplayBufferSamples(NamedTuple):
    """A batch of fixed-length sequences sampled from replay.

    ``observations`` contains one more state than there are transitions:
    [s_0, ..., s_T], whereas actions/rewards/dones contain T entries.
    T = burn_in_length + sequence_length.
    """

    observations: PyTorchObs
    actions: th.Tensor
    rewards: th.Tensor
    dones: th.Tensor
    episode_starts: th.Tensor
    batch_size: int
    sequence_length: int
    burn_in_length: int
