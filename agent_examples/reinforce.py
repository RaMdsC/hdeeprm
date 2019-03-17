"""
An example of an Agent implementing the REINFORCE algorithm in HDeepRM.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hdeeprm.agent import PolicyLearningAgent

class ReinforceAgent(PolicyLearningAgent):
    """Class for the agent implementing the REINFORCE algorithm.

The inner model is based on a deep neural network of the following structure:

  | Input layer: (observation_size x hidden)
  | Hidden layer #0: (hidden x hidden)
  | Hidden layer #1: (hidden x hidden)
  | Hidden layer #2: (hidden x hidden)
  | Output layer: (hidden x action_size)

All layers are :class:`~torch.nn.Linear`. A :meth:`~torch.nn.functional.leaky_relu` activation
function is applied to the first four layers, while a :meth:`~torch.nn.functional.softmax` is
applied to the last layer for outputting the probability distribution over actions.

Attributes:
    inner_model (dict):
        The inner model implementation of a neural network.
    """

    def __init__(self, gamma: float, hidden: int, action_size: int, observation_size: int) -> None:
        super(ReinforceAgent, self).__init__(gamma)
        self.input = nn.Linear(observation_size, hidden)
        self.hidden_0 = nn.Linear(hidden, hidden)
        self.hidden_1 = nn.Linear(hidden, hidden)
        self.hidden_2 = nn.Linear(hidden, hidden)
        self.output = nn.Linear(hidden, action_size)

    def forward_policy(self, observation: np.ndarray) -> torch.Tensor:
        int_0 = F.leaky_relu(self.input(observation))
        int_1 = F.leaky_relu(self.hidden_0(int_0))
        int_2 = F.leaky_relu(self.hidden_1(int_1))
        int_3 = F.leaky_relu(self.hidden_2(int_2))
        out = self.output(int_3)
        return F.softmax(out, dim=1)
