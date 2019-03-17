"""
An example of an Agent implementing the Advantage Actor-Critic algorithm in HDeepRM.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hdeeprm.agent import PolicyLearningAgent, ValueLearningAgent

class ActorCriticAgent(PolicyLearningAgent, ValueLearningAgent):
    """Class for the agent implementing the Adavantage Actor-Critic algorithm.

The inner model is based on two deep neural networks, an Actor for learning the policy and
a Critic for learning the value estimation. They have the following structure:

Shared:

  | Input layer: (observation_size x hidden)

Actor:

  | Hidden layer #0: (hidden x hidden)
  | Hidden layer #1: (hidden x hidden)
  | Hidden layer #2: (hidden x hidden)
  | Output layer: (hidden x action_size)

Critic:

  | Hidden layer #0: (hidden x hidden)
  | Hidden layer #1: (hidden x hidden)
  | Hidden layer #2: (hidden x hidden)
  | Output layer: (hidden x 1)

All layers are :class:`~torch.nn.Linear`. The input layer is shared between both networks, since the
input size is the same. A :meth:`~torch.nn.functional.leaky_relu` activation function is applied to
the first four layers of each network. In the Actor case, the last layer is applied a
:meth:`~torch.nn.functional.softmax` function for outputting the probability distribution over
actions. In the Critic, the last layer is not applied any activation since we are interested in the
predicted value for the last neuron.

Attributes:
    inner_model (dict):
        The inner model implementation with the Actor and Critic.
    """

    def __init__(self, gamma: float, hidden: int, action_size: int, observation_size: int) -> None:
        super(ActorCriticAgent, self).__init__(gamma)
        self.input = nn.Linear(observation_size, hidden)
        self.actor_hidden_0 = nn.Linear(hidden, hidden)
        self.actor_hidden_1 = nn.Linear(hidden, hidden)
        self.actor_hidden_2 = nn.Linear(hidden, hidden)
        self.actor_output = nn.Linear(hidden, action_size)
        self.critic_hidden_0 = nn.Linear(hidden, hidden)
        self.critic_hidden_1 = nn.Linear(hidden, hidden)
        self.critic_hidden_2 = nn.Linear(hidden, hidden)
        self.critic_output = nn.Linear(hidden, 1)

    def decide(self, observation: np.ndarray) -> int:
        probs, value = self.process(observation)
        self.save_value(value)
        return self.get_action(probs)

    def forward(self, observation: np.ndarray) -> tuple:
        observation = torch.from_numpy(observation).float().unsqueeze(0)
        probs = self.forward_policy(observation)
        value = self.forward_value(observation)
        return probs, value

    def loss(self) -> float:
        rews = self.transform_rewards()
        advs = self.calculate_advantages(rews)
        return torch.stack(self.policy_loss(advs)).sum()\
               + torch.stack(self.value_loss(rews)).sum()

    def forward_policy(self, observation: np.ndarray) -> torch.Tensor:
        out_0 = F.leaky_relu(self.input(observation))
        out_1 = F.leaky_relu(self.actor_hidden_0(out_0))
        out_2 = F.leaky_relu(self.actor_hidden_1(out_1))
        out_3 = F.leaky_relu(self.actor_hidden_2(out_2))
        return F.softmax(self.actor_output(out_3), dim=1)

    def forward_value(self, observation: np.ndarray) -> torch.Tensor:
        out_0 = F.leaky_relu(self.input(observation))
        out_1 = F.leaky_relu(self.critic_hidden_0(out_0))
        out_2 = F.leaky_relu(self.critic_hidden_1(out_1))
        out_3 = F.leaky_relu(self.critic_hidden_2(out_2))
        return self.critic_output(out_3)

    def calculate_advantages(self, trewards: list) -> list:
        """Calculates the advantages for the Agent.

Advantages measure the improvement of taking an action with respect to the average value obtained
at that state.

Args:
    trewards (list):
        List of previously transformed rewards.

Returns:
    List of advantages
        """

        advantages = []
        for value, treward in zip(self.values, trewards):
            advantages.append(treward - value.item())
        return advantages

    def forward(self, *inp):
        pass
