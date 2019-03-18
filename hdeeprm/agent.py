"""
Agents learn when interacting with the workload management environment, and take care of selecting
the scheduling policies. This module defines superclasses for all the agents developed for HDeepRM.
"""

import logging
import typing
import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F

class Agent(nn.Module):
    """Superclass for all Agents.

Any agent makes decisions, which are expressed in the :meth:`~hdeeprm.agent.Agent.decide` method. It
uses PyTorch :class:`~torch.nn.Module` as a base class for defining forward passes.

Attributes:
    rewards (list):
        List of rewards populated during the interaction with the environment. Rewards are feedback
        on agent's behaviour.
    """

    def __init__(self) -> None:
        super().__init__()
        self.rewards = []

    def observe(self, environment) -> np.ndarray:
        """Observes the current state of the environment.

Through observing, the agent gains information about the environment.

Args:
    environment (:class:`~hdeeprm.environment.Environment`):
        The environment to be observed.

Returns:
    A NumPy :class:`~numpy.ndarray` as an observation.
        """

        return environment.observation()

    def decide(self, observation: np.ndarray) -> int:
        """Chooses an action given an observation from the environment.

Chosen actions are given by the action space describe in :class:`~hdeeprm.environment.Environment`.

Args:
    observation (:class:`~numpy.ndarray`):
        The observation as a proxy of the environment's state.

Returns:
    Action ID to be taken over the environment.
        """

        raise NotImplementedError

    def alter(self, action: int, environment) -> None:
        """Alters the environment by applying the action.

An alteration consists of stepping the environment. During this step, job scheduling and resource
state changes are issued into Batsim.

Args:
    action (int):
        Action ID to be applied.
    environment (:class:`~hdeeprm.environment.Environment`):
        The environment to be altered.
        """

        environment.step(action)

    def rewarded(self, environment) -> None:
        """Provides the agent with a reward for its action.

Rewards are provided from the environment, and they are adjusted to the optimization objective.
        """

        self.rewards.append(environment.reward())

class ClassicAgent(Agent):
    """Fixed-policy traditional agent.

Represents an agent based on a traditional fixed policy. It can be used as a comparison against deep
reinforcement learning agents being developed. May implement any of the policy pairs in the action
space.
    """

    def __init__(self) -> None:
        super().__init__()

    def decide(self, observation: np.ndarray) -> int:
        """Returns the fixed action, which is the only one available.

See :meth:`hdeeprm.agent.Agent.decide`.
        """

        return 0

class LearningAgent(Agent):
    """Agent which learns based on its actions.

A generic learning agent processes observations through its inner model. At the end of the, the loss
given the decision chain is calculated, and utilized for updating the inner model parameters.

Attributes:
    gamma (float):
        Hyperparameter, user provided. Discount factor for rewards, inbetween [0, 1). When close to
        1, rewards from a more distant future will be considered for updating the model and
        viceversa.
    """

    def __init__(self, gamma: float) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, observation: np.ndarray) -> typing.Any:
        """Process the given observation by forwarding through inner model.

The inner model might be a neural network, which forwards the observation through its layers. The
result is defined by the structure of last layers in the inner model, and can be defined by the
user. Usual cases are policy learning and value learning, see
:meth:`hdeeprm.agent.PolicyLearningAgent.forward`
and :meth:`hdeeprm.agent.ValueLearningAgent.forward`.

Args:
    observation (:class:`~numpy.ndarray`):
        The observation as a proxy of the environment's state.

Returns:
    Defined by the user's inner model.
        """

        raise NotImplementedError

    def loss(self) -> float:
        """Measures how good the agent has decided based on its rewards and decision outputs.

The loss is used for updating the agent's inner model weights. Its definition depends on that
of the model.

Returns:
    The loss value for the agent.
        """

        raise NotImplementedError

    def transform_rewards(self) -> list:
        """Discount and scale the rewards.

Discounting and scaling of rewards are applied for adjusting the relevance of recency in obtaining
them. :attr:`~hdeeprm.agent.LearningAgent.gamma` is used as the discount factor.

Returns:
    A list with the transformed rewards for further processing.
        """

        # Discounting
        treward = 0
        trewards = []
        for reward in self.rewards[::- 1]:
            treward = reward + self.gamma * treward
            trewards.insert(0, treward)
        # Scaling
        trewards = torch.tensor(trewards)
        if len(trewards) > 1:
            trewards = (trewards - trewards.mean()) /\
                       (trewards.std() + np.finfo(np.float32).eps.item())
        return trewards.tolist()

class PolicyLearningAgent(LearningAgent):
    """Learns an adaptive policy as a probability distribution over actions.

In contrast with fixed-policy :class:`~hdeeprm.agent.ClassicAgent`, this agent learns to select
different actions given distinct observations from the environment. This means that job and resource
selections may be different depending on the environment state.

Attributes:
    probs (list):
        First action preferences of the simulation for evolution insights.
    log_probs (list):
        Log probabilities for loss calculation at the end of the simulation.
    """

    def __init__(self, gamma: float) -> None:
        super().__init__(gamma)
        self.probs = None
        self.log_probs = []

    def decide(self, observation: np.ndarray) -> int:
        """Process the observation and return the action ID.

Args:
    observation (:class:`~numpy.ndarray`):
        The observation as a proxy of the environment's state.

Returns:
    Action ID for altering the environment.
        """

        probs = self(observation)
        if not self.probs:
            self.probs = probs.detach().numpy().flatten().tolist()
        return self.get_action(probs)

    def forward(self, observation: np.ndarray) -> torch.Tensor:
        """Applies the forward pass of the inner model.

Outputs the probability distribution over actions.

Args:
    observation (:class:`~numpy.ndarray`):
        The observation as a proxy of the environment's state.

Returns:
    Tensor with the probability distribution over actions.
        """

        observation = torch.from_numpy(observation).float().unsqueeze(0)
        return self.forward_policy(observation)

    def loss(self) -> float:
        """Calculates the loss for the policy learning agent.

First it transforms the rewards, which result from the agent's behaviour during the simulation. It
calculates the losses for each step and then it sums them.

Returns:
    Sum of all the policy losses, which constitute the agent loss for the run.
        """

        rews = self.transform_rewards()
        policy_loss = self.policy_loss(rews)
        return torch.cat(policy_loss).sum()

    def forward_policy(self, observation: np.ndarray) -> torch.Tensor:
        """Forward pass of the inner model for learning the policy.

When the observation is processed, a probability distribution over the possible actions is given as
output. The agent then chooses an action given this distribution, which depends on the observation.

Args:
    observation (:class:`~numpy.ndarray`):
        The observation as a proxy of the environment's state.

Returns:
    Tensor with the probability distribution over the possible actions.
        """

        raise NotImplementedError

    def get_action(self, probabilities: torch.Tensor) -> int:
        """Gets an action based on the provided probabilities.

An action is sampled randomly from the probability distribution outputted from the forward pass. The
log probability of the action is stored for further calculating the loss.

Args:
    probabilities (:class:`~torch.Tensor`):
        Tensor with the probability distribution over actions.

Returns:
    Action ID to be taken over the environment.
        """

        logging.info('Probs %s', probabilities)
        dist = Categorical(probabilities)
        action = dist.sample()
        #p = probabilities.detach().numpy().flatten()
        #action = torch.tensor(np.random.choice(range(len(p)), p=p))
        with open('actions.log', 'a+') as out_f:
            out_f.write(f'{action.item()}\n')
        self.save_log_prob(dist.log_prob(action))
        return action.item()

    def policy_loss(self, rews_or_advs: list) -> list:
        """Calculates the losses based on the policy learned.

It is calculated based on log losses, and it is negative due to optimizers using gradient descent.
For higher rewards, losses are lower.

Args:
    rews_or_advs (list):
        List with transformed rewards or advantages (see
        `Actor-Critic example <TODO>`_).

Returns:
    List with the policy losses.
        """

        policy_loss = []
        with open('log_probs.log', 'a+') as out_f:
            out_f.write(f'{self.log_probs}\n')
        for log_prob, rew_or_adv in zip(self.log_probs, rews_or_advs):
            policy_loss.append(- log_prob * rew_or_adv)
        return policy_loss

    def save_log_prob(self, log_prob: torch.Tensor) -> None:
        """Saves a log prob in :attr:`~hdeeprm.agent.PolicyLearningAgent.log_probs`.

Args:
    log_prob (:class:`~torch.Tensor`):
        Log probability for the selected action to be stored.
        """

        self.log_probs.append(log_prob)

class ValueLearningAgent(LearningAgent):
    """Learns a value estimation for being at a given observation (state).

The value is an approach to estimate the agent's objective, which is the maximization of the
discounted cumulative reward. It indicates the future value of being in the current state
(observation). When the value is high, it means that the Agent will advance on its objective.

Attributes:
    values (list):
        Values for loss calculation at the end of the simulation.
    """

    def __init__(self, gamma: float) -> None:
        super().__init__(gamma)
        self.values = []

    def forward(self, observation: np.ndarray) -> torch.Tensor:
        """Applies the forward pass of the inner model.

Outputs the value estimation of being at that state.

Args:
    observation (:class:`~numpy.ndarray`):
        The observation as a proxy of the environment's state.

Returns:
    Tensor with the value estimation.
        """

        observation = torch.from_numpy(observation).float().unsqueeze(0)
        return self.forward_value(observation)

    def loss(self) -> float:
        """Calculates the loss for the value learning agent.

First it transforms the rewards, which result from the agent's behaviour during the simulation. It
calculates the losses for each step and then it sums them.

Returns:
    Sum of all the value losses, which constitute the agent loss for the run.
        """

        rews = self.transform_rewards()
        value_loss = self.value_loss(rews)
        return torch.cat(value_loss).sum()

    def forward_value(self, observation: np.ndarray) -> torch.Tensor:
        """Forward pass of the inner model for learning the value estimation.

When the observation is processed, a scalar is given as a value estimation of being at that state.

Args:
    observation (:class:`~numpy.ndarray`):
        The observation as a proxy of the environment's state.

Returns:
    Tensor with the value estimation for the given observation.
        """

        raise NotImplementedError

    def value_loss(self, rews: list) -> list:
        """Calculates the losses based on the values estimation learned.

It is calculated based on `Huber loss <https://en.wikipedia.org/wiki/Huber_loss>`_. It compares how
good of an approximation the agent has provided as expected future reward in comparison with the
actual reward.

Args:
    rews (list):
        List with transformed rewards.

Returns:
    List with the value losses.
        """

        value_loss = []
        for value, rew in zip(self.values, rews):
            value_loss.append(F.smooth_l1_loss(value, torch.tensor([rew])))
        return value_loss

    def save_value(self, value: torch.Tensor) -> None:
        """Saves a value in :attr:`~hdeeprm.agent.ValueLearningAgent.values`.

Args:
    value (:class:`~torch.Tensor`):
        Value for the current observation to be stored.
        """

        self.values.append(value)
