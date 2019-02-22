"""
Agents learn when interacting with the Workload Management
environment and take care of selecting the mapping policies.
This module defines all the agents available in HDeepRM.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Agent(nn.Module):
    """
    Base Agent.
    Superclass for all agents.
    """

    def __init__(self):
        super(Agent, self).__init__()
        # Episode scoped reward history
        self.rewards = []

    def decide(self, observation):
        """
        Chooses an action to take over the environment.
        """

        raise NotImplementedError

class ClassicAgent(Agent):
    """
    Classic Agent.
    Chooses actions based on a pair of fixed policies.
    """

    policy_pair_to_action = {
        'random-random': 0,
        'random-high_flops': 1,
        'random-high_mem': 2,
        'random-high_mem_bw': 3,
        'random-low_energy': 4,
        'first-random': 5,
        'first-high_flops': 6,
        'first-high_mem': 7,
        'first-high_mem_bw': 8,
        'first-low_energy': 9,
        'shortest-random': 10,
        'shortest-high_flops': 11,
        'shortest-high_mem': 12,
        'shortest-high_mem_bw': 13,
        'shortest-low_energy': 14,
        'low_mem-random': 15,
        'low_mem-high_flops': 16,
        'low_mem-high_mem': 17,
        'low_mem-high_mem_bw': 18,
        'low_mem-low_energy': 19,
        'low_mem_bw-random': 20,
        'low_mem_bw-high_flops': 21,
        'low_mem_bw-high_mem': 22,
        'low_mem_bw-high_mem_bw': 23,
        'low_mem_bw-low_energy': 24
    }

    def __init__(self, policy_pair):
        super(ClassicAgent, self).__init__()
        self.action = ClassicAgent.policy_pair_to_action[policy_pair]

    def decide(self, observation):
        """
        Returns the fixed action associated to its policy pair.
        """

        return self.action

class LearningAgent(Agent):
    """
    Agent which learns based on its actions.
    Superclass for policy and value based learners.
    """

    def __init__(self, gamma):
        super(LearningAgent, self).__init__()
        # Discount factor for rewards
        self.gamma = gamma

    def process(self, observation):
        """
        Process the given observation. This produces
        an output based on the learning objective.
        """

        raise NotImplementedError

    def loss(self):
        """
        The loss measures how good the agent has decided
        based on its rewards and decision outputs.
        """

        raise NotImplementedError

    def transform_rewards(self):
        """
        Discounting and scaling of rewards are applied for adjusting
        the relevance of recency in obtaining them.
        """

        # Discounting
        treward = 0
        trewards = []
        for reward in self.rewards[::- 1]:
            treward = reward + self.gamma * treward
            trewards.insert(0, treward)
        # Scaling
        trewards = torch.Tensor(trewards)
        if len(trewards) > 1:
            trewards = (trewards - trewards.mean()) / (trewards.std() + np.finfo(np.float32).eps.item())
        return trewards

class PolicyLearningAgent(LearningAgent):
    """
    Policy Learning Agent.
    It learns a policy for mapping jobs into resources.
    """

    def __init__(self, gamma):
        super(PolicyLearningAgent, self).__init__(gamma)
        # Episode scoped log probs
        self.log_probs = []

    def forward_policy(self, observation):
        """
        Constitutes the forward pass of the neural network
        for learning the Policy.
        It takes an observation and produces a set of action probabilities.
        """

        raise NotImplementedError

    def get_action(self, probabilities):
        """
        Gets an action based on the passed probabilities.
        """

        logging.info('Probs %s', probabilities)
        dist = torch.distributions.Categorical(probabilities)
        action = dist.sample()
        # Save the log prob
        self.save_log_prob(dist.log_prob(action))
        return action.item()

    def policy_loss(self, rews_or_advs):
        """
        Obtains the loss for the agent's behaviour during the episode.
        This is used to update the agent preferences.
        It is calculated based on log loss, and it is negative due to
        optimizers using gradient descent. For higher rewards, loss is lower.
        """

        policy_loss = []
        for log_prob, rew_or_adv in zip(self.log_probs, rews_or_advs):
            policy_loss.append(- log_prob * rew_or_adv)
        return policy_loss

    def save_log_prob(self, log_prob):
        """
        Saves a log prob to the episode history.
        """

        self.log_probs.append(log_prob)

class ValueLearningAgent(LearningAgent):
    """
    Value Learning Agent.
    It learns an estimation of the expected long-term value
    of selecting certain jobs and resources for a given
    environment state.
    """

    def __init__(self, gamma):
        super(ValueLearningAgent, self).__init__(gamma)
        # Episode scoped values
        self.values = []

    def forward_value(self, observation):
        """
        Constitutes the forward pass of the neural network
        for learning the long-term expected value estimation.
        """

        raise NotImplementedError

    def value_loss(self, rews):
        """
        Value loss is calculated based on Huber loss.
        It compares how good of an approximation the Critic has provided
        as expected future reward in comparison with the actual reward.
        """

        value_loss = []
        for value, rew in zip(self.values, rews):
            value_loss.append(F.smooth_l1_loss(value, torch.Tensor([rew])))
        return value_loss

    def save_value(self, value):
        """
        Saves a value to the episode history.
        """

        self.values.append(value)

class ReinforceAgent(PolicyLearningAgent):
    """
    REINFORCE Agent.
    It learns a Policy based on a Deep Neural Network inner model.
    """

    def __init__(self, gamma, hidden, action_size, observation_size):
        super(ReinforceAgent, self).__init__(gamma)
        # Input linear layer (takes observation as input)
        self.input = nn.Linear(observation_size, hidden)
        # First hidden layer (512x512)
        self.hidden_0 = nn.Linear(hidden, hidden)
        # Second hidden layer (512x512)
        self.hidden_1 = nn.Linear(hidden, hidden)
        # Third hidden layer (512x512)
        self.hidden_2 = nn.Linear(hidden, hidden)
        # Output linear layer (outputs action logits)
        self.output = nn.Linear(hidden, action_size)

    def decide(self, observation):
        # Process the observation
        probs = self.process(observation)
        # Get the action from the probability distribution
        return self.get_action(probs)

    def process(self, observation):
        observation = torch.from_numpy(observation).float().unsqueeze(0)
        return self.forward_policy(observation)

    def loss(self):
        rews = self.transform_rewards()
        policy_loss = self.policy_loss(rews)
        return torch.cat(policy_loss).sum()

    def forward_policy(self, observation):
        out_0 = F.leaky_relu(self.input(observation))
        out_1 = F.leaky_relu(self.hidden_0(out_0))
        out_2 = F.leaky_relu(self.hidden_1(out_1))
        out_3 = F.leaky_relu(self.hidden_2(out_2))
        return F.softmax(self.output(out_3), dim=1)

class ActorCriticAgent(PolicyLearningAgent, ValueLearningAgent):
    """
    Actor-Critic Agent.
    Learns both a Policy and a value estimation for each
    state-action pair.
    """

    def __init__(self, gamma, hidden, action_size, observation_size):
        super(ActorCriticAgent, self).__init__(gamma)
        # Common input linear layer (takes observation as input)
        self.input = nn.Linear(observation_size, hidden)
        ## Actor
        self.actor_hidden_0 = nn.Linear(hidden, hidden)
        self.actor_hidden_1 = nn.Linear(hidden, hidden)
        self.actor_hidden_2 = nn.Linear(hidden, hidden)
        # Output layer (outputs action logits)
        self.actor_output = nn.Linear(hidden, action_size)
        ## Critic
        self.critic_hidden_0 = nn.Linear(hidden, hidden)
        self.critic_hidden_1 = nn.Linear(hidden, hidden)
        self.critic_hidden_2 = nn.Linear(hidden, hidden)
        # Output layer (outputs value estimate of state-action)
        self.critic_output = nn.Linear(hidden, 1)

    def decide(self, observation):
        # Process the observation
        probs, value = self.process(observation)
        # Save the value estimation
        self.save_value(value)
        # Get the action from the probability distribution
        return self.get_action(probs)

    def process(self, observation):
        observation = torch.from_numpy(observation).float().unsqueeze(0)
        probs = self.forward_policy(observation)
        value = self.forward_value(observation)
        return probs, value

    def loss(self):
        rews = self.transform_rewards()
        advs = self.calculate_advantages(rews)
        policy_loss = self.policy_loss(advs)
        value_loss = self.value_loss(rews)
        return torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()

    def forward_policy(self, observation):
        out_0 = F.leaky_relu(self.input(observation))
        out_1 = F.leaky_relu(self.actor_hidden_0(out_0))
        out_2 = F.leaky_relu(self.actor_hidden_1(out_1))
        out_3 = F.leaky_relu(self.actor_hidden_2(out_2))
        return F.softmax(self.actor_output(out_3), dim=1)

    def forward_value(self, observation):
        out_0 = F.leaky_relu(self.input(observation))
        out_1 = F.leaky_relu(self.critic_hidden_0(out_0))
        out_2 = F.leaky_relu(self.critic_hidden_1(out_1))
        out_3 = F.leaky_relu(self.critic_hidden_2(out_2))
        return self.critic_output(out_3)

    def calculate_advantages(self, trewards):
        """
        Obtains advantages, which measure the improvement of
        taking an action with respect to the average value obtained
        at that state.
        """

        advantages = []
        for value, treward in zip(self.values, trewards):
            advantages.append(treward - value.item())
        return advantages
