"""
Defines the class for HDeepRM Workload Manager.
"""

import logging
import os
import numpy as np
import torch
from hdeeprm.agent import ClassicAgent, ReinforceAgent, ActorCriticAgent
from hdeeprm.entrypoints.BaseWorkloadManager import BaseWorkloadManager
from hdeeprm.environment import HDeepRMEnv

class HDeepRMWorkloadManager(BaseWorkloadManager):
    """
    HDeepRM Workload Manager.
    """

    def __init__(self, options):
        super().__init__(options)
        # Options
        self.options = options
        # HDeepRM environment
        self.env = HDeepRMEnv(self, options['objective'], float(options['queue_sensitivity']))
        # Reward logging
        self.rewards_log = open('rewards.log', 'a+')
        # Create the agent
        self.agent, self.optimizer = self.get_agent()
        # Decision steps
        self.step = 0
        # Flags for controlling when the JOB_SUBMITTED or JOB_COMPLETED events are
        # notified of
        self.jobs_submitted = False
        self.jobs_completed = False
        # Flag for recording when an action is taken
        self.action_taken = False

    def get_agent(self):
        """
        Generates the agent.
        It obtains information about the state of the environment via obervations.
        It also takes actions which alter the environment.
        """

        optimizer = None
        if self.options['agent'] == 'classic':
            agent = ClassicAgent(self.options['policy'])
        else:
            if self.options['agent'] == 'reinforce':
                agent = ReinforceAgent(float(self.options['gamma']), int(self.options['hidden']),
                                       self.env.action_size, self.env.observation_size)
            elif self.options['agent'] == 'actor_critic':
                agent = ActorCriticAgent(float(self.options['gamma']), int(self.options['hidden']),
                                         self.env.action_size, self.env.observation_size)
            else:
                raise ValueError('Unrecognized agent type')
            # Load previous trained model if exists
            if os.path.exists('saved_model.pt'):
                agent.load_state_dict(torch.load('saved_model.pt'))
            logging.debug('Model parameters: %s', [parameter for parameter in agent.parameters()])
            # If this is a train run, create the optimizer
            if self.options['run'] == 'train':
                optimizer = torch.optim.Adam(agent.parameters(), lr=float(self.options['lr']))
            # If not, seed torch for reproducibility
            else:
                torch.random.manual_seed(self.seed)
        return agent, optimizer

    def onSimulationEnds(self):
        # If this is a train run, get loss from agent
        if self.options['run'] == 'train':
            loss = self.agent.loss()
            logging.debug('From onSimulationEnds - Loss: %s', loss)
            # Update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # Save the updated model
            torch.save(self.agent.state_dict(), 'saved_model.pt')
        # Log scheduled metrics
        logging.info('Max scheduled jobs in one step: %s', self.scheduled_step["max"])
        logging.info('Min scheduled jobs in one step: %s', self.scheduled_step["min"])
        logging.info('Average scheduled jobs in one step: %s', self.scheduled_step["total"] / self.scheduled_step["num_steps"])
        # Save reward history
        logging.debug('List of rewards: %s', self.agent.rewards)
        self.rewards_log.write(f'{np.sum(self.agent.rewards)}\n')
        # Close the reward log
        self.rewards_log.close()

    def onJobSubmission(self, job):
        super().onJobSubmission(job)
        self.jobs_submitted = True

    def onJobCompletion(self, job):
        super().onJobCompletion(job)
        self.jobs_completed = True

    def onNoMoreEvents(self):
        # All events in the message have been processed
        # Check if the previous message was the one containing job events
        if self.bs.running_simulation:
            # If an action has been taken, 
            if self.action_taken:
                self.action_taken = False
                # Calculate the reward for the taken action
                self.env.provide_reward(self.agent)
            if (self.jobs_submitted or self.jobs_completed) and self.job_scheduler.nb_pending_jobs:
                self.jobs_submitted = self.jobs_completed = False
                # Set the new environment observation
                self.env.new_observation()
                logging.info('Step %s', self.step)
                logging.info('Observation %s', self.env.observation)
                # Determine the action given the current state
                action = self.agent.decide(self.env.observation)
                logging.info('Action %s', action)
                # Step the environment
                self.env.step(action)
                self.action_taken = True
                self.step += 1
            # Modify resource states
            self.change_resource_states()
