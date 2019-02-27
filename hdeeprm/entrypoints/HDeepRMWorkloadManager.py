"""
The HDeepRM Workload Manager is able to evaluate Deep Reinforcement Learning policies in the
framework.
"""

import logging
import importlib.util as iutil
import inspect
import os.path as path
import numpy as np
import torch
from batsim.batsim import Job
from hdeeprm.agent import Agent, ClassicAgent, LearningAgent
from hdeeprm.entrypoints.BaseWorkloadManager import BaseWorkloadManager
from hdeeprm.environment import HDeepRMEnv

class HDeepRMWorkloadManager(BaseWorkloadManager):
    """Entrypoint for Deep Reinforcement Learning experimentation.

This Workload Manager generates the HDeepRM Environment and provides a reference of it to the Agent,
who is in charge of making the decisions. It also orchestrates the simulation flow by handling
Batsim events and calling the Agent when it is necessary. It extends the
:class:`~hdeeprm.entrypoints.BaseWorkloadManager.BaseWorkloadManager` for basic event handling.

Attributes:
    env (:class:`~hdeeprm.environment.HDeepRMEnv`):
        The Workload Management Environment. The Agent observes and interacts with it.
    agent (:class:`~hdeeprm.agent.Agent`):
        The Agent in charge of making decisions by observing and altering the Environment.
    optimizer (:class:`~torch.optim.Optimizer`):
        Optimizer for updating the Agent's inner model weights at the end of the simulation.
    step (int):
        Current decision step.
    flow_flags (dict):
        Control the event flow. Fields:

          | jobs_submitted (bool) -
            Becomes ``True`` when at least one job has been submitted.
          | jobs_completed (bool) -
            Becomes ``True`` when at least one job has been completed.
          | action_taken (bool):
            Becomes ``True`` when an action has been taken by the Agent. This triggers the reward
              procedure.
    """

    def __init__(self, options: dict) -> None:
        super().__init__(options)
        self.env = HDeepRMEnv(self, options['env'])
        self.agent, self.optimizer = self.create_agent(options['agent'], options['seed'])
        self.step = 0
        self.flow_flags = {
            'jobs_submitted': False,
            'jobs_completed': False,
            'action_taken': False
        }

    def create_agent(self, agent_options: dict, seed: int) -> tuple:
        """Generates the Agent based on the agent options.

The agent class is obtained from the user provided file. It is instantiated according to its parent
class. Previously saved models might be loaded if the user indicates so in command line.

Args:
    agent_options (dict): options for the Agent creation. User provided.
    seed (int): random seed for torch library reproducibility when evaluating.

Returns:
    A tuple with the created Agent and the optimizer in case of training.
        """

        # Obtain the agent class
        agent_module_name = path.splitext(path.basename(agent_options['file']))[0]
        spec = iutil.spec_from_file_location(agent_module_name, agent_options['file'])
        agent_module = iutil.module_from_spec(spec)
        spec.loader.exec_module(agent_module)
        agent_class = [cl for na, cl in inspect.getmembers(agent_module, inspect.isclass)
                       if getattr(cl, '__module__', None) == agent_module_name
                       and issubclass(cl, Agent)][0]
        # Pass options depending on the superclass
        if issubclass(agent_class, ClassicAgent):
            agent = agent_class(agent_options['policy_pair'])
        elif issubclass(agent_class, LearningAgent):
            agent = agent_class(float(agent_options['gamma']), int(agent_options['hidden']),
                                self.env.action_size, self.env.observation_size)
        else:
            raise TypeError('Unrecognized agent superclass')
        # Load previously trained model if the user indicated as option
        if 'input_model' in agent_options:
            agent.load_state_dict(torch.load(agent_options['input_model']))
        # Create the optimizer if this is a training run
        optimizer = None
        if agent_options['run'] == 'train':
            optimizer = torch.optim.Adam(agent.parameters(), lr=float(agent_options['lr']))
        # If not, seed torch for reproducibility
        else:
            torch.random.manual_seed(seed)
        return agent, optimizer

    def onSimulationEnds(self) -> None:
        """Handler triggered when the simulation has ended.
        
Triggered when receiving a
`SIMULATION_ENDS <https://batsim.readthedocs.io/en/latest/protocol.html#simulation-ends>`_ event.
If the Agent evaluated has been in training mode, the loss is calculated to update its inner model
weights. The updated model is saved if the user has indicated so in command line. Rewards are also
logged for observing performance.
        """

        if self.options['agent']['run'] == 'train':
            loss = self.agent.loss()
            # Update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if 'output_model' in self.options['agent']:
                torch.save(self.agent.state_dict(), self.options['agent']['output_model'])
        # Log scheduled metrics
        logging.info('Max scheduled jobs in one step: %s', self.scheduled_step["max"])
        logging.info('Min scheduled jobs in one step: %s', self.scheduled_step["min"])
        logging.info('Average scheduled jobs in one step: %s', self.scheduled_step["total"]\
                                                               / self.scheduled_step["num_steps"])
        # Save reward history
        with open('rewards.log', 'a+') as out_f:
            out_f.write(f'{np.sum(self.agent.rewards)}\n')

    def onJobSubmission(self, job: Job) -> None:
        """Set the "jobs_submitted" flag to ``True``.

Further details on this handler on the base
:meth:`~hdeeprm.entrypoints.BaseWorkloadManager.BaseWorkloadManager.onJobSubmission`.
        """

        super().onJobSubmission(job)
        self.flow_flags['jobs_submitted'] = True

    def onJobCompletion(self, job: Job) -> None:
        """Set the "jobs_completed" flag to ``True``.

Further details on this handler on the base
:meth:`~hdeeprm.entrypoints.BaseWorkloadManager.BaseWorkloadManager.onJobCompletion`.
        """

        super().onJobCompletion(job)
        self.flow_flags['jobs_completed'] = True

    def onNoMoreEvents(self) -> None:
        """
When there are no more events in the current time step, the following flow occurs:

1. The Agent observes the Environment, obatining an approximation of the state.
2. The Agent processes this observation through its inner model, and decides which action to take,
3. The Agent alters the Environment based on the selected action.
4. In the next decision step, the Agent will be rewarded for its action.
        """

        if self.bs.running_simulation:
            if self.flow_flags['action_taken']:
                # The Agent is rewarded
                self.agent.rewarded(self.env)
                self.flow_flags['action_taken'] = False
            if (self.flow_flags['jobs_submitted'] or self.flow_flags['jobs_completed'])\
               and self.job_scheduler.nb_pending_jobs:
                # The Agent observes the Environment
                observation = self.agent.observe(self.env)
                logging.info('Step %s', self.step)
                logging.info('Observation %s', observation)
                # The Agent decides which action to take
                action = self.agent.decide(observation)
                logging.info('Action %s', action)
                # The Agent alters the Environment
                self.agent.alter(action, self.env)
                self.step += 1
                self.flow_flags['action_taken'] = True
                self.flow_flags['jobs_submitted'] = self.flow_flags['jobs_completed'] = False
            # Modify resource states
            self.change_resource_states()
