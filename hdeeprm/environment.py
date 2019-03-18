"""
The environment is the representation of the agent's observable context.
"""

from collections import OrderedDict
from functools import partial
import logging
import gym
import gym.spaces
import numpy as np

class Environment(gym.Env):
    """Environment for workload management in HDeepRM.

It is composed of an action space and an observation space. For every decision step, the agent
selects an action, which is applied to the environment. This involves mapping pending jobs to
available cores. Changes in environment's state are manifested as observations. For each action
taken, the environment provides a reward as feedback to the agent based on its objective. The
environment implementation is compliant with OpenAI gym format.

Any observation is formed by the following data fields:
  | - Fraction of available memory in each node
  | - Fraction of available memory bandwidth in each processor
  | - Fraction of current GFLOPs and Watts with respect to the maximum values for each core
  | - Fraction left for completing the served job by the core
  | - Fraction of requested resources with respect to the maximum values of requested
    time/cores/mem/mem_bw for pending jobs; five percentiles are shown (min, Q1, med, Q3, max) such
    that the agent can devise a job distribution
  | - Variability ratio of job queue size with respect to last observation

The action space is constituted by 37 possible actions, including a void action:
  +---------------+-----------------------------------------------+
  | Job selection | Core selection                                |
  +===============+=======+=======+=======+=======+=======+=======+
  |               | RANDM | HICOM | HICOR | HIMEM | HIMBW | LPOWR |
  +---------------+-------+-------+-------+-------+-------+-------+
  | RANDM         | 0     | 1     | 2     | 3     | 4     | 5     |
  +---------------+-------+-------+-------+-------+-------+-------+
  | FIARR         | 6     | 7     | 8     | 9     | 10    | 11    |
  +---------------+-------+-------+-------+-------+-------+-------+
  | SHORT         | 12    | 13    | 14    | 15    | 16    | 17    |
  +---------------+-------+-------+-------+-------+-------+-------+
  | SMALL         | 18    | 19    | 20    | 21    | 22    | 23    |
  +---------------+-------+-------+-------+-------+-------+-------+
  | LRMEM         | 24    | 25    | 26    | 27    | 28    | 29    |
  +---------------+-------+-------+-------+-------+-------+-------+
  | LRMBW         | 30    | 31    | 32    | 33    | 34    | 35    |
  +---------------+-------+-------+-------+-------+-------+-------+
  | Void action   | 36    |       |       |       |       |       |
  +---------------+-------+-------+-------+-------+-------+-------+

Job selection policies:
  | - RANDM (`random`): random job in the job queue.
  | - FIARR (`first`): oldest job in the job queue.
  | - SHORT (`shortest`): job with the least requested running time.
  | - SMALL (`smallest`): job with the least requested cores.
  | - LRMEM (`low_mem`): job with the least requested memory capacity.
  | - LRMBW (`low_mem_bw`): job with the least requested memory bandwidth.

Core selection policies:
  | - RANDM (`random`): random core in the core pool.
  | - HICOM (`high_gflops`): core with the highest peak compute capability.
  | - HICOR (`high_cores`): core in the processor with the most amount of available cores.
  | - HIMEM (`high_mem`): core in the node with the most amount of current memory capacity.
  | - HIMBW (`high_mem_bw`): core in the processor with the most amount of current memory bandwidth.
  | - LPOWR (`low_power`): core with the lowest power consumption.

Possible objectives for the agent:
  | - Average job slowdown: on average, how much of the service time is due to stalling of jobs in
    the job queue.
  | - Average job completion time: on average, how much service time for jobs in the platform.
  | - Utilization: number of active cores over the simulation time.
  | - Makespan: time span from the arrival of the absolute first job until the completion of the
    absolute last job.
  | - Energy consumption: total amount of energy consumed during the simulation.
  | - Energy Delay Product (EDP): product of the energy consumption by the makespan.

Attributes:
    workload_manager (:class:`~hdeeprm.entrypoints.HDeepRMWorkloadManager.HDeepRMWorkloadManager`):
        Reference to HDeepRM workload manager required to schedule the jobs on the decision step.
    action_space (gym.spaces.Discrete):
        The action space described above. See `Spaces <https://gym.openai.com/docs/#spaces>`_.
    action_keys (list):
        List of sorting key pairs indexed by action IDs. Keys are applied to the job scheduler and
        the resource manager selections.
    observation_space (gym.spaces.Box):
        The observation space described above. See `Spaces <https://gym.openai.com/docs/#spaces>`_.
    reward (function):
        Mapped to a reward function depending on the agent's objective.
    queue_sensitivity (float):
        Sensitivity of the observation to variations in job queue size. If sensitivity is high,
        larger variations will be noticed, however smaller ones will not have significant impact.
        If sensitivity is low, smaller variations will be noticed and large ones will be clipped,
        thus impactless.
    last_job_queue_length (int):
        Last value of the job queue length. Used for calculating the variation ratio.
    """

    def __init__(self, workload_manager, env_options: dict) -> None:
        self.workload_manager = workload_manager
        # Selection policies
        job_selections = OrderedDict({
            'random': None,
            'first': lambda job: job.submit_time,
            'shortest': lambda job: job.req_time,
            'smallest': lambda job: job.requested_resources,
            'low_mem': lambda job: job.mem,
            'low_mem_bw': lambda job: job.mem_bw
        })
        core_selections = OrderedDict({
            'random': None,
            'high_gflops': lambda core: - core.processor['gflops_per_core'],
            'high_cores': lambda core: - len([c for c in core.processor['local_cores']\
                                              if c.state['served_job']]),
            'high_mem': lambda core: - core.processor['node']['current_mem'],
            'high_mem_bw': lambda core: - core.processor['current_mem_bw'],
            'low_power': lambda core: core.processor['power_per_core']
        })
        # Parse the actions selected by the user
        self.actions = []
        if 'actions' in env_options:
            for sel in env_options['actions']['selection']:
                for job_sel, core_sels in sel.items():
                    for core_sel in core_sels:
                        self.actions.append(
                            (job_selections[job_sel], core_selections[core_sel])
                        )
            self.with_void = env_options['actions']['void']
        else:
            for job_sel in job_selections.values():
                for core_sel in core_selections.values():
                    self.actions.append((job_sel, core_sel))
            self.with_void = True
        # Add the void action if specified
        if self.with_void:
            nb_actions = len(self.actions) + 1
        else:
            nb_actions = len(self.actions)
        self.action_space = gym.spaces.Discrete(nb_actions)

        # Observation types
        observation_types = {
            'normal': {
                'size': self.workload_manager.resource_manager.platform['total_nodes'] +\
                        self.workload_manager.resource_manager.platform['total_processors'] +\
                        self.workload_manager.resource_manager.platform['total_cores'] * 3 + 21,
                'observation': partial(self._base_observation, otype='normal')
            },
            'small': {
                'size': self.workload_manager.resource_manager.platform['total_nodes'] +\
                        self.workload_manager.resource_manager.platform['total_processors'] + 21,
                'observation': partial(self._base_observation, otype='small')
            },
            'minimal': {
                'size': 21,
                'observation': partial(self._base_observation, otype='minimal')
            }
        }
        # Set the observation type
        if 'observation' in env_options:
            observation_type = env_options['observation']
        else:
            observation_type = 'normal'
        self.observation = observation_types[observation_type]['observation']
        self.observation_space = gym.spaces.Box(
            low=np.zeros(observation_types[observation_type]['size'], dtype=np.float32),
            high=np.ones(observation_types[observation_type]['size'], dtype=np.float32),
            dtype=np.float32
        )
        # Sets the correspondent reward function based on the objective
        objective_to_reward = {
            'avg_job_slowdown': self.avg_job_slowdown_reward,
            'avg_job_completion': self.avg_job_completion_reward,
            'avg_utilization': self.avg_utilization_reward,
            'makespan': self.makespan_reward,
            'energy_consumption': self.energy_consumption_reward,
            'edp': self.edp_reward
        }
        self.reward = objective_to_reward[env_options['objective']]
        self.queue_sensitivity = env_options['queue_sensitivity']
        self.last_job_queue_length = 0

    @property
    def action_size(self) -> int:
        """Action space size.

Utilized for output layer sizing in agent's inner models.

Returns:
    The size of the action space.
        """

        return self.action_space.n

    @property
    def observation_size(self) -> int:
        """Observation space size.

Utilized for input layer sizing in agent's inner models.

Returns:
    The size of the observation space.
        """

        return self.observation_space.shape[0]

    def _base_observation(self, otype: str) -> np.ndarray:
        """Constructs and provides the agent with an observation.

The observation composition is explained in detail in :class:`~hdeeprm.environment.Environment`.

Args:
    otype (str):
        Type of the observation.
Returns:
    The observation as a NumPy array.
        """

        def to_range(variation_ratio: float) -> float:
            """Converts variation ratio to range [0, 1].

Args:
    variation_ratio (float):
        Variation ratio of the job queue size with respect to last observation.

Returns:
    The variation ratio in range [0, 1].
            """

            variation_ratio = (variation_ratio + self.queue_sensitivity)\
                              / (2 * self.queue_sensitivity)
            # Clip it
            return max(0.0, min(1.0, variation_ratio))

        observation = []
        if otype != 'minimal':
            for cluster in self.workload_manager.resource_manager.platform['clusters']:
                for node in cluster['local_nodes']:
                    # Node: current fraction of memory available
                    observation.append(node['current_mem'] / node['max_mem'])
                    for processor in node['local_processors']:
                        # Processor: current fraction of memory BW available
                        # Memory bandwidth is not bounded, since several jobs might be
                        # overutilizing it, clip it to a minimum
                        observation.append(
                            max(0.0, processor['current_mem_bw']) / processor['max_mem_bw']
                        )
                        if otype != 'small':
                            for core in processor['local_cores']:
                                if not core.state['served_job']:
                                    # If the core is not active, remaining percentage is 0.0
                                    remaining_per = 0.0
                                else:
                                    # Percentage remaining of the current served job
                                    core.update_completion(self.workload_manager.bs.time())
                                    remaining_per = core.get_remaining_per()
                                observation.extend(
                                    [core.state['current_gflops'] / processor['gflops_per_core'],
                                    core.state['current_power'] / processor['power_per_core'],
                                    remaining_per]
                                )
        req_time = np.array(
            [job.req_time for job in self.workload_manager.job_scheduler.pending_jobs])
        req_core = np.array(
            [job.requested_resources for job in self.workload_manager.job_scheduler.pending_jobs])
        req_mem = np.array(
            [job.mem for job in self.workload_manager.job_scheduler.pending_jobs])
        req_mem_bw = np.array(
            [job.mem_bw for job in self.workload_manager.job_scheduler.pending_jobs])
        # Calculate percentiles for each requested resource
        # Each percentile is tranformed into a fraction relative to the maximum value
        job_limits = self.workload_manager.resource_manager.platform['job_limits']
        reqes = (req_time, req_core, req_mem, req_mem_bw)
        reqns = ('time', 'core', 'mem', 'mem_bw')
        maxes = (job_limits['max_time'], job_limits['max_core'],
                 job_limits['max_mem'], job_limits['max_mem_bw'])
        for reqe, reqn, maxe in zip(reqes, reqns, maxes):
            pmin = np.min(reqe) / maxe
            p25 = np.percentile(reqe, 25) / maxe
            pmed = np.median(reqe) / maxe
            p75 = np.percentile(reqe, 75) / maxe
            pmax = np.max(reqe) / maxe
            logging.info('Relative percentiles for %s: %s %s %s %s %s',
                         reqn, pmin, p25, pmed, p75, pmax)
            observation.extend([pmin, p25, pmed, p75, pmax])
        # Fraction of queue variation since last observation
        if not self.last_job_queue_length and self.workload_manager.job_scheduler.nb_pending_jobs:
            # Past queue was empty and current queue has jobs
            variation_ratio = 1.0
        else:
            # 0.0 when current queue is 1 x queue_sensitivity less than past queue
            # 1.0 when current queue is 1 x queue_sensitivity more than past queue
            # 0.5 indicates no variability
            variation = self.workload_manager.job_scheduler.nb_pending_jobs\
                        - self.last_job_queue_length
            variation_ratio = to_range(
                variation / min(self.workload_manager.job_scheduler.nb_pending_jobs,
                                self.last_job_queue_length)
            )
        observation.append(variation_ratio)
        self.last_job_queue_length = self.workload_manager.job_scheduler.nb_pending_jobs
        return np.array(observation, dtype=np.float32)

    def step(self, action: int) -> None:
        """Step representing the environment alteration.

Jobs are mapped into available resources and further communicated to Batsim. If ``void`` action
is selected, no scheduling occurs.

Args:
    action (int):
        Action ID to be applied.
        """

        assert self.action_space.contains(action), f'{action} ({type(action)}) invalid'
        # Check for void action, if so do not do anything
        if self.with_void and action == self.action_size - 1:
            logging.warning('Void action selected!')
        else:
            # Set actions as given by the key
            keys = self.actions[action]
            self.workload_manager.job_scheduler.sorting_key = keys[0]
            self.workload_manager.resource_manager.sorting_key = keys[1]
            # Schedule jobs
            self.workload_manager.schedule_jobs()

    def avg_job_slowdown_reward(self) -> float:
        """Reward when the objetive is to minimize average job slowdown.

It is the negative inverse summation of requested times. If the agent is prioritizing short jobs,
slowdowns will also go down, because the working set of jobs will do too.

Returns:
    Negative inverse summation of requested times of all jobs active in the system.
        """

        pending_jobs_req_time = [
            job.req_time for job in self.workload_manager.job_scheduler.pending_jobs
        ]
        active_jobs = set(core.state['served_job'] for core\
                          in self.workload_manager.resource_manager.core_pool\
                          if core.state['served_job'])
        active_jobs_req_time = [job.req_time for job in active_jobs]
        all_jobs_req_time = np.array([*pending_jobs_req_time, *active_jobs_req_time])
        if all_jobs_req_time.size == 0:
            return 0.0
        return np.sum(- 1.0 / all_jobs_req_time)

    def avg_job_completion_reward(self) -> float:
        """Reward when the objective is to minimize average job completion time.

It is the negative the number of unfinished jobs in the system. As more jobs are completed, the
reward will be higher.

Returns:
    Negative number of unfinished jobs in the system.
        """

        return - (self.workload_manager.job_scheduler.nb_pending_jobs\
                  + self.workload_manager.job_scheduler.nb_active_jobs)

    def avg_utilization_reward(self) -> float:
        """Reward when the objective is to maximize average utilization.

Average utilization is the average number of active resources during the simulation. Reward is then
the number of active resources.

Returns:
    Number of active resources.
        """

        return len(
            [core for core in self.workload_manager.resource_manager.core_pool if core.served_job]
        )

    def makespan_reward(self) -> float:
        """Reward when the objective is to minimize makespan.

Makespan is the total time from the arrival of the first job to the completion of the last one.
Reward is the total number of current GFLOPs in the data centre. Higher throughputs will lead to
lower makespans.

Returns:
    Current total GFLOPs provided by the data centre service.
        """

        return sum([core.state['current_gflops'] for core\
                    in self.workload_manager.resource_manager.core_pool])

    def energy_consumption_reward(self) -> float:
        """Reward when the objective is to minimize total energy consumption.

It is negative the of current power usage in the data centre. Keeping the power low will decrease
total energy consumed.

Returns:
    Negative the power usage in the data centre service.
        """

        return - sum([core.state['current_power'] for core\
                      in self.workload_manager.resource_manager.core_pool])

    def edp_reward(self) -> float:
        """Reward when the objective is to minimize the Energy-Delay Product (EDP).

TODO
        """

    def render(self, mode='human'):
        """Not used."""

    def reset(self):
        """Not used."""
