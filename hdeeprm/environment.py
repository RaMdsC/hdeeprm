"""
The Environment is the representation of the Agent's observable context.
"""

import logging
import gym
import gym.spaces
import numpy as np
from hdeeprm.entrypoints.HDeepRMWorkloadManager import HDeepRMWorkloadManager

class HDeepRMEnv(gym.Env):
    """Environment for Workload Management in HDeepRM.

It is composed of an Action space and an Observation space. For every decision step, the Agent
selects an Action, which is applied to the Environment. This involves mapping pending Jobs to
available Cores. Changes in Environment's state are manifested as Observations. For each Action
taken, the Environment provides a Reward as feedback to the Agent based on its objective. The
Environment implementation is compliant with OpenAI gym format.

Any Observation is formed by the following data fields:
  | Fraction of available memory in each Node
  | Fraction of available memory bandwidth in each Processor
  | Fraction of current FLOPs and Watts with respect to the maximum values for each Core
  | Fraction left for completing the served Job by the Core
  | Fraction of requested resources with respect to the maximum values of requested
    time/cores/mem/mem_bw for pending Jobs; five percentiles are shown (min, Q1, med, Q3, max) such
    that the network can devise a Job distribution
  | Variability ratio of Job Queue size with respect to last observation

The Action space is constituted by 26 possible actions, including a void action:
  +-------------------+------------------------------------------------------------------------+
  | Job selection     | Core selection                                                         |
  +===================+========+=================+=============+================+==============+
  |                   | Random | Highest compute | Highest mem | Highest mem BW | Lowest power |
  +-------------------+--------+-----------------+-------------+----------------+--------------+
  | Random            | 0      | 1               | 2           | 3              | 4            |
  +-------------------+--------+-----------------+-------------+----------------+--------------+
  | First arrived     | 5      | 6               | 7           | 8              | 9            |
  +-------------------+--------+-----------------+-------------+----------------+--------------+
  | Shortest          | 10     | 11              | 12          | 13             | 14           |
  +-------------------+--------+-----------------+-------------+----------------+--------------+
  | Lowest req mem    | 15     | 16              | 17          | 18             | 19           |
  +-------------------+--------+-----------------+-------------+----------------+--------------+
  | Lowest req mem BW | 20     | 21              | 22          | 23             | 24           |
  +-------------------+--------+-----------------+-------------+----------------+--------------+
  |                   |        |                 |             | Void action    | 25           |
  +-------------------+--------+-----------------+-------------+----------------+--------------+

Possible objectives for the Agent and thus rewards:
  | Average Job Slowdown: on average, how much of the service time is due to stalling of Jobs in
    the Job Queue.
  | Average Job Completion Time: on average, how much service time for Jobs in the Platform.
  | Utilization: number of active Cores over the simulation time.
  | Makespan: time span from the arrival of the absolute first Job until the completion of the
    absolute last Job.
  | Energy consumption: total amount of energy consumed during the simulation.

Attributes:
    workload_manager (:class:`~hdeeprm.entrypoints.HDeepRMWorkloadManager`):
        Reference to HDeepRM Workload Manager required to schedule the Jobs on the decision step.
    action_space (gym.spaces.Discrete):
        The Action space described above. See `Spaces <https://gym.openai.com/docs/#spaces>`_.
    action_keys (list):
        List of sorting key pairs indexed by action IDs. Keys are applied to the Job Scheduler and
          the Resource Manager selections.
    observation_space (gym.spaces.Box):
        The Observation space described above. See `Spaces <https://gym.openai.com/docs/#spaces>`_.
    observation (:class:`~numpy.ndarray`):
        Current Observation for the Agent, representative of the Environment state.
    get_reward (function):
        Mapped to a reward function depending on the Agent's objective.
    queue_sensitivity (float):
        Sensitivity of the Agent to variations in Job Queue size. If sensitivity is high, larger
          variations will be noticed, however smaller ones will not have significant impact. If
            sensitivity is low, smaller variations will be noticed and large ones will be clipped,
              thus impactless.
    last_job_queue_length (int):
        Last value of the Job Queue length. Used for calculating the variability ratio.
    """

    def __init__(self, workload_manager: HDeepRMWorkloadManager, objective: str,
                 queue_sensitivity: float) -> None:
        # Reference to the Workload Manager required for the environment progress
        self.workload_manager = workload_manager

        # Action space
        # Implemented by varying the sorting key of data structures
        self.action_space = gym.spaces.Discrete(26)
        self.action_keys = [
            (None, None),
            (None, lambda res: - res.processor['flops_per_core']),
            (None, lambda res: - res.processor['node']['current_mem']),
            (None, lambda res: - res.processor['current_mem_bw']),
            (None, lambda res: res.processor['power_per_core']),
            (lambda job: job.submit_time, None),
            (lambda job: job.submit_time, lambda res: - res.processor['flops_per_core']),
            (lambda job: job.submit_time, lambda res: - res.processor['node']['current_mem']),
            (lambda job: job.submit_time, lambda res: - res.processor['current_mem_bw']),
            (lambda job: job.submit_time, lambda res: res.processor['power_per_core']),
            (lambda job: job.req_time, None),
            (lambda job: job.req_time, lambda res: - res.processor['flops_per_core']),
            (lambda job: job.req_time, lambda res: - res.processor['node']['current_mem']),
            (lambda job: job.req_time, lambda res: - res.processor['current_mem_bw']),
            (lambda job: job.req_time, lambda res: res.processor['power_per_core']),
            (lambda job: job.mem, None),
            (lambda job: job.mem, lambda res: - res.processor['flops_per_core']),
            (lambda job: job.mem, lambda res: - res.processor['node']['current_mem']),
            (lambda job: job.mem, lambda res: - res.processor['current_mem_bw']),
            (lambda job: job.mem, lambda res: res.processor['power_per_core']),
            (lambda job: job.mem_bw, None),
            (lambda job: job.mem_bw, lambda res: - res.processor['flops_per_core']),
            (lambda job: job.mem_bw, lambda res: - res.processor['node']['current_mem']),
            (lambda job: job.mem_bw, lambda res: - res.processor['current_mem_bw']),
            (lambda job: job.mem_bw, lambda res: res.processor['power_per_core'])
        ]
        self.action_size = self.action_space.n

        # Observation space
        observation_size = self.workload_manager.resource_manager.platform['total_nodes'] +\
                           self.workload_manager.resource_manager.platform['total_processors'] +\
                           self.workload_manager.resource_manager.platform['total_cores'] * 3 + 21
        self.observation_space = gym.spaces.Box(low=np.zeros(observation_size, dtype=np.float32),
                                                high=np.ones(observation_size, dtype=np.float32),
                                                dtype=np.float32)
        self.observation = None

        # Sets the correspondent reward function based on the objective
        objective_to_reward = {
            'avg_job_slowdown': self.avg_job_slowdown_reward,
            'avg_job_completion': self.avg_job_completion_reward,
            'utilization': self.utilization_reward,
            'makespan': self.makespan_reward,
            'energy_consumption': self.energy_consumption_reward
        }
        self.get_reward = objective_to_reward[objective]
        
        # Queue sensitivity.
        self.queue_sensitivity = queue_sensitivity

        # Last value of the job queue length for calculating increments
        self.last_job_queue_length = 0

    def new_observation(self):
        """
        Sets the new observation of the environment.
        """

        def to_range(variation_ratio):
            """
            Converts variation ratio to range [0, 1]
            """

            variation_ratio = (variation_ratio + self.queue_sensitivity) / (2 * self.queue_sensitivity)
            # Clip it
            return max(0.0, min(1.0, variation_ratio))

        observation = []
        for cluster in self.resource_manager.platform.local_clusters:
            for node in cluster.local_nodes:
                # Node current percentage of mem available
                observation.append(node.current_mem / node.max_mem)
                for processor in node.local_processors:
                    # Processor current percentage of mem_bw
                    # Memory bandwidth is not bounded, since several jobs might be
                    # overutilizing it, clip it to a minimum
                    observation.append(max(0.0, processor.current_mem_bw) / processor.max_mem_bw)
                    for core in processor.local_cores:
                        if not core.served_job:
                            # If the core is not active, remaining
                            # percentage is 0.0
                            remaining_per = 0.0
                        else:
                            # Percentage remaining of the current served job
                            core.update_completion(self.workload_manager.bs.time())
                            remaining_per = core.get_remaining_per()
                        observation.extend([core.current_flops / processor.flops_per_core,
                                            core.current_power / processor.power_per_core,
                                            remaining_per])
        req_time = np.array([job.req_time for job in self.job_scheduler.pending_jobs])
        req_core = np.array([job.requested_resources for job in self.job_scheduler.pending_jobs])
        req_mem = np.array([job.mem for job in self.job_scheduler.pending_jobs])
        req_mem_bw = np.array([job.mem_bw for job in self.job_scheduler.pending_jobs])
        # Calculate percentiles for each requested resource
        # Each percentile is tranformed into a % relative to the maximum value
        platform_ref = self.resource_manager.platform
        requ = (req_time, req_core, req_mem, req_mem_bw)
        requ_names = ('time', 'core', 'mem', 'mem_bw')
        maxi = (platform_ref.max_time_per_job, platform_ref.max_core_per_job,
                platform_ref.max_mem_per_job, platform_ref.max_mem_bw_per_job)
        for r, rn, m in zip(requ, requ_names, maxi):
            pmin = np.min(r) / m
            p25 = np.percentile(r, 25) / m
            pmed = np.median(r) / m
            p75 = np.percentile(r, 75) / m
            pmax = np.max(r) / m
            logging.info('Relative percentiles for %s: %s %s %s %s %s', rn, pmin, p25, pmed, p75, pmax)
            logging.debug('Values: %s', r)
            observation.extend([pmin, p25, pmed, p75, pmax])
        # Percentage of queue variability since last observation
        if not self.last_job_queue_length and self.job_scheduler.nb_pending_jobs:
            # Past queue was empty and current queue has jobs
            variation_ratio = 1.0
        else:
            # 0.0 when current queue is 1xQueue Sensitivity less than past queue
            # 1.0 when current queue is 1xQueue Sensitivity more than past queue
            # 0.5 indicates no variability
            variation = self.job_scheduler.nb_pending_jobs - self.last_job_queue_length
            variation_ratio = to_range(variation / min(self.job_scheduler.nb_pending_jobs, self.last_job_queue_length))
        observation.append(variation_ratio)
        self.last_job_queue_length = self.job_scheduler.nb_pending_jobs
        self.observation = np.array(observation, dtype=np.float32)

    def step(self, action):
        assert self.action_space.contains(action), f'{action} ({type(action)}) invalid'
        # Check for void action, if so do not do anything
        if action == 25:
            logging.warning('Void action selected!')
        else:
            # Set actions as given by the key
            keys = self.action_keys[action]
            self.job_scheduler.sorting_key = keys[0]
            self.resource_manager.sorting_key = keys[1]
            # Schedule jobs
            self.workload_manager.schedule_jobs()

    def render(self):
        pass

    def reset(self):
        pass

    def avg_job_slowdown_reward(self):
        """
        Calculates the reward when the objetive is to minimize
        average job slowdown.
        Reward is the mean of requested times for all jobs currently
        in the system. Higher means when longest jobs in the queue,
        which indicates that shorter jobs are being scheduled,
        and thus avg job slowdown is kept low.
        """

        pending_jobs_req_time = [job.req_time for job in self.job_scheduler.pending_jobs]
        active_jobs_req_time = [core.served_job.req_time for core\
                                in self.resource_manager.core_pool if core.served_job]
        all_jobs_req_time = np.array([*pending_jobs_req_time, *active_jobs_req_time])
        if not all_jobs_req_time:
            return 0.0
        return np.mean(all_jobs_req_time)

    def avg_job_completion_reward(self):
        """
        Calculates the reward when the objective is to minimize
        average job completion time.
        Reward is the number of completed jobs.
        """

        return self.job_scheduler.nb_completed_jobs

    def utilization_reward(self):
        """
        Calculates the reward when the objective is to minimize
        makespan, that is, total time to execute all the jobs.
        Reward is number of active resources.
        """

        return len([core for core in self.resource_manager.core_pool if core.served_job])

    def makespan_reward(self):
        """
        Calculates the reward when the objective is to minimize
        makespan, that is, total time to execute all the jobs.
        Reward is the total number of current flops in the data centre.
        """

        return sum([core.current_flops for core in self.resource_manager.core_pool])

    def energy_consumption_reward(self):
        """
        Calculates the reward when the objective is to minimize
        energy consumption.
        Reward is inverse the current power usage in the data centre.
        """

        return 1 / sum([core.current_power for core in self.resource_manager.core_pool])

    def provide_reward(self, agent):
        """
        Obtains the reward after each action based on the objective.
        Provides it to the agent for feedback.
        """

        # Get the reward
        reward = self.get_reward()
        logging.info('Reward %s', reward)
        # Give it to the agent
        agent.rewards.append(reward)
