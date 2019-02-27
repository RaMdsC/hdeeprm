"""
The Environment is the representation of the Agent's observable context.
"""

import logging
import gym
import gym.spaces
import numpy as np

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
    workload_manager (:class:`~hdeeprm.entrypoints.HDeepRMWorkloadManager.HDeepRMWorkloadManager`):
        Reference to HDeepRM Workload Manager required to schedule the Jobs on the decision step.
    action_space (gym.spaces.Discrete):
        The Action space described above. See `Spaces <https://gym.openai.com/docs/#spaces>`_.
    action_keys (list):
        List of sorting key pairs indexed by action IDs. Keys are applied to the Job Scheduler and
          the Resource Manager selections.
    observation_space (gym.spaces.Box):
        The Observation space described above. See `Spaces <https://gym.openai.com/docs/#spaces>`_.
    reward (function):
        Mapped to a reward function depending on the Agent's objective.
    queue_sensitivity (float):
        Sensitivity of the Observation to variations in Job Queue size. If sensitivity is high,
          larger variations will be noticed, however smaller ones will not have significant impact.
            If sensitivity is low, smaller variations will be noticed and large ones will be
              clipped, thus impactless.
    last_job_queue_length (int):
        Last value of the Job Queue length. Used for calculating the variation ratio.
    """

    def __init__(self, workload_manager, env_options: dict) -> None:
        self.workload_manager = workload_manager
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
        observation_size = self.workload_manager.resource_manager.platform['total_nodes'] +\
                           self.workload_manager.resource_manager.platform['total_processors'] +\
                           self.workload_manager.resource_manager.platform['total_cores'] * 3 + 21
        self.observation_space = gym.spaces.Box(low=np.zeros(observation_size, dtype=np.float32),
                                                high=np.ones(observation_size, dtype=np.float32),
                                                dtype=np.float32)
        # Sets the correspondent reward function based on the objective
        objective_to_reward = {
            'avg_job_slowdown': self.avg_job_slowdown_reward,
            'avg_job_completion': self.avg_job_completion_reward,
            'avg_utilization': self.avg_utilization_reward,
            'makespan': self.makespan_reward,
            'energy_consumption': self.energy_consumption_reward
        }
        self.reward = objective_to_reward[env_options['objective']]
        self.queue_sensitivity = env_options['queue_sensitivity']
        self.last_job_queue_length = 0

    @property
    def action_size(self) -> int:
        """Action space size.

Utilized for output layer sizing in Agent's inner models.

Returns:
    The size of the Action space.
        """

        return self.action_space.n

    @property
    def observation_size(self) -> int:
        """Observation space size.

Utilized for input layer sizing in Agent's inner models.

Returns:
    The size of the Observation space.
        """

        return self.observation_space.shape[0]

    def observation(self) -> np.ndarray:
        """Constructs and provides the Agent with an observation.

The observation composition is explained in detail in :class:`~hdeeprm.environment.HDeepRMEnv`.

Returns:
    The observation as a NumPy array.
        """

        def to_range(variation_ratio: float) -> float:
            """Converts variation ratio to range [0, 1].

Args:
    variation_ratio (float):
        Variation ratio of the Job Queue size with respect to last observation.

Returns:
    The variation ratio in range [0, 1].
            """

            variation_ratio = (variation_ratio + self.queue_sensitivity)\
                              / (2 * self.queue_sensitivity)
            # Clip it
            return max(0.0, min(1.0, variation_ratio))

        observation = []
        for cluster in self.workload_manager.resource_manager.platform['clusters']:
            for node in cluster['local_nodes']:
                # Node: current fraction of memory available
                observation.append(node['current_mem'] / node['max_mem'])
                for processor in node['local_processors']:
                    # Processor: current fraction of memory BW available
                    # Memory bandwidth is not bounded, since several jobs might be overutilizing it,
                    # clip it to a minimum
                    observation.append(max(0.0,
                                           processor['current_mem_bw']) / processor['max_mem_bw'])
                    for core in processor['local_cores']:
                        if not core.state['served_job']:
                            # If the core is not active, remaining percentage is 0.0
                            remaining_per = 0.0
                        else:
                            # Percentage remaining of the current served job
                            core.update_completion(self.workload_manager.bs.time())
                            remaining_per = core.get_remaining_per()
                        observation.extend(
                            [core.state['current_flops'] / processor['flops_per_core'],
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
            # 0.0 when current queue is 1xQueue Sensitivity less than past queue
            # 1.0 when current queue is 1xQueue Sensitivity more than past queue
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
        """Step representing the Environment alteration.

Jobs are mapped into available resources and further communicated to Batsim. If ``void`` action
is selected, on scheduling occurs.

Args:
    action (int):
        Action ID to be applied.
        """

        assert self.action_space.contains(action), f'{action} ({type(action)}) invalid'
        # Check for void action, if so do not do anything
        if action == 25:
            logging.warning('Void action selected!')
        else:
            # Set actions as given by the key
            keys = self.action_keys[action]
            self.workload_manager.job_scheduler.sorting_key = keys[0]
            self.workload_manager.resource_manager.sorting_key = keys[1]
            # Schedule jobs
            self.workload_manager.schedule_jobs()

    def avg_job_slowdown_reward(self) -> float:
        """Reward when the objetive is to minimize average job slowdown.

It is the mean of requested times for all jobs currently in the system. Higher means when longest
jobs in the queue, which indicates that shorter jobs are being scheduled, and thus average job
slowdown is kept low.

Returns:
    Mean of requested times for all jobs in the system.
        """

        pending_jobs_req_time = [
            job.req_time for job in self.workload_manager.job_scheduler.pending_jobs
        ]
        active_jobs_req_time = [
            core.state['served_job'].req_time for core\
            in self.workload_manager.resource_manager.core_pool if core.state['served_job']
        ]
        all_jobs_req_time = np.array([*pending_jobs_req_time, *active_jobs_req_time])
        if not all_jobs_req_time:
            return 0.0
        return np.mean(all_jobs_req_time)

    def avg_job_completion_reward(self) -> float:
        """Reward when the objective is to minimize average job completion time.

It is the number of completed jobs, as more jobs being completed implies shorter completion times.

Returns:
    Number of completed jobs.
        """

        return self.workload_manager.job_scheduler.nb_completed_jobs

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
Reward is the total number of current FLOPs in the data centre. Higher throughputs will lead to
lower makespans.

Returns:
    Current total FLOPs provided by the data centre service.
        """

        return sum([core.state['current_flops'] for core\
                    in self.workload_manager.resource_manager.core_pool])

    def energy_consumption_reward(self) -> float:
        """Reward when the objective is to minimize total energy consumption.

It is the inverse of the current power usage in the data centre. Keeping the power low will decrease
total energy consumed.

Returns:
    Inverse the power usage in the data centre service.
        """

        return 1 / sum([core.state['current_power'] for core\
                        in self.workload_manager.resource_manager.core_pool])

    def render(self, mode='human'):
        pass

    def reset(self):
        pass
