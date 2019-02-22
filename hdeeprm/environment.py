"""
Environment for the Workload Management task.
This is compliant with OpenAI gym environment format.
"""

import logging
import gym
import gym.spaces
import numpy as np

class HDeepRMEnv(gym.Env):
    """
    Description
        Jobs arrive at an HPC heterogeneous cluster. A workload manager is in charge
        of scheduling the jobs over the available computing resources.

    Observation space
        Composed of the following data fields:
        - % of available memory in each node
        - % of available memory bandwidth in each processor
        - % of working FLOPS and working Power for each core
        - % left for completing the served job by the core
        - % with respect to the maximum values of requested time/cores/mem/mem_bw
          for pending jobs; we show five percentiles (min, Q1, med, Q3, max) such
          that the network can devise a job distribution
        - % variability of job queue size with respect to last observation

    Action space
        There are 26 possible actions
        Num     Action
        0       Random job, random resource
        1       Random job, highest computing capability resource
        2       Random job, less memory conflicted resource
        3       Random job, less memory bandwidth conflicted resource
        4       Random job, lowest energy consuming resource
        5       First arrived job, random resource
        6       First arrived job, highest computing capability resource
        7       First arrived job, less memory conflicted resource
        8       First arrived job, less memory bandwidth conflicted resource
        9       First arrived job, lowest energy consuming resource
        10      Shortest job, random resource
        11      Shortest job, highest computing capability resource
        12      Shortest job, less memory conflicted resource
        13      Shortest job, less memory bandwidth conflicted resource
        14      Shortest job, lowest energy consuming resource
        15      Lowest memory job, random resource
        16      Lowest memory job, highest computing capability resource
        17      Lowest memory job, less memory conflicted resource
        18      Lowest memory job, less memory bandwidth conflicted resource
        19      Lowest memory job, lowest energy consuming resource
        20      Lowest memory bandwidth job, random resource
        21      Lowest memory bandwidth job, highest computing capability resource
        22      Lowest memory bandwidth job, less memory conflicted resource
        23      Lowest memory bandwidth job, less memory bandwidth conflicted resource
        24      Lowest memory bandwidth job, lowest energy consuming resource
        25      Void action, reserve jobs for next scheduling round
    """

    def __init__(self, workload_manager, objective, queue_sensitivity):
        # These are references to decision system elements
        # required for the environment progress
        self.workload_manager = workload_manager
        self.job_scheduler = workload_manager.job_scheduler
        self.resource_manager = workload_manager.resource_manager

        # Action space
        # Agent may choose between 26 policies
        # These are implemented by varying the sorting key of data structures
        self.action_space = gym.spaces.Discrete(26)
        self.action_keys = [
            (None, None),
            (None, lambda res: - res.processor.flops_per_core),
            (None, lambda res: - res.processor.node.current_mem),
            (None, lambda res: - res.processor.current_mem_bw),
            (None, lambda res: res.processor.power_per_core),
            (lambda job: job.submit_time, None),
            (lambda job: job.submit_time, lambda res: - res.processor.flops_per_core),
            (lambda job: job.submit_time, lambda res: - res.processor.node.current_mem),
            (lambda job: job.submit_time, lambda res: - res.processor.current_mem_bw),
            (lambda job: job.submit_time, lambda res: res.processor.power_per_core),
            (lambda job: job.req_time, None),
            (lambda job: job.req_time, lambda res: - res.processor.flops_per_core),
            (lambda job: job.req_time, lambda res: - res.processor.node.current_mem),
            (lambda job: job.req_time, lambda res: - res.processor.current_mem_bw),
            (lambda job: job.req_time, lambda res: res.processor.power_per_core),
            (lambda job: job.mem, None),
            (lambda job: job.mem, lambda res: - res.processor.flops_per_core),
            (lambda job: job.mem, lambda res: - res.processor.node.current_mem),
            (lambda job: job.mem, lambda res: - res.processor.current_mem_bw),
            (lambda job: job.mem, lambda res: res.processor.power_per_core),
            (lambda job: job.mem_bw, None),
            (lambda job: job.mem_bw, lambda res: - res.processor.flops_per_core),
            (lambda job: job.mem_bw, lambda res: - res.processor.node.current_mem),
            (lambda job: job.mem_bw, lambda res: - res.processor.current_mem_bw),
            (lambda job: job.mem_bw, lambda res: res.processor.power_per_core)
        ]
        self.action_size = self.action_space.n

        # Observation space
        # Composed of different data from the data centre state
        self.observation_size = self.resource_manager.platform.total_nodes +\
                                self.resource_manager.platform.total_processors +\
                                self.resource_manager.platform.total_cores * 3 + 21
        self.observation_space = gym.spaces.Box(low=np.zeros(self.observation_size, dtype=np.float32),
                                                high=np.ones(self.observation_size, dtype=np.float32),
                                                dtype=np.float32)
        self.observation = None
        # Sets the correspondent reward function based on the objective
        objective_to_reward = {
            'avg_job_slowdown': self.avg_job_slowdown_reward,
            'avg_job_completion': self.avg_job_completion_reward,
            'utilization': self.utilization_reward,
            'makespan': self.makespan_reward
        }
        self.get_reward = objective_to_reward[objective]
        
        # Queue sensitivity. Determines sensitivity of the agent to variations in queue size.
        # If sensitivity is high, larger variations will be noticed, however smaller
        # ones will not have much impact.
        # If sensitivity is low, smaller variations will be noticed and large ones
        # will be clipped, thus impactless.
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
