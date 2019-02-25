"""
Defines HDeepRM managers, which are in charge of mapping Jobs into Resources.
"""

import logging
import pickle
import random
from procset import ProcSet
from batsim.batsim import Job

class JobScheduler:
    """Selects Jobs from the Job Queue to be processed in the Platform.

The Job selection policy is defined by the sorting key. Only one job is peeked in order to check
for sufficient resources available for it.

Attributes:
    pending_jobs (list):
        Job Queue. All incoming jobs arrive in this data structure.
    nb_active_jobs (int):
        Number of Jobs being served by the Platform.
    nb_completed_jobs (int):
        Number of Jobs already served by the Platform.
    peeked_job (Job):
        Cached next Job to be processed. This saves sorting the Job Queue a second time.
    sorting_key (function):
        Key defining the Job selection policy.
    """

    def __init__(self) -> None:
        self.pending_jobs = []
        self.nb_active_jobs = 0
        self.nb_completed_jobs = 0
        self.peeked_job = None
        self.sorting_key = None

    def peek_job(self) -> Job:
        """Returns a reference to the first selected job.

This is the first Job to be processed given the selected policy. This method does not remove the
Job from the Job Queue.

Returns:
    The reference to the first selected Job.
        """

        if not self.sorting_key:
            self.peeked_job = random.choice(self.pending_jobs)
        else:
            self.pending_jobs.sort(key=self.sorting_key)
            self.peeked_job = self.pending_jobs[0]
        return self.peeked_job

    def new_job(self, job: Job) -> None:
        """Inserts a new job in the queue.

By default, it is appended to the right end of the queue.

Args:
    job (Job):
        Incoming Job to be inserted into the Job Queue.
        """
        self.pending_jobs.append(job)

    def remove_job(self) -> None:
        """Removes the first selected job from the queue.

It uses the cached peeked Job for removal.
        """
        self.pending_jobs.remove(self.peeked_job)

    @property
    def nb_pending_jobs(self) -> int:
        """int: Number of pending jobs, equal to the current length of the queue."""
        return len(self.pending_jobs)

class ResourceManager:
    """Selects Resources and maintains Resource states for serving incoming Jobs.

The Resource selection policy is defined by the sorting key. The Core Pool is filtered by this key
to obtain the required Resources by the Job. Resources have a state, which describes their
availability as well as computational capability and power consumption. Resource selection might
fail if there are not enough available Resources for the selected Job.

Attributes:
    state_changes (dict):
        Maps Resources to P-state changes for sending to Batsim
    platform (dict):
        Resource Hierarchy for relations. See :class:`~hdeeprm.resource.Resource` for fields.
    core_pool (list):
        Contains all Resources (Cores) in the Platform for filtering.
    sorting_key (function):
        Key defining the Resource selection policy.
    """

    def __init__(self) -> None:
        # Store the PState changes for the resources
        self.state_changes = {}
        with open('./res_hierarchy.pkl', 'rb') as in_f:
            self.platform, self.core_pool = pickle.load(in_f)
        # Used for sorting the resources
        self.sorting_key = None

    def get_resources(self, job: Job, now: float) -> ProcSet:
        """Gets a set of resources for the selected job.

State of resources change as they are being selected.

Args:
    job (Job):
        Job to be served by the selected Resources. Used for checking requirements.
    now (float):
        Current simulation time in seconds.

Returns:
    Set of Resources as a :class:`procset.ProcSet`. None if not enough Resources available.
        """

        # Save the temporarily selected cores. We might not be able to provide the
        # asked amount of cores, so we would need to return them back
        selected = []
        # Also save a reference to modified IDs to later communicate their changes
        # to Batsim
        modified = {}
        for _ in range(job.requested_resources):
            available = [core for core in self.core_pool if not core.served_job and\
                         core.processor.node.current_mem >= job.mem]
            # Sorting is needed everytime we access since completed jobs or assigned resources
            # might have changed the state
            if self.sorting_key:
                available.sort(key=self.sorting_key)
            if available:
                if not self.sorting_key:
                    selected_id = random.choice(available).bs_id
                else:
                    selected_id = available[0].bs_id
                # Update the core state
                modified = {**modified, **self.update_state(job, [selected_id], 'LOCKED', now)}
                # Add its ID to the temporarily selected core buffer
                selected.append(selected_id)
            else:
                # There are no sufficient resources, return the state of the
                # temporarily selected
                self.update_state(job, selected, 'FREE', now)
                return None
        # Store modifications for commit
        self.state_changes = {**self.state_changes, **modified}
        return ProcSet(*selected)

    def update_state(self, job: Job, id_list: list, new_state: int, now: float) -> dict:
        """Modifies the state of the computing resources.
        This affects speed, power and availability for selection.
        Modifications are local to the Decision System until communicated
        to the Simulator.
        """

        # Modify states of the cores
        # We associate each affected core with a new P-State
        modified = {}
        logging.debug('From update_state - Resource ID list: %s', id_list)
        for bs_id in id_list:
            resource = self.core_pool[bs_id]
            processor = resource.processor
            if new_state == 'LOCKED':
                resource.set_state(0, now, job)
                modified[resource.bs_id] = 0
                for core in processor.local_cores:
                    # If this is the first active core in the processor,
                    # set the state of the rest of cores to 2 (indirect energy consumption)
                    if core.pstate == 3:
                        core.set_state(2, now)
                        modified[core.bs_id] = 2
                    # If the memory bandwidth capacity is now overutilized,
                    # transition every active core of the processor into state 1 (reduced FLOPS)
                    if processor.current_mem_bw < 0.0 and core.pstate == 0:
                        logging.warning('Memory bandwidth overutilized!')
                        core.set_state(1, now)
                        modified[core.bs_id] = 1
            elif new_state == 'FREE':
                resource.set_state(2, now)
                modified[resource.bs_id] = 2
                all_inactive = all(not core.served_job for core in processor.local_cores)
                for core in processor.local_cores:
                    # If this was the last core being utilized, lower all
                    # cores of processor from indirect energy consuming
                    if all_inactive:
                        core.set_state(3, now)
                        modified[core.bs_id] = 3
                    # If bandwidth is now not overutilized, scale
                    # to full potential (P0) other active cores
                    if processor.current_mem_bw >= 0.0 and core.pstate == 1:
                        core.set_state(0, now)
                        modified[core.bs_id] = 0
            else:
                print('Error: unknown state')
                raise ValueError
        return modified
