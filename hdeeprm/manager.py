"""
Defines HDeepRM managers, which are in charge of mapping Jobs to resources.
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
    peeked_job (batsim.batsim.Job):
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
    job (batsim.batsim.Job):
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
    """Selects Cores and maintains Core states for serving incoming Jobs.

The Core selection policy is defined by the sorting key. The Core Pool is filtered by this key
to obtain the required resources by the Job. Cores have a state, which describes their
availability as well as computational capability and power consumption. Core selection might
fail if there are not enough available resources for the selected Job.

Attributes:
    state_changes (dict):
        Maps Cores to P-state changes for sending to Batsim
    platform (dict):
        Resource Hierarchy for relations. See :class:`~hdeeprm.resource.Core` for fields.
    core_pool (list):
        Contains all Cores in the Platform for filtering.
    over_utilization (dict):
        Tracks over-utilizations of cores, memory capacity and memory bandwidth.
    sorting_key (function):
        Key defining the Core selection policy.
    """

    def __init__(self) -> None:
        self.state_changes = {}
        with open('./res_hierarchy.pkl', 'rb') as in_f:
            self.platform, self.core_pool = pickle.load(in_f)
        logging.info('Reference speed %s', self.platform['reference_speed'])
        # Add the job resource requirement limits to the resource hierarchy
        with open('./job_limits.pkl', 'rb') as in_f:
            self.platform['job_limits'] = pickle.load(in_f)
        self.over_utilization = {
            'core': [],
            'mem': [],
            'mem_bw': {
                'nb_procs': self.platform['total_processors'],
                'procs': {}
            } 
        }
        self.sorting_key = None

    def get_resources(self, job: Job, now: float) -> ProcSet:
        """Gets a set of resources for the selected job.

State of resources change as they are being selected.

Args:
    job (batsim.batsim.Job):
        Job to be served by the selected Cores. Used for checking requirements.
    now (float):
        Current simulation time in seconds.

Returns:
    Set of Cores as a :class:`~procset.ProcSet`. None if not enough resources available.
        """

        # Save the temporarily selected cores. We might not be able to provide the
        # asked amount of cores, so we would need to return them back
        selected = []
        # Also save a reference to modified IDs to later communicate their changes
        # to Batsim
        modified = {}
        # Record memory bandwidth utilization changes
        mem_bw_utilization_changes = set()
        available = [core for core in self.core_pool if not core.state['served_job']]
        if len(available) < job.requested_resources:
            self.over_utilization['core'].append(now)
            return None
        for _ in range(job.requested_resources):
            mem_available = [
                core for core in available if core.processor['node']['current_mem'] >= job.mem
            ]
            # Sorting is needed everytime we access since completed jobs or assigned cores
            # might have changed the state
            if self.sorting_key:
                mem_available.sort(key=self.sorting_key)
            if mem_available:
                if not self.sorting_key:
                    selected_id = random.choice(mem_available).bs_id
                else:
                    selected_id = mem_available[0].bs_id
                # Update the core state
                _modified, _mem_bw_utilization_changes = self.update_state(job, [selected_id],
                                                                           'LOCKED', now)
                modified = {**modified, **_modified}
                mem_bw_utilization_changes = mem_bw_utilization_changes.union(
                    _mem_bw_utilization_changes)
                # Add its ID to the temporarily selected core buffer
                selected.append(selected_id)
                available = [core for core in available if core.bs_id not in selected]
            else:
                # There are no sufficient resources, revert the state of the
                # temporarily selected
                self.update_state(job, selected, 'FREE', now)
                self.over_utilization['mem'].append(now)
                return None
        # Check and record memory bandwidth over-utilizations
        for proc_id in mem_bw_utilization_changes:
            if not proc_id in self.over_utilization['mem_bw']['procs']:
                self.over_utilization['mem_bw']['procs'][proc_id] = {'state': 1, 'values': [now]}
            elif self.over_utilization['mem_bw']['procs'][proc_id]['state'] == 0:
                self.over_utilization['mem_bw']['procs'][proc_id]['state'] = 1
                self.over_utilization['mem_bw']['procs'][proc_id]['values'].append(now)
        # Store modifications for commit
        self.state_changes = {**self.state_changes, **modified}
        return ProcSet(*selected)

    def update_state(self, job: Job, id_list: list, new_state: str, now: float) -> tuple:
        """Modifies the state of the computing resources.

This affects speed, power and availability for selection. Modifications are local to the Decision
System until communicated to the Simulator. Modifying the state of a Core might trigger alterations
on Cores in the same Processor or Node scope due to shared resources.

Args:
    job (batsim.batsim.Job):
        Job served by the selected Cores. Used for updating resource contention.
    id_list (list):
        IDs of the Cores to be updated directly.
    new_state (str):
        Either "LOCKED" (makes Cores unavailable) or "FREE" (makes Cores available).
    now (float):
        Current simulation time in seconds.

Returns:
    A dictionary with all directly and indirectly modified Cores. Keys are the Cores IDs, values are
    the new P-states.
        """

        # Modify states of the cores
        # Associate each affected core with a new P-State
        modified = {}
        # Record memory bandwidth utlization changes
        mem_bw_utilization_changes = set()
        for bs_id in id_list:
            score = self.core_pool[bs_id]
            processor = score.processor
            if new_state == 'LOCKED':
                score.set_state(0, now, job)
                modified[score.bs_id] = 0
                for lcore in processor['local_cores']:
                    # If this is the first active core in the processor,
                    # set the state of the rest of cores to 2 (indirect energy consumption)
                    if lcore.state['pstate'] == 3:
                        lcore.set_state(2, now)
                        modified[lcore.bs_id] = 2
                    # If the memory bandwidth capacity is now overutilized,
                    # transition every active core of the processor into state 1 (reduced GFLOPS)
                    if processor['current_mem_bw'] < 0.0 and lcore.state['pstate'] == 0:
                        mem_bw_utilization_changes.add(processor['id'])
                        lcore.set_state(1, now)
                        modified[lcore.bs_id] = 1
            elif new_state == 'FREE':
                score.set_state(2, now)
                modified[score.bs_id] = 2
                all_inactive = all(not lcore.state['served_job']
                                   for lcore in processor['local_cores'])
                for lcore in processor['local_cores']:
                    # If this was the last core being utilized, lower all
                    # cores of processor from indirect energy consuming
                    if all_inactive:
                        lcore.set_state(3, now)
                        modified[lcore.bs_id] = 3
                    # If bandwidth is now not overutilized, scale
                    # to full potential (P0) other active cores
                    if processor['current_mem_bw'] >= 0.0 and lcore.state['pstate'] == 1:
                        mem_bw_utilization_changes.add(processor['id'])
                        lcore.set_state(0, now)
                        modified[lcore.bs_id] = 0
            else:
                raise ValueError('Error: unknown state')
        return modified, mem_bw_utilization_changes
