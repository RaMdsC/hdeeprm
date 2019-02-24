"""
Defines HDeepRM managers, which are in charge of mapping Jobs into Resources.
"""

import logging
import pickle as pkl
import random
from procset import ProcSet

class JobScheduler:
    """
    Base Job Scheduler.
    """

    def __init__(self):
        self.pending_jobs = []
        self.nb_active_jobs = 0
        self.nb_completed_jobs = 0
        self.peeked_job = None
        self.sorting_key = None

    def peek_job(self):
        """
        Returns a reference to the first selected job
        without removing it from the queue.
        """
        if not self.sorting_key:
            self.peeked_job = random.choice(self.pending_jobs)
        else:
            self.pending_jobs.sort(key=self.sorting_key)
            self.peeked_job = self.pending_jobs[0]
        return self.peeked_job

    def new_job(self, job):
        """
        Inserts a new job in the queue.
        """
        self.pending_jobs.append(job)

    def remove_job(self):
        """
        Removes the first selected job from the queue.
        """
        self.pending_jobs.remove(self.peeked_job)

    @property
    def nb_pending_jobs(self):
        """
        Number of pending jobs, equal to the current length of the queue.
        """
        return len(self.pending_jobs)

class ResourceManager:
    """
    Base Resource Manager.
    """

    def __init__(self):
        # Store the PState changes for the resources
        self.state_changes = {}
        with open('./res_hierarchy.pkl', 'rb') as in_f:
            self.platform, self.core_pool = pkl.load(in_f)
        # Used for sorting the resources
        self.sorting_key = None

    def get_resources(self, job, now):
        """
        Gets the set of resources for the selected job.
        State of resources change as they are being selected.
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
        logging.debug('Job %s, allocated %s', job.id, selected)
        logging.debug('Res FLOPS %s', [self.core_pool[sid].current_flops for sid in selected])
        return ProcSet(*selected)

    def update_state(self, job, id_list, new_state, now):
        """
        Modifies the state of the computing resources.
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
