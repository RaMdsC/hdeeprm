"""
Defines the superclass for all Workload Managers.
"""

import logging
import random
import numpy as np
from pybatsim.batsim.batsim import BatsimScheduler
from hdeeprm.manager import JobScheduler, ResourceManager

class BaseWorkloadManager(BatsimScheduler):
    """
    Base Workload Manager.
    """

    def __init__(self, options):
        super().__init__(options)
        self.bs = None
        # Seed the PRNGs of the libraries
        self.seed = int(options['seed'])
        self.set_seed()
        # Job scheduler and resource manager
        self.job_scheduler = JobScheduler()
        self.resource_manager = ResourceManager()
        # Create the logger
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename='insights.log', level=getattr(logging, options['log_level']))
        # Stats about jobs scheduled every decision step
        self.scheduled_step = {}
        self.scheduled_step['max'] = None
        self.scheduled_step['min'] = None
        self.scheduled_step['total'] = 0
        self.scheduled_step['num_steps'] = 0

    def onJobSubmission(self, job):
        # Enhance the job with requested ops and memory information
        # This is the user estimated ops and time for running. It differs from
        # the actual running time in the Batsim profile. The job scheduler
        # uses it to make decisions
        logging.info('Job submitted')
        job.req_ops = self.bs.profiles[job.workload][job.profile]['req_ops']
        job.req_time = self.bs.profiles[job.workload][job.profile]['req_time']
        job.mem = self.bs.profiles[job.workload][job.profile]['mem']
        job.mem_bw = self.bs.profiles[job.workload][job.profile]['mem_bw']
        self.job_scheduler.new_job(job)

    def onJobCompletion(self, job):
        # Update the individual states of each newly available core
        logging.info('Job completed')
        logging.debug('Res FLOPS %s', [self.resource_manager.core_pool[alloc_id].current_flops\
                                       for alloc_id in job.allocation])
        self.resource_manager.state_changes = {**self.resource_manager.state_changes,
                                               **self.resource_manager.update_state(job,
                                                                                    list(job.allocation),
                                                                                    'FREE',
                                                                                    self.bs.time())}
        self.job_scheduler.nb_active_jobs -= 1
        self.job_scheduler.nb_completed_jobs += 1

    def onJobMessage(self, timestamp, job, message):
        pass

    def onJobsKilled(self, jobs):
        pass

    def onMachinePStateChanged(self, nodeid, pstate):
        pass

    def onReportEnergyConsumed(self, consumed_energy):
        pass

    def onAnswerProcessorTemperatureAll(self, proc_temperature_all):
        pass

    def onAnswerAirTemperatureAll(self, air_temperature_all):
        pass

    def onAddResources(self, to_add):
        pass

    def onRemoveResources(self, to_remove):
        pass

    def onRequestedCall(self):
        pass

    def onNoMoreEvents(self):
        # All events in Batsim message have been processed
        if self.bs.running_simulation:
            self.schedule_jobs()
            self.change_resource_states()

    def set_seed(self):
        """
        Sets the random seed for reproducibility.
        """

        random.seed(self.seed)
        np.random.seed(self.seed)

    def schedule_jobs(self):
        """
        Select jobs for scheduling.
        Looks for the first selected job from the queue and allocates
        resources. Does this until no resources are available for next job.
        """

        scheduled_jobs = []
        serviceable = True
        while(self.job_scheduler.nb_pending_jobs and serviceable):
            job = self.job_scheduler.peek_job()
            # Pass the current timestamp for registering job entrance in the resource
            resources = self.resource_manager.get_resources(job, self.bs.time())
            if not resources:
                serviceable = False
            else:
                job.allocation = resources
                scheduled_jobs.append(job)
                self.job_scheduler.remove_job()

        # Execute the jobs if they exist
        if scheduled_jobs:
            # Update scheduled per step metrics
            if not self.scheduled_step['max'] or len(scheduled_jobs) > self.scheduled_step['max']:
                self.scheduled_step['max'] = len(scheduled_jobs)
            if not self.scheduled_step['min'] or len(scheduled_jobs) < self.scheduled_step['min']:
                self.scheduled_step['min'] = len(scheduled_jobs)
            self.scheduled_step['total'] += len(scheduled_jobs)
            self.scheduled_step['num_steps'] += 1
            self.job_scheduler.nb_active_jobs += len(scheduled_jobs)
            for scheduled_job in scheduled_jobs:
                logging.debug('Scheduled %s', scheduled_job)
            self.bs.execute_jobs(scheduled_jobs)

    def change_resource_states(self):
        """
        Send resource state changes to Batsim.
        This affects speed and power consumption in the simulation.
        """

        for pstate in (0, 1, 2, 3):
            resources = [i for i, s in self.resource_manager.state_changes.items() if s == pstate]
            if resources:
                self.bs.set_resource_state(resources, pstate)
        self.resource_manager.state_changes = {}
