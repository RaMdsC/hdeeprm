"""
A basic Workload Manager for heterogeneous Platforms.
"""

import json
import logging
import random
import numpy as np
from procset import ProcSet
from batsim.batsim import BatsimScheduler, Job
from hdeeprm.manager import JobScheduler, ResourceManager

class BaseWorkloadManager(BatsimScheduler):
    """Entrypoint for classic and non-Reinforcement Learning experimentation.

It provides a Job Scheduler for Job selection and a Resource Manager for resource selection and
state management. It handles fundamental events coming from Batsim such as Job submissions and
Job completions, while it also orchestrates the Job to Core mapping and sends updates about Core
states to the simulation.

Attributes:
    job_scheduler (:class:`~hdeeprm.manager.JobScheduler`):
        It manages the Job Queue and selects pending Jobs.
    resource_manager (:class:`~hdeeprm.manager.ResourceManager`):
        Selects Cores from the Core Pool and maintains their states, including shared resource
        conflicts.
    scheduled_step (dict):
        Statistics about number of scheduled Jobs per decision step.
    """

    def __init__(self, options: dict) -> None:
        super().__init__(options)
        # Reference to Batsim proxy, populated dynamically
        self.bs = None
        # Set random seed for reproducibility
        seed = int(options['seed'])
        random.seed(seed)
        np.random.seed(seed)
        # Setup logging
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename='insights.log', filemode='w+',
                            level=getattr(logging, options['log_level']))
        logging.getLogger('batsim').setLevel(logging.CRITICAL)
        self.job_scheduler = JobScheduler()
        self.resource_manager = ResourceManager()
        self.scheduled_step = {
            'max': None,
            'min': None,
            'total': 0,
            'num_steps': 0
        }

    def onJobSubmission(self, job: Job) -> None:
        """Handler triggered when a job has been submitted.

Triggered when receiving a
`JOB_SUBMITTED <https://batsim.readthedocs.io/en/latest/protocol.html#job-submitted>`_ event.
Arriving Jobs are enhanced with HDeepRM parameters specified in the profile field. These are
requested operations, requested time, memory and memory bandwidth. The requested time is estimated
by the user, and differs from the one Batsim will use for processing the Job. The Job is sent to
the Job Scheduler for waiting in the Job Queue.

Args:
    job (batsim.batsim.Job):
        The arriving Job.
        """

        job.req_ops = self.bs.profiles[job.workload][job.profile]['req_ops']
        job.req_time = self.bs.profiles[job.workload][job.profile]['req_time']
        job.mem = self.bs.profiles[job.workload][job.profile]['mem']
        job.mem_bw = self.bs.profiles[job.workload][job.profile]['mem_bw']
        logging.debug('Job arrived: %s %s %s %s %s', job.id, job.req_time, job.req_ops, job.mem,
                      job.mem_bw)
        self.job_scheduler.new_job(job)

    def onJobCompletion(self, job: Job) -> None:
        """Handler triggered when a job has been completed.

When a `JOB_COMPLETED <https://batsim.readthedocs.io/en/latest/protocol.html#job-completed>`_ event
is received, the Job's allocated Cores are freed, thus the Resource Manager updates their state as
well as the one of Cores in the same Processor and/or Node scope.

Args:
    job (batsim.batsim.Job):
        The completed Job.
        """

        # Record memory bandwidth over-utilization releases
        modified, mem_bw_utilization_changes = self.resource_manager.update_state(
            job, list(job.allocation), 'FREE', self.bs.time()
        )
        for proc_id in mem_bw_utilization_changes:
            if self.resource_manager.over_utilization['mem_bw']['procs'][proc_id]['state'] == 1:
                self.resource_manager.over_utilization['mem_bw']['procs'][proc_id]['state'] = 0
                initial_t = self.resource_manager\
                    .over_utilization['mem_bw']['procs'][proc_id]['values'].pop()
                self.resource_manager\
                    .over_utilization['mem_bw']['procs'][proc_id]['values'].append(
                        (initial_t, self.bs.time() - initial_t)
                    )
        self.resource_manager.state_changes = {
            **self.resource_manager.state_changes,
            **modified
        }
        self.job_scheduler.nb_active_jobs -= 1
        self.job_scheduler.nb_completed_jobs += 1

    def onNoMoreEvents(self) -> None:
        """Handler triggered when there are no more events for the time step.

If there are no more events, it means all Jobs have arrived and completed, and thus they have been
handled. The Workload Manager proceeds to schedule the Jobs and send Batsim the resource state
changes.
        """

        if self.bs.running_simulation:
            self.schedule_jobs()
            self.change_resource_states()

    def onSimulationEnds(self) -> None:
        """Handler triggered when the simulation has ended.

It records the over-utilizations during the simulation.
        """

        with open('overutilizations.json', 'w+') as out_f:
            json.dump(self.resource_manager.over_utilization, out_f)

    def schedule_jobs(self) -> None:
        """Maps pending Jobs into available resources.

Looks for the first selected Job from the Job Queue given the Job selection policy, and allocates
Cores given the Job requirements and the Core selection policy. Jobs are scheduled until no more
pending Jobs or no more resources available for the next selected.
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
            self.bs.execute_jobs(scheduled_jobs)

    def change_resource_states(self) -> None:
        """Sends resource state changes to Batsim.

This alters the Cores P-states in the simulation, thus affecting computational capability and power
consumption.
        """

        for pstate in (0, 1, 2, 3):
            resources = [i for i, s in self.resource_manager.state_changes.items() if s == pstate]
            if resources:
                self.bs.set_resource_state(ProcSet(*resources), pstate)
        # Reset for next decision step
        self.resource_manager.state_changes = {}

    def onAddResources(self, to_add):
        """Not used."""

    def onAnswerAirTemperatureAll(self, air_temperature_all):
        """Not used."""

    def onAnswerProcessorTemperatureAll(self, proc_temperature_all):
        """Not used."""

    def onJobMessage(self, timestamp, job, message):
        """Not used."""

    def onJobsKilled(self, jobs):
        """Not used."""

    def onMachinePStateChanged(self, nodeid, pstate):
        """Not used."""

    def onRemoveResources(self, to_remove):
        """Not used."""

    def onReportEnergyConsumed(self, consumed_energy):
        """Not used."""

    def onRequestedCall(self):
        """Not used."""
