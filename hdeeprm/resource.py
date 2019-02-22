"""
Defines all the resources in the heterogeneous platform.
Platform: root resource, contains rest of resources.
Cluster: contains a set of Nodes.
Node: provides shared memory and a set of Processors.
Processor: provides bandwidth for accessing memory and a set of Cores.
Core: basic computing resource.
"""

import logging

class Platform:
    """
    Platform resource. Heterogeneous data centre.
    """

    def __init__(self, job_limits):
        # Resource totals
        self.total_nodes = 0
        self.total_processors = 0
        self.total_cores = 0
        # Job limits
        self.max_time_per_job = job_limits['max_time']
        self.max_core_per_job = job_limits['max_core']
        self.max_mem_per_job = job_limits['max_mem']
        self.max_mem_bw_per_job = job_limits['max_mem_bw']
        # Reference speed
        # Based on 2.75 GHz 8-wide vector machine
        ref_machine = job_limits['reference_machine']
        self.reference_speed = ref_machine['clock_rate'] * ref_machine['dpflop_vector_width']
        # Store a reference to the Clusters
        self.local_clusters = []

class Cluster:
    """
    Cluster resource. Zone with a set of nodes within the DC.
    """

    def __init__(self, platform):
        self.platform = platform
        # Store a reference to local Nodes
        self.local_nodes = []

class Node:
    """
    Node resource. Server with a set of processors and memory capacity.
    """

    def __init__(self, cluster, mem):
        self.cluster = cluster
        self.max_mem = mem
        self.current_mem = mem
        # Store a reference to local Processors
        self.local_processors = []

class Processor:
    """
    Processor resource. CPU, GPU or other with a set of computing cores and
    memory access bandwidth.
    """

    def __init__(self, node, mem_bw, flops_per_core, power_per_core):
        self.node = node
        self.max_mem_bw = mem_bw
        self.current_mem_bw = mem_bw
        self.flops_per_core = flops_per_core
        self.power_per_core = power_per_core
        # Store a reference to local Cores
        self.local_cores = []

class Core:
    """
    Core resource. Uniquely identifiable in Batsim.
    Associated to a computing capability and a power consumption.
    """

    def __init__(self, processor, bs_id):
        self.processor = processor
        self.bs_id = bs_id
        # By default, core is idle
        self.pstate = 3
        self.current_flops = 0.0
        self.current_power = 0.05 * self.processor.power_per_core
        # Served job, when core inactive this is None
        self.served_job = None
        # State tracking variables
        self.last_update = None
        self.remaining_ops = None

    def set_state(self, new_pstate, now, new_served_job=None):
        """
        Sets the P-state of the core.
        P-state dictates the availability, computing speed and power
        consumption.
        """

        logging.debug('Core %s, state %s', self.bs_id, new_pstate)
        # Active core
        if new_pstate == 0 or new_pstate == 1:
            if not self.served_job:
                self.last_update = now
                self.served_job = new_served_job
                self.remaining_ops = new_served_job.req_ops
                self.processor.current_mem_bw -= self.served_job.mem_bw
                logging.debug('From set_state - Processor: %s', self.processor)
                logging.debug('From set_state - Processor id list: %s', [core.bs_id for core in self.processor.local_cores])
                logging.debug('From set_state - Proc current memBW: %s', self.processor.current_mem_bw)
                self.processor.node.current_mem -= self.served_job.mem
            else:
                # Update the completion state
                self.update_completion(now)
            # 100% Power
            self.current_power = self.processor.power_per_core
        if new_pstate == 0:
            # 100% FLOPS
            self.current_flops = self.processor.flops_per_core
        if new_pstate == 1:
            # 75% FLOPS
            self.current_flops = 0.75 * self.processor.flops_per_core
        # Inactive core
        if new_pstate == 2 or new_pstate == 3:
            # 0% FLOPS
            self.current_flops = 0.0
            if self.served_job:
                self.processor.current_mem_bw += self.served_job.mem_bw
                self.processor.node.current_mem += self.served_job.mem
                self.served_job = None
        if new_pstate == 2:
            # 15% Power
            self.current_power = 0.15 * self.processor.power_per_core
        if new_pstate == 3:
            # 5% Power
            self.current_power = 0.05 * self.processor.power_per_core
        if new_pstate not in range(4):
            raise ValueError('Error: unknown p-state')
        self.pstate = new_pstate

    def update_completion(self, now):
        """
        Updates the job operations left.
        """
        time_delta = now - self.last_update
        self.remaining_ops -= self.current_flops * time_delta
        self.last_update = now

    def get_remaining_per(self):
        """
        Provides the remaining percentage of the job being served.
        """
        return self.remaining_ops / self.served_job.req_ops
