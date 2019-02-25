"""
Resource class and functionality for defining the Resource Hierarchy in the Decision System.
"""

from batsim.batsim import Job

class Resource:
    """Resource representing a Core in the Platform.

Resources process Jobs inside the Platform. They are uniquely identifiable in Batsim and provide a
computing capability for a given power consumption.

Attributes:
    processor (``dict``):
        Parent Processor data structure. Data fields:
            node (``dict``):
                Parent Node data structure. Data fields:
                    cluster (``dict``):
                        Parent Cluster data structure. Data fields:
                            platform (``dict``):
                                Root Platform data structure. Data fields:
                                    total_nodes (``int``):
                                        Total Nodes in the Platform.
                                    total_processors (``int``):
                                        Total Processors in the Platform.
                                    total_cores (``int``):
                                        Total Cores in the Platform.
                                    job_limits (``dict``):
                                        Resource request limits for any Job. Data fields:
                                            max_time (``int``):
                                                Maximum requested time in seconds.
                                            max_core (``int``):
                                                Maximum requested Cores.
                                            max_mem (``int``):
                                                Maximum requested Memory in MB.
                                            max_mem_bw (``int``):
                                                Maximum requested Memory BW in GB/s.
                                            reference_machine (``dict``):
                                                Reference host for measures. Data fields:
                                                    clock_rate (``float``):
                                                        Machine clock speed.
                                                    dpflop_vector_width (``int``):
                                                        Width of vector operations in 32B blocks.

                                    reference_speed (``float``):
                                        Speed for tranforming time into operations.
                                    clusters (``list(dict)``):
                                        Reference to all Clusters inside the Platform.

                            local_nodes (``list(dict)``):
                                Reference to local Nodes to the Cluster.

                    max_mem (``int``):
                        Maximum memory capacity of the Node in MB.
                    current_mem (``int``):
                        Current memory capacity of the Node in MB.
                    local_processors (``list(dict)``):
                        Reference to local Processors to the Node.

            max_mem_bw (``float``):
                Maximum memory BW capacity of the Processor in GB/s.
            current_mem_bw (``float``):
                Current memory BW capacity of the Processor in GB/s.
            flops_per_core (``float``):
                Maximum FLOPs per Core in the Processor.
            power_per_core (``float``):
                Maximum Watts per Core in the Processor.
            local_cores (``list(:class:`.Resource`)``):
                Reference to local Cores to the Processor.

    bs_id (``int``):S
        Unique identification. Also used in Batsim.
    state (``dict``):
        Defines the current state of the Resource. Data fields:
            pstate (``int``):
                P-state for the Core.
            current_flops (``float``): current computing capability in FLOPs.
            current_power (``float``): current power consumption in Watts.
            served_job (``Job``): Job being served by the Resource.
    """

    def __init__(self, processor: dict, bs_id: int) -> None:
        self.processor = processor
        self.bs_id = bs_id
        # By default, core is idle
        self.state = {
            'pstate': 3,
            'current_flops': 0.0,
            'current_power': 0.05 * self.processor['power_per_core'],
            # When active, the Resource is serving a Job which is stored as part of its state
            # Remaining operations and updates along simulation are tracked
            'served_job': None
        }

    def set_state(self, new_pstate: int, now: float, new_served_job: Job = None) -> None:
        """
Sets the state of the Resource. It dictates the availability, computing speed and power consumption.

Args:
    new_pstate: new P-state for the Resource.
    now: current simulation time in seconds.
    new_served_job: reference to the Job now being served by the Resource.

Returns:
    None
        """

        # Active core
        if new_pstate in (0, 1):
            if not self.state['served_job']:
                new_served_job.last_update = now
                new_served_job.remaining_ops = new_served_job.req_ops
                self.state['served_job'] = new_served_job
                self.processor['current_mem_bw'] -= new_served_job.mem_bw
                self.processor['node']['current_mem'] -= new_served_job.mem
            else:
                # Update the completion state
                self.update_completion(now)
            # 100% Power
            self.state['current_power'] = self.processor['power_per_core']
            if new_pstate == 0:
                # 100% FLOPS
                self.state['current_flops'] = self.processor['flops_per_core']
            else:
                # 75% FLOPS
                self.state['current_flops'] = 0.75 * self.processor['flops_per_core']
        # Inactive core
        elif new_pstate in (2, 3):
            if self.state['served_job']:
                self.processor['current_mem_bw'] += self.state['served_job'].mem_bw
                self.processor['node']['current_mem'] += self.state['served_job'].mem
                self.state['served_job'] = None
            # 0% FLOPS
            self.state['current_flops'] = 0.0
            if new_pstate == 2:
                # 15% Power
                self.state['current_power'] = 0.15 * self.processor['power_per_core']
            else:
                # 5% Power
                self.state['current_power'] = 0.05 * self.processor['power_per_core']
        else:
            raise ValueError('Error: unknown P-state')
        self.state['pstate'] = new_pstate

    def update_completion(self, now: float) -> None:
        """
Updates the Job operations left.

Args:
    now: current simulation time in seconds.

Returns:
    None
        """

        time_delta = now - self.state['served_job'].last_update
        self.state['served_job'].remaining_ops -= self.state['current_flops'] * time_delta
        self.state['served_job'].last_update = now

    def get_remaining_per(self) -> None:
        """
Provides the remaining percentage of the Job being served.

Args:
    None

Returns:
    None
        """

        return self.state['served_job'].remaining_ops / self.state['served_job'].req_ops
