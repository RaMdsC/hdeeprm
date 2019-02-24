"""
Resource class and functionality for defining the Resource Hierarchy in the Decision System.
"""

from pybatsim.batsim.batsim import Job

class Resource:
    """
Resource representing a Core in the Platform. Uniquely identifiable in Batsim. Associated to a
computing capability and a power consumption.

Attributes:
    processor (dict): parent Processor data structure.
        node (dict): parent Node data structure.
            cluster (dict): parent Cluster data structure.
                platform (dict): root Platform data structure.
                    total_nodes (int): total Nodes in the Platform.
                    total_processors (int): total Processors in the Platform.
                    total_cores (int): total Cores in the Platform.
                    job_limits (dict): resource request limits for a Job in the Platform.
                        max_time (int): maximum requested time.
                        max_core (int): maximum requested Cores.
                        max_mem (int): maximum requested memory.
                        max_mem_bw (int): maximum requested memory bandwidth.
                        reference_machine (dict): reference host for measures.
                            clock_rate (float): machine clock speed.
                            dpflop_vector_width (int): width of vector operations
                                (in 32-byte blocks).
                    reference_speed (float): speed to tranform time into operations.
                    clusters (list): reference to Clusters inside the Platform.
                local_nodes (list): reference to local Nodes to the Cluster.
            max_mem (int): maximum memory capacity of the Node.
            current_mem (int): current memory capacity of the Node.
            local_processors (list): reference to local Processors to the Node.
        max_mem_bw (float): maximum memory BW capacity of the Processor.
        current_mem_bw (float): current memory BW capacity of the Processor.
        flops_per_core (float): maximum FLOPs per Core in the Processor.
        power_per_core (float): maximum Watts per Core in the Processor.
        local_cores (list): reference to local Cores to the Processor.
    bs_id (int): unique identification. Also used in Batsim.
    state (dict): defines the current state of the Resource.
        pstate (int): P-state for the Core.
        current_flops (float): current computing capability in FLOPs.
        current_power (float): current power consumption in Watts.
        served_job (Job): Job being served by the Resource.
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
