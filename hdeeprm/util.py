"""
Utilities for parsing and generating Workloads, Platforms and Resource Hierarchies.
"""

import json
import os.path as path
import pickle
import defusedxml.minidom as mxml
import numpy
import numpy.random as nprnd
import hdeeprm.resource as res
from hdeeprm.__xml__ import exml, XMLElement

def generate_workload(workload_file_path: str, nb_resources: int, nb_jobs: int) -> None:
    """SWF-formatted Workload -> Batsim-ready JSON format.

Parses a SWF formatted Workload file into a Batsim-ready JSON file. Generates as many jobs as
specified in "nb_jobs".

Args:
    workload_file_path (str):
        Location of the SWF Workload file in the system.
    nb_resources (int):
        Total number of resources (Cores) in the Platform.
    nb_jobs (int):
        Total number of jobs for the generated Workload.
    """

    with open(workload_file_path, 'r') as in_f:
        # Workload
        workload = {
            'nb_res': nb_resources,
            'jobs': [],
            'profiles': {}
        }
        # Read each Job from the SWF file and parse the fields
        min_submit_time = None
        for job_id, line in zip(range(nb_jobs), in_f):
            # Skip comments
            if line.startswith(';'):
                continue
            job_info = tuple(map(lambda par: int(float(par)), line.split()))
            # Job
            job = {
                'id': job_id,
                'subtime': job_info[1],
                'res': job_info[7],
                'profile': None
            }
            # Profile
            profile = {
                'type': 'parallel_homogeneous',
                'com': 0,
                'cpu': None,
                'req_time': job_info[8],
                'req_ops': None,
                'mem': job_info[9],
                'mem_bw': None
            }
            # Skip if submission time, requested resources, requested time per resource and both
            # memory parameters are not specified. In SWF, this is indicated by - 1.
            # job_info[6] = used_mem
            if any(map(lambda parameter: parameter < 0,
                       (job['subtime'], job['res'], profile['req_time']))):
                continue
            if profile['mem'] < 0:
                if job_info[6] < 0:
                    continue
                else:
                    profile['mem'] = job_info[6]
            # Need to shift the initial submission time by the minimum since the trace does not
            # start at 0
            if not min_submit_time:
                min_submit_time = job['subtime']
            job['subtime'] -= min_submit_time
            # Calculate FLOPs per resource. The original trace provides time per resource, here it
            # is normalized to FLOPs with respect to a 2.75 GHz 8-wide vector CPU (reference
            # machine, 22e9 ops/s)
            profile['req_ops'] = int(22e9 * profile['req_time'])
            # User estimates in Job requested time have been shown to be generally overestimated.
            # The distribution here used is taken as an approximation to that shown in
            # [Tsafrir et al. 2007]
            profile['cpu'] = int(22e9 * nprnd.choice(
                numpy.arange(0.05, 1.3, 0.05),
                p=numpy.array([0.15, 0.09, 0.07] + 5 * [0.04] + 5 *\
                            [0.03] + 10 * [0.02] + [0.06, 0.08])
            ) * profile['req_time'])
            # Original trace provides memory in KB, convert it to MB for framework compatibility
            profile['mem'] = int(profile['mem'] / 1000)
            # Calculate the sustained memory bandwidth requirement per resource. No info on original
            # trace, this is synthetically produced from a random uniform distribution with values
            # (4, 8, 12, 16, 20, 24).
            profile['mem_bw'] = int(nprnd.choice(numpy.arange(4, 25, 4)))
            # Profile name based on parameters. This may be shared between multiple Jobs if they
            # share their requirements.
            job['profile'] = f'{profile["req_time"]}_{profile["mem"]}_{profile["mem_bw"]}'
            workload['profiles'].setdefault(job['profile'], profile)
            workload['jobs'].append(job)
    # Write the data structure into the JSON output
    with open('workload.json', 'w') as out_f:
        json.dump(workload, out_f)

def generate_platform(platform_file_path: str, gen_platform_xml: bool = True,
                      gen_res_hierarchy: bool = False) -> None:
    """HDeepRM JSON Platform -> Batsim-ready XML format + Resource Hierarchy.

Parses a HDeepRM JSON formatted platform definition and outputs both a Batsim-ready XML file
and the Resource Hierarchy pickled for the Decision System to understand relations between Cores,
Processors and Nodes.

Args:
    platform_file_path (str):
        Location of the HDeepRM Platform file in the system.
    gen_platform_xml (bool):
        If ``True``, generate the Platform XML. Defaults to ``True``.
    gen_res_hierarchy (bool):
        If ``True``, generate the Resource Hierarchy. Defaults to ``False``.
    """

    # Define shared state
    shared_state = {
        # Type of resources in the Platform
        'types': None,
        'gen_platform_xml': gen_platform_xml,
        'gen_res_hierarchy': gen_res_hierarchy,
        'counters': {'cluster': 0, 'node': 0, 'core': 0},
        # Core pool for filtering and selecting Cores is initially empty
        'core_pool': [],
        'cluster_xml': None,
        # Need to temporarly store information about up / down routes, since the XML DTD spec
        # imposes setting them at the end of generating all nodes. Initially, these are empty
        'udlink_routes': []
    }
    root_desc, shared_state['types'] = _load_data(platform_file_path)
    if shared_state['gen_platform_xml']:
        root_xml, main_zone_xml = _root_xml()
    if shared_state['gen_res_hierarchy']:
        root_el = _root_el(root_desc)
    _generate_clusters(shared_state, root_desc, root_el, main_zone_xml)
    if shared_state['gen_platform_xml']:
        _global_links(shared_state, root_desc, main_zone_xml)
        _zone_routes(root_desc, main_zone_xml)
        _write_platform_definition(root_xml)
    if shared_state['gen_res_hierarchy']:
        _write_resource_hierarchy(shared_state, root_el)

def _load_data(platform_file_path: str) -> tuple:
    data_path = path.join(path.dirname(__file__), 'data')
    with open(platform_file_path, 'r') as in_f,\
         open(path.join(data_path, 'network_types.json'), 'r') as nt_f,\
         open(path.join(data_path, 'node_types.json'), 'r') as nd_f,\
         open(path.join(data_path, 'processor_types.json'), 'r') as pr_f:
        root_desc = json.load(in_f)
        types = {'network': json.load(nt_f), 'node': json.load(nd_f), 'processor': json.load(pr_f)}
    return root_desc, types

def _root_xml() -> tuple:
    # Generates the 'platform' and 'main zone' XML elements. These contain the 'master zone' and all
    # the clusters.
    root_xml = exml.Element('platform', attrib={'version': '4.1'})
    main_zone_xml = exml.SubElement(root_xml, 'zone', attrib={'id': 'main', 'routing': 'Full'})
    # Master zone and master host. Master host is not taken into account in energy measures
    exml.SubElement(
        exml.SubElement(
            exml.SubElement(main_zone_xml,
                            'zone', attrib={'id': 'master', 'routing': 'None'}),
            'host', attrib={'id': 'master_host', 'speed': '1Gf'}),
        'prop', attrib={'id': 'watt_per_state', 'value': '0.0:0.0'})
    return root_xml, main_zone_xml

def _root_el(root_desc: dict) -> dict:
    # Resource hierarchy starts on the Platform element
    return {
        'total_nodes': 0,
        'total_processors': 0,
        'total_cores': 0,
        # Maximum ReqTime, ReqCore, ReqMem and ReqMemBW per Job in the Platform.
        'job_limits': root_desc['job_limits'],
        # Based on 2.75 GHz 8-wide vector machine
        'reference_speed': root_desc['job_limits']['reference_machine']['clock_rate'] *\
                           root_desc['job_limits']['reference_machine']['dpflop_vector_width'],
        'clusters': []
    }

def _generate_clusters(shared_state: dict, root_desc: dict, root_el: dict,
                       main_zone_xml: XMLElement) -> None:
    for cluster_desc in root_desc['clusters']:
        if shared_state['gen_platform_xml']:
            shared_state['cluster_xml'] = _cluster_xml(shared_state, main_zone_xml)
        if shared_state['gen_res_hierarchy']:
            cluster_el = _cluster_el(root_el)
        _generate_nodes(shared_state, cluster_desc, cluster_el)
        if shared_state['gen_platform_xml']:
            # All hosts have been generated, now create the up / down routes
            _udlink_routes(shared_state)
        shared_state['counters']['cluster'] += 1

def _cluster_xml(shared_state: dict, main_zone_xml: XMLElement) -> XMLElement:
    cluster_xml = exml.SubElement(
        main_zone_xml, 'zone',
        attrib={'id': f'clu_{shared_state["counters"]["cluster"]}', 'routing': 'Full'})
    # Each cluster has a router to communicate to other clusters and the master zone
    exml.SubElement(cluster_xml, 'router',
                    attrib={'id': f'rou_{shared_state["counters"]["cluster"]}'})
    return cluster_xml

def _cluster_el(root_el: dict) -> dict:
    cluster_el = {
        'platform': root_el,
        'local_nodes': []
    }
    root_el['clusters'].append(cluster_el)
    return cluster_el

def _generate_nodes(shared_state: dict, cluster_desc: dict, cluster_el: dict) -> None:
    # Nodes
    for node_desc in cluster_desc['nodes']:
        for _ in range(node_desc['number']):
            if shared_state['gen_platform_xml']:
                _node_xml(shared_state, cluster_desc)
            if shared_state['gen_res_hierarchy']:
                node_el = _node_el(shared_state, node_desc, cluster_el)
            _generate_processors(shared_state, node_desc, node_el)
            shared_state['counters']['node'] += 1

def _node_xml(shared_state: dict, cluster_desc: dict) -> None:
    # There is no Node XML, since Batsim only understands "compute resources", in this case Cores.
    # Create the node UP/DOWN link. SPLITDUPLEX model is utilized for simulating TCP connections
    # characteristics
    udlink_attrs = {'id': f'udl_{shared_state["counters"]["node"]}',
                    'sharing_policy': 'SPLITDUPLEX'}
    # Bandwidth
    udlink_attrs.update(shared_state['types']['network'][cluster_desc['local_links']['type']])
    # Latency
    udlink_attrs.update({'latency': cluster_desc['local_links']['latency']})
    exml.SubElement(shared_state['cluster_xml'], 'link', attrib=udlink_attrs)

def _node_el(shared_state: dict, node_desc: dict, cluster_el: dict) -> dict:
    # Transform memory from GB to MB
    max_mem = shared_state['types']['node'][node_desc['type']]['memory']['capacity'] * 1000
    node_el = {
        'cluster': cluster_el,
        # Memory is tracked at Node-level
        'max_mem': max_mem,
        'current_mem': max_mem,
        'local_processors': []
    }
    cluster_el['platform']['total_nodes'] += 1
    cluster_el['local_nodes'].append(node_el)
    return node_el

def _generate_processors(shared_state: dict, node_desc: dict, node_el: dict) -> None:
    # Processors
    for proc_desc in shared_state['types']['node'][node_desc['type']]['processors']:
        # Computational capability per Core in FLOPs
        flops_per_core = shared_state['types']['processor'][proc_desc['type']]['clock_rate'] *\
                         shared_state['types']['processor'][proc_desc['type']]['dpflops_per_cycle']
        # Power consumption per Core in Watts
        power_per_core = shared_state['types']['processor'][proc_desc['type']]['power'] /\
                         shared_state['types']['processor'][proc_desc['type']]['cores']
        if shared_state['gen_platform_xml']:
            flops_per_core_xml, power_per_core_xml = _proc_xml(flops_per_core, power_per_core)
        for _ in range(proc_desc['number']):
            if shared_state['gen_res_hierarchy']:
                proc_el = _proc_el(shared_state, proc_desc, node_el, flops_per_core, power_per_core)
            _generate_cores(shared_state, flops_per_core_xml, power_per_core_xml, proc_desc,
                            proc_el)

def _proc_xml(flops_per_core: float, power_per_core: float) -> tuple:
    # For each processor several P-states are defined based on the utilization
    # P0 - 100% FLOPS - 100% Power / core - Job scheduled on the core
    # P1 - 75% FLOPS - 100% Power / core - Job scheduled on the core but constraint by memory BW
    # P2 - 0% FLOPS - 15% Power / core - Job not scheduled but other job in same processor cores
    # P3 - 0% FLOPS - 5% Power / core - Processor idle
    # These are further associated to each individual core
    flops_per_core_xml = {'speed': (f'{flops_per_core:.3f}Gf, {0.75 * flops_per_core:.3f}Gf, '
                                    f'{0.001:.3f}f, {0.001:.3f}f')}
    power_per_core_xml = (f'{power_per_core:.3f}:{power_per_core:.3f}, '
                          f'{power_per_core:.3f}:{power_per_core:.3f}, '
                          f'{0.15 * power_per_core:.3f}:{0.15 * power_per_core:.3f}, '
                          f'{0.05 * power_per_core:.3f}:{0.05 * power_per_core:.3f}')
    return flops_per_core_xml, power_per_core_xml

def _proc_el(shared_state: dict, proc_desc: dict, node_el: dict, flops_per_core: float,
             power_per_core: float) -> dict:
    max_mem_bw = shared_state['types']['processor'][proc_desc['type']]['mem_bw']
    proc_el = {
        'node': node_el,
        # Memory bandwidth is tracked at Processor-level
        'max_mem_bw': max_mem_bw,
        'current_mem_bw': max_mem_bw,
        'flops_per_core': flops_per_core,
        'power_per_core': power_per_core,
        'local_cores': []
    }
    node_el['cluster']['platform']['total_processors'] += 1
    node_el['local_processors'].append(proc_el)
    return proc_el

def _generate_cores(shared_state: dict, flops_per_core_xml: dict, power_per_core_xml: str,
                    proc_desc: dict, proc_el: dict) -> None:
    # Cores
    for _ in range(shared_state['types']['processor'][proc_desc['type']]['cores']):
        if shared_state['gen_platform_xml']:
            _core_xml(shared_state, flops_per_core_xml, power_per_core_xml)
        if shared_state['gen_res_hierarchy']:
            _core_el(shared_state, proc_el)
        shared_state['counters']['core'] += 1

def _core_xml(shared_state: dict, flops_per_core_xml: dict, power_per_core_xml: str) -> None:
    # Create the Core XML element and associate power properties
    core_attrs = {'id': f'cor_{shared_state["counters"]["core"]}', 'pstate': '3'}
    core_attrs.update(flops_per_core_xml)
    exml.SubElement(exml.SubElement(shared_state['cluster_xml'], 'host', attrib=core_attrs),
                    'prop', attrib={'id': 'watt_per_state', 'value': power_per_core_xml})
    # Append the up / down route parameters for generating after all the Cores
    shared_state['udlink_routes'].append((f'cor_{shared_state["counters"]["core"]}',
                                          f'udl_{shared_state["counters"]["node"]}'))

def _core_el(shared_state: dict, proc_el: dict) -> None:
    core_el = res.Resource(proc_el, shared_state['counters']['core'])
    proc_el.node.cluster.platform.total_cores += 1
    proc_el.local_cores.append(core_el)
    shared_state['core_pool'].append(core_el)

def _udlink_routes(shared_state: dict) -> None:
    for udlink_route in shared_state['udlink_routes']:
        core_id, udlink_id = udlink_route
        udlink_route_down_xml = exml.SubElement(
            shared_state['cluster_xml'], 'route',
            attrib={'src': f'rou_{shared_state["counters"]["cluster"]}', 'dst': core_id,
                    'symmetrical': 'NO'})
        exml.SubElement(
            udlink_route_down_xml, 'link_ctn', attrib={'id': udlink_id, 'direction': 'DOWN'})
        udlink_route_up_xml = exml.SubElement(
            shared_state['cluster_xml'], 'route',
            attrib={'src': core_id, 'dst': f'rou_{shared_state["counters"]["cluster"]}',
                    'symmetrical': 'NO'})
        exml.SubElement(
            udlink_route_up_xml, 'link_ctn', attrib={'id': udlink_id, 'direction': 'UP'})
    # Clean the routes for next cluster
    shared_state['udlink_routes'] = []

def _global_links(shared_state: dict, root_desc: dict, main_zone_xml: XMLElement) -> None:
    # Create a global link for each cluster for connecting to the master host
    for cluster_n in range(len(root_desc['clusters'])):
        global_link_attrs = {'id': f'glob_lnk_{cluster_n}', 'sharing_policy': 'SPLITDUPLEX'}
        # Bandwidth
        global_link_attrs.update(
            shared_state['types']['network'][root_desc['global_links']['type']])
        # Latency
        global_link_attrs.update({'latency': root_desc['global_links']['latency']})
        exml.SubElement(main_zone_xml, 'link', attrib=global_link_attrs)

def _zone_routes(root_desc: dict, main_zone_xml: XMLElement) -> None:
    # Create the zone routes over the global links
    for cluster_n in range(len(root_desc['clusters'])):
        zone_route_down_xml = exml.SubElement(
            main_zone_xml, 'zoneRoute',
            attrib={'src': 'master', 'dst': f'clu_{cluster_n}', 'gw_src': 'master_host',
                    'gw_dst': f'rou_{cluster_n}', 'symmetrical': 'NO'})
        exml.SubElement(
            zone_route_down_xml, 'link_ctn',
            attrib={'id': f'glob_lnk_{cluster_n}', 'direction': 'DOWN'})
        zone_route_up_xml = exml.SubElement(
            main_zone_xml, 'zoneRoute',
            attrib={'src': f'clu_{cluster_n}', 'dst': 'master', 'gw_src': f'rou_{cluster_n}',
                    'gw_dst': 'master_host', 'symmetrical': 'NO'})
        exml.SubElement(
            zone_route_up_xml, 'link_ctn',
            attrib={'id': f'glob_lnk_{cluster_n}', 'direction': 'UP'})

def _write_platform_definition(root_xml: XMLElement) -> None:
    # Write the Simgrid / Batsim compliant platform to an output file
    with open('platform.xml', 'w') as out_f:
        out_f.write(mxml.parseString(
            (f'<!DOCTYPE platform SYSTEM "https://simgrid.org/simgrid.dtd">'
             f'{exml.tostring(root_xml).decode()}')).toprettyxml(indent='  ',
                                                                 encoding='utf-8').decode())

def _write_resource_hierarchy(shared_state: dict, root_el: dict) -> None:
    # Pickle the Platform hierarchy and Core pool
    with open('res_hierarchy.pkl', 'wb') as out_f:
        pickle.dump((root_el, shared_state['core_pool']), out_f)
