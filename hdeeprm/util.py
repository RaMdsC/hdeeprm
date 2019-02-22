"""
Provides internal utility functions for HDeepRM.
"""

import json
import os
import pickle as pkl
from defusedxml.lxml import _etree as xml
import defusedxml.minidom as xmlm
import numpy as np
from hdeeprm.resource import Platform, Cluster, Node, Processor, Core

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')

def generate_workload(workload_file_path, nb_resources, nb_jobs):
    """
    Takes a SWF formatted Workload file and parses it into
    a Batsim-ready JSON file.
    """

    with open(workload_file_path, 'r') as in_f:
        # Workload and data fields
        workload = {}
        workload['nb_res'] = nb_resources
        workload['jobs'] = []
        workload['profiles'] = {}

        # Read each Job from the SWF file and parse the fields
        min_submit_time = None
        job_id = 0
        for line in in_f:
            # Stop if the number of jobs has been reached
            if job_id == nb_jobs:
                break
            # Skip comments
            if line.startswith(';'):
                continue
            job_info = tuple(map(lambda par: int(float(par)), line.split()))
            # Skip if submit_time, req_resources, req_time_per_res and both
            # memory parameters are not specified.
            # In SWF, this is indicated by - 1.
            submit_time = job_info[1]
            used_mem_per_res = job_info[6]
            req_resources = job_info[7]
            req_time_per_res = job_info[8]
            req_mem_per_res = job_info[9]
            if any(map(lambda parameter: parameter < 0,
                       (submit_time, req_resources, req_time_per_res))):
                continue
            if req_mem_per_res < 0:
                if used_mem_per_res < 0:
                    continue
                else:
                    req_mem_per_res = used_mem_per_res
            # Need to shift the initial submission time by the minimum
            # since the trace does not start at 0
            if not min_submit_time:
                min_submit_time = submit_time
            submit_time -= min_submit_time
            # User estimates in Job requested time have been shown to be
            # generally overestimated. The distribution here used is taken
            # as an approximation to that shown in [Tsafrir et al. 2007]
            real_to_estimate = np.arange(0.05, 1.3, 0.05)
            probabilities = np.array([0.15, 0.09, 0.07] + 5 * [0.04]\
                                     + 5 * [0.03] + 10 * [0.02] + [0.06, 0.08])
            real_time_per_res = np.random.choice(real_to_estimate,
                                                 p=probabilities)\
                                * req_time_per_res
            # Calculate FLOPs per resource. The original trace provides time
            # per resource, here it is normalized to FLOPs with respect to a
            # 2.75 GHz 8-wide vector CPU (reference machine, 22e9 ops/s)
            req_ops_per_res = int(22e9 * req_time_per_res)
            real_ops_per_res = int(22e9 * real_time_per_res)
            # Original trace provides memory in KB, convert it to MB
            # for framework compatibility
            req_mem_per_res = int(req_mem_per_res / 1000)
            # Calculate the sustained memory bandwidth requirement
            # per resource. No info on original trace, this is synthetically
            # produced from a random uniform distribution with values
            # (4, 8, 12, 16, 20, 24).
            req_mem_bw_per_res = int(np.random.choice(np.arange(4, 25, 4)))
            # Profile name based on parameters. This may be shared between
            # multiple Jobs if they share their requirements.
            profile_name = (f'{req_time_per_res}_{req_mem_per_res}_'
                            f'{req_mem_bw_per_res}')
            # Profile and data fields
            profile = {}
            profile['type'] = 'parallel_homogeneous'
            profile['com'] = 0
            profile['cpu'] = real_ops_per_res
            profile['req_time'] = req_time_per_res
            profile['req_ops'] = req_ops_per_res
            profile['mem'] = req_mem_per_res
            profile['mem_bw'] = req_mem_bw_per_res
            workload['profiles'].setdefault(profile_name, profile)
            # Job and data fields
            job = {}
            job['id'] = job_id
            job['subtime'] = submit_time
            job['res'] = req_resources
            job['profile'] = profile_name
            workload['jobs'].append(job)

            job_id += 1

    # Write the data structure into the JSON output
    with open('workload.json', 'w') as out_f:
        json.dump(workload, out_f)

def generate_platform(platform_file_path, definition=True, hierarchy=False):
    """
    Takes HDeepRM JSON formatted platform definition and outputs
    both a Batsim-ready XML file and the Platform object hierarchy
    for the Decision System to understand relations.
    """

    # Load data from HDeepRM to parse the platform description
    with open(platform_file_path, 'r') as in_f,\
         open(os.path.join(DATA_PATH, 'network_types.json'), 'r') as net_f,\
         open(os.path.join(DATA_PATH, 'node_types.json'), 'r') as nod_f,\
         open(os.path.join(DATA_PATH, 'processor_types.json'), 'r') as pro_f:
        platform_description = json.load(in_f)
        network_types = json.load(net_f)
        node_types = json.load(nod_f)
        processor_types = json.load(pro_f)
    if definition:
        # Root platform element
        platform_xml = xml.Element('platform', attrib={'version': '4.1'})
        # Main zone (contains master zone and all clusters)
        main_zone_xml = xml.SubElement(platform_xml, 'zone', attrib=
                                       {'id': 'main', 'routing': 'Full'})
        # Master zone (workload management algorithms run inside this host)
        master_zone_xml = xml.SubElement(main_zone_xml, 'zone', attrib=
                                         {'id': 'master', 'routing': 'None'})
        master_host_xml = xml.SubElement(master_zone_xml, 'host', attrib=
                                         {'id': 'master_host', 'speed': '1Gf'})
        # Master host not taken into account in energy measures
        xml.SubElement(master_host_xml, 'prop', attrib=
                       {'id': 'watt_per_state', 'value': '0.0:0.0'})
    if hierarchy:
        # Resource hierarchy starts on the Platform element
        platform_el = Platform(platform_description['job_limits'])
        # Core pool for filtering and selecting Cores
        core_pool = []
    cluster_n = 0
    node_n = 0
    core_n = 0
    # Clusters
    for cluster in platform_description['clusters']:
        if definition:
            cluster_id = f'clu_{cluster_n}'
            cluster_xml = xml.SubElement(main_zone_xml, 'zone', attrib=
                                         {'id': cluster_id, 'routing': 'Full'})
            # Each cluster has a router to communicate to other clusters
            # and the master zone
            router_id = f'rou_{cluster_n}'
            xml.SubElement(cluster_xml, 'router', attrib={'id': router_id})
            # Need to temporarly store information about up / down routes,
            # since the XML DTD spec imposes setting them at the end of
            # generating all nodes
            udlink_routes = []
        if hierarchy:
            cluster_el = Cluster(platform_el)
            platform_el.local_clusters.append(cluster_el)
        # Nodes
        for node in cluster['nodes']:
            node_template = node_types[node['type']]
            for _ in range(node['number']):
                if definition:
                    # There is no Node or Processor XML, since Batsim only
                    # understands "compute resources", in this case Cores
                    # Up / down link
                    # SPLITDUPLEX model is utilized for simulating TCP
                    # connections characteristics
                    udlink_id = f'udl_{node_n}'
                    udlink_attrs = {'id': udlink_id,
                                    'sharing_policy': 'SPLITDUPLEX'}
                    # Bandwidth
                    udlink_attrs.update(
                        network_types[cluster['local_links']['type']])
                    # Latency
                    udlink_attrs.update(
                        {'latency': cluster['local_links']['latency']})
                    xml.SubElement(cluster_xml, 'link', attrib=udlink_attrs)
                if hierarchy:
                    # Transform memory from GB to MB
                    node_el = Node(cluster_el,
                                   node_template['memory']['capacity'] * 1000)
                    platform_el.total_nodes += 1
                    cluster_el.local_nodes.append(node_el)
                # Processors
                for proc in node_template['processors']:
                    proc_template = processor_types[proc['type']]
                    # For each processor several P-states are defined based
                    # on the utilization
                    # P0 - 100% FLOPS - 100% Power / core
                    #   Job scheduled on the core
                    # P1 - 75% FLOPS - 100% Power / core
                    #   Job scheduled on the core but constraint by memory BW
                    # P2 - 0% FLOPS - 15% Power / core
                    #   Job not scheduled but some other job in same
                    #   processor cores
                    # P3 - 0% FLOPS - 5% Power / core
                    #   Processor idle
                    #
                    # These are further associated to each individual core
                    flops_per_core = proc_template['clock_rate'] *\
                                     proc_template['dpflops_per_cycle']
                    power_per_core = proc_template['power'] /\
                                     proc_template['cores']
                    if definition:
                        core_speed = {
                            'speed': (f'{flops_per_core:.3f}Gf, '
                                      f'{0.75 * flops_per_core:.3f}Gf, '
                                      f'{0.001:.3f}f, '
                                      f'{0.001:.3f}f')
                        }
                        core_power = (
                            f'{power_per_core:.3f}:{power_per_core:.3f}, '
                            f'{power_per_core:.3f}:{power_per_core:.3f}, '
                            f'{0.15 * power_per_core:.3f}:'
                            f'{0.15 * power_per_core:.3f}, '
                            f'{0.05 * power_per_core:.3f}:'
                            f'{0.05 * power_per_core:.3f}'
                        )
                    for _ in range(proc['number']):
                        if hierarchy:
                            proc_el = Processor(node_el,
                                                proc_template['mem_bw'],
                                                flops_per_core,
                                                power_per_core)
                            platform_el.total_processors += 1
                            node_el.local_processors.append(proc_el)
                        # Cores
                        for _ in range(proc_template['cores']):
                            if definition:
                                core_id = f'cor_{core_n}'
                                core_attrs = {'id': core_id, 'pstate': '3'}
                                core_attrs.update(core_speed)
                                core_xml = xml.SubElement(cluster_xml, 'host',
                                                          attrib=core_attrs)
                                # Associate power properties
                                xml.SubElement(core_xml, 'prop', attrib=
                                               {'id': 'watt_per_state',
                                                'value': core_power})
                                # Append the up / down route parameters for
                                # generating after all cores
                                udlink_routes.append((core_id, udlink_id))
                            if hierarchy:
                                core_el = Core(proc_el, core_n)
                                platform_el.total_cores += 1
                                core_pool.append(core_el)
                                proc_el.local_cores.append(core_el)
                            core_n += 1
                node_n += 1
        cluster_n += 1
        if definition:
            # All hosts have been generated, now create the up / down routes
            for udlink_route in udlink_routes:
                core_id, udlink_id = udlink_route
                udlink_route_down_xml = xml.SubElement(cluster_xml, 'route',
                                                       attrib=
                                                       {'src': router_id,
                                                        'dst': core_id,
                                                        'symmetrical': 'NO'})
                xml.SubElement(udlink_route_down_xml, 'link_ctn',
                               attrib={'id': udlink_id, 'direction': 'DOWN'})
                udlink_route_up_xml = xml.SubElement(cluster_xml, 'route',
                                                     attrib=
                                                     {'src': core_id,
                                                      'dst': router_id,
                                                      'symmetrical': 'NO'})
                xml.SubElement(udlink_route_up_xml, 'link_ctn',
                               attrib={'id': udlink_id, 'direction': 'UP'})
    if definition:
        # Create a global link for each cluster for connecting
        # to the master host
        for cluster_n in range(len(platform_description['clusters'])):
            global_link_attrs = {'id': f'glob_lnk_{cluster_n}',
                                 'sharing_policy': 'SPLITDUPLEX'}
            # Bandwidth
            global_link_attrs.update(
                network_types[platform_description['global_links']['type']])
            # Latency
            global_link_attrs.update(
                {'latency': platform_description['global_links']['latency']})
            xml.SubElement(main_zone_xml, 'link', attrib=global_link_attrs)
        # Create the zone routes over the global links
        for cluster_n in range(len(platform_description['clusters'])):
            cluster_id = f'clu_{cluster_n}'
            router_id = f'rou_{cluster_n}'
            zone_route_down_xml = xml.SubElement(main_zone_xml, 'zoneRoute',
                                                 attrib={
                                                     'src': 'master',
                                                     'dst': cluster_id,
                                                     'gw_src': 'master_host',
                                                     'gw_dst': router_id,
                                                     'symmetrical': 'NO'})
            xml.SubElement(zone_route_down_xml, 'link_ctn',
                           attrib={'id': f'glob_lnk_{cluster_n}',
                                   'direction': 'DOWN'})
            zone_route_up_xml = xml.SubElement(main_zone_xml, 'zoneRoute',
                                               attrib={
                                                   'src': cluster_id,
                                                   'dst': 'master',
                                                   'gw_src': router_id,
                                                   'gw_dst': 'master_host',
                                                   'symmetrical': 'NO'})
            xml.SubElement(zone_route_up_xml, 'link_ctn',
                           attrib={'id': f'glob_lnk_{cluster_n}',
                                   'direction': 'UP'})
        # Write the Simgrid / Batsim compliant platform to an output file
        with open('platform.xml', 'w') as out_f:
            out_f.write(
                xmlm.parseString((f'<!DOCTYPE platform SYSTEM '
                                  f'"https://simgrid.org/simgrid.dtd">'
                                  f'{xml.tostring(platform_xml).decode()}'))\
                                 .toprettyxml(indent='  ',
                                              encoding='utf-8').decode())
    if hierarchy:
        # Pickle the Platform hierarchy and Core pool
        with open('res_hierarchy.pkl', 'wb') as out_f:
            pkl.dump((platform_el, core_pool), out_f)
