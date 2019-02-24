"""
Command line scripts for managing HDeepRM experiments.
"""

import argparse as ap
import json
import os.path as path
import random as rnd
import subprocess as sp
import numpy as np
from hdeeprm.util import generate_workload, generate_platform

def launch():
    """
Script for launching HDeepRM experiments. It takes care of creating the Platform, the Workload and
running both Batsim and PyBatsim.

Cmd Args:
    @1 Options file
    """

    parser = ap.ArgumentParser(description='Launches HDeepRM experiments')
    parser.add_argument('options_file_path', type=str, help='The options file path in the system')
    args = parser.parse_args()

    # Load the options
    print('Loading options')
    with open(args.options_file_path, 'r') as in_f:
        options = json.load(in_f)

    # Set the seed for random libraries
    print('Setting the random seed')
    rnd.seed(options['seed'])
    np.random.seed(options['seed'])

    # Check if Workload, Platform and Resource Hierarchy are already present
    files = ('workload.json', 'platform.xml', 'res_hierarchy.pkl')
    present = {}
    for fil in files:
        ele = path.splitext(fil)[0]
        if path.isfile(f'./{fil}'):
            print(f'Skipping {ele} generation - Using file in current directory')
            present[ele] = True
        else:
            present[ele] = False

    # Generate the Workload
    if not present['workload']:
        print('Generating workload JSON')
        generate_workload(options['workload_file_path'], options['nb_resources'],
                          options['nb_jobs'])
        print('Saved "workload.json" in current directory')

    # Generate the Platform and Resource Hierarchy
    if not present['platform'] or not present['res_hierarchy']:
        if not present['platform']:
            print('Generating platform XML')
        if not present['res_hierarchy']:
            print('Generating resource hierarchy')
        generate_platform(options['platform_file_path'], gen_platform_xml=not present['platform'],
                          gen_res_hierarchy=not present['res_hierarchy'])
        print('Saved "platform.xml" and "res_hierarchy.pkl" in current directory')

    # Launch both PyBatsim and Batsim instances for running the simulation
    sp.Popen(('pybatsim', '-o', json.dumps(options['pybatsim']),
              path.join(
                  path.dirname(path.realpath(__file__)), 'entrypoints/HDeepRMWorkloadManager.py')
             ), stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    sp.run(('batsim', '-E', '-w', 'workload.json', '-p', 'platform.xml'), check=True)
