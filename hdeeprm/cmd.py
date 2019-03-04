"""
Command line scripts for managing HDeepRM experiments.
"""

import argparse as ap
import json
import os.path as path
import random as rnd
import subprocess as sp
import evalys.jobset as ejs
import evalys.visu.lifecycle as evl
import evalys.visu.series as evs
import matplotlib.pyplot as plt
import numpy as np
from hdeeprm.util import generate_workload, generate_platform

def launch() -> None:
    """Script for launching HDeepRM experiments.

It takes care of creating the Platform XML file, the Workload JSON file and the Resource Hierarcy.
It also runs both Batsim and PyBatsim.

Command line arguments:
    | ``options_file`` - Options file in JSON
    | ``agent`` (Optional) File with the learning agent definition
    | ``inmodel`` - (Optional) Path for previous model loading
    | ``outmodel`` - (Optional) Path for saving new model
    """

    parser = ap.ArgumentParser(description='Launches HDeepRM experiments')
    parser.add_argument('options_file', type=str, help='Options file defining the experiment')
    parser.add_argument('-a', '--agent', type=str, help='File with the learning agent definition')
    parser.add_argument('-im', '--inmodel', type=str, help='Path for previous model loading')
    parser.add_argument('-om', '--outmodel', type=str, help='Path for saving new model')
    args = parser.parse_args()

    # Load the options
    print('Loading options')
    with open(args.options_file, 'r') as in_f:
        options = json.load(in_f)
        options['pybatsim']['seed'] = options['seed']
        if args.agent:
            options['pybatsim']['agent']['file'] = path.abspath(args.agent_file)
        if args.inmodel:
            options['pybatsim']['agent']['input_model'] = path.abspath(args.inmodel)
        if args.outmodel:
            options['pybatsim']['agent']['output_model'] = path.abspath(args.outmodel)

    # Set the seed for random libraries
    print('Setting the random seed')
    rnd.seed(options['seed'])
    np.random.seed(options['seed'])

    # Check if Platform, Resource Hierarchy, Workload and Job Limits are already present
    files = ('platform.xml', 'res_hierarchy.pkl', 'workload.json', 'job_limits.pkl')
    skipped = []
    present = {}
    for fil in files:
        if path.isfile(f'./{fil}'):
            present[fil] = True
        else:
            present[fil] = False

    # Generate the Platform and Resource Hierarchy
    new_reference_speed = False
    if not present['platform.xml'] or not present['res_hierarchy.pkl']:
        if not present['platform.xml']:
            print('Generating platform XML')
        else:
            skipped.append('platform')
        if not present['res_hierarchy.pkl']:
            print('Generating resource hierarchy')
            new_reference_speed = True
        else:
            skipped.append('res_hierarchy')
        generate_platform(options['platform_file_path'],
                          gen_platform_xml=not present['platform.xml'],
                          gen_res_hierarchy=not present['res_hierarchy.pkl'])
        print('Saved "platform.xml" and "res_hierarchy.pkl" in current directory')
    else:
        skipped.append('platform', 'res_hierarchy')

    # Generate the Workload
    if not present['workload.json'] or not present['job_limits.pkl'] or new_reference_speed:
        print('Generating workload JSON and job limits')
        generate_workload(options['workload_file_path'], options['nb_resources'],
                          options['nb_jobs'])
        print('Saved "workload.json" and "job_limits.pkl" in current directory')
    else:
        skipped.append('workload', 'job_limits')

    for skip in skipped:
        print(f'Skipping {skip} generation - Using file in current directory')

    # Launch both PyBatsim and Batsim instances for running the simulation
    sp.Popen(('pybatsim', '-o', json.dumps(options['pybatsim']),
              path.join(
                  path.dirname(path.realpath(__file__)), 'entrypoints/HDeepRMWorkloadManager.py')
             ), stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    sp.run(('batsim', '-E', '-w', 'workload.json', '-p', 'platform.xml'), check=True)

def stat() -> None:
    """Script for analysing stats from HDeepRM outcomes.

It utilizes `Evalys <https://gitlab.inria.fr/batsim/evalys>`_ for plotting useful information from
output files. Supports *queue_size*, *utilization* and *lifecycle*.

Command line arguments:
    | ``stat`` - Type of stat to be printed.
    """

    parser = ap.ArgumentParser(description='Plots information about the simulation outcome')
    parser.add_argument('stat', type=str, choices=('queue_size', 'utilization', 'lifecycle'),
                        help='Statistic to visualise')
    args = parser.parse_args()

    jobset = ejs.JobSet.from_csv('./out_jobs.csv')
    if args.stat == 'queue_size':
        evs.plot_series(jobset, name='queue', title='Queue size over time')
    elif args.stat == 'utilization':
        evs.plot_series(jobset, name='utilization', title='Utilization over time')
    elif args.stat == 'lifecycle':
        evl.plot_lifecycle(jobset)
    plt.show()
