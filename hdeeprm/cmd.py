"""
Command line scripts for managing HDeepRM experiments.
"""

import argparse as ap
import csv
import json
import os
import os.path as path
import random as rnd
import subprocess as sp
import evalys.jobset as ejs
import evalys.visu.gantt as evg
import evalys.visu.lifecycle as evl
import evalys.visu.series as evs
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
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
    parser.add_argument('-nr', '--nbruns', type=int, default=1,
                        help='Number of simulations to be run')
    args = parser.parse_args()

    # Load the options
    print('Loading options')
    with open(args.options_file, 'r') as in_f:
        options = json.load(in_f)
        options['pybatsim']['seed'] = options['seed']
        if args.agent:
            options['pybatsim']['agent']['file'] = path.abspath(args.agent)
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
        skipped.extend(('platform', 'res_hierarchy'))

    # Generate the Workload
    if not present['workload.json'] or not present['job_limits.pkl'] or new_reference_speed:
        print('Generating workload JSON and job limits')
        generate_workload(options['workload_file_path'], options['nb_resources'],
                          options['nb_jobs'])
        print('Saved "workload.json" and "job_limits.pkl" in current directory')
    else:
        skipped.extend(('workload', 'job_limits'))

    for skip in skipped:
        print(f'Skipping {skip} generation - Using file in current directory')

    # Launch both PyBatsim and Batsim instances for running the simulation
    for _ in range(args.nbruns):
        pybs = sp.Popen(('pybatsim', '-o', json.dumps(options['pybatsim']),
                         path.join(path.dirname(path.realpath(__file__)),
                                   'entrypoints/HDeepRMWorkloadManager.py')),
                        stdout=sp.DEVNULL, stderr=sp.DEVNULL)
        sp.run(('batsim', '-E', '-w', 'workload.json', '-p', 'platform.xml'), check=True)
        pybs.communicate()

def visual() -> None:
    """Script for analysing stats from HDeepRM outcomes.

It utilizes `Evalys <https://gitlab.inria.fr/batsim/evalys>`_ for plotting useful information from
output files. Supports *queue_size*, *utilization*, *lifecycle*, *gantt* and *gantt_no_label*.

Command line arguments:
    | ``stat`` - Type of stat to be printed.
      ``save`` - (Optional) Save the plot in the provided file path.
    """

    parser = ap.ArgumentParser(description='Plots information about the simulation outcome')
    parser.add_argument('stat', type=str,
                        choices=('queue_size', 'utilization', 'lifecycle', 'gantt',
                                 'gantt_no_label'),
                        help='Statistic to visualise')
    parser.add_argument('-s', '--save', type=str, help='Save the plot in the specified file path')
    args = parser.parse_args()

    # Obtain the title for the plots given the agent configuration
    with open('./options.json', 'r') as in_f:
        options = json.load(in_f)
    agent_options = options['pybatsim']['agent']
    if agent_options['type'] == 'CLASSIC':
        title = f'{agent_options["type"]} {agent_options["policy_pair"]}'
    elif agent_options['type'] == 'LEARNING':
        title = (f'{agent_options["type"]} {agent_options["run"]} {agent_options["hidden"]} '
                    f'{agent_options["lr"]} {agent_options["gamma"]}')
    else:
        raise ValueError('Invalid agent type in "options.json"')

    # Create the Job Set
    jobset = ejs.JobSet.from_csv('./out_jobs.csv', resource_bounds=(0, options['nb_resources']))
    if args.stat == 'queue_size':
        evs.plot_series(jobset, name='queue', title=f'Queue size over time for {title}')
        plt.ylabel('Pending jobs')
    elif args.stat == 'utilization':
        evs.plot_series(jobset, name='utilization', title=f'Utilization over time for {title}')
        plt.ylabel('Active cores')
    elif args.stat == 'lifecycle':
        evl.plot_lifecycle(jobset, title=f'Job lifecycle for {title}')
    elif args.stat == 'gantt':
        evg.plot_gantt(jobset, title=f'Gantt chart for {title}')
        plt.ylabel('Cores')
    elif args.stat == 'gantt_no_label':
        evg.plot_gantt(jobset, title=f'Gantt chart for {title}', labeler=lambda _: '')
        plt.ylabel('Cores')

    plt.xlabel('Simulation time')
    if args.save:
        plt.savefig(args.save)
    plt.show()

def metrics() -> None:
    """Script for comparing metrics between simulation runs.

Plots a grid with different metrics resulting from *out_schedule.csv*.

Command line arguments:
    | ``res1`` - *out_schedule.csv* file from run 1.
      ``res2`` - *out_schedule.csv* file from run 2.
    """

    parser = ap.ArgumentParser(description='Compares metrics between simulation runs')
    parser.add_argument('res1', type=str, help='out_schedule.csv file from run 1')
    parser.add_argument('res2', type=str, help='out_schedule.csv file from run 2')
    parser.add_argument('-s', '--save', type=str, help='Save the plot in the specified file path')
    args = parser.parse_args()

    with open(args.res1, 'r') as res1_f,\
         open(args.res2, 'r') as res2_f:
        res1_reader = csv.reader(res1_f, delimiter=',')
        res2_reader = csv.reader(res2_f, delimiter=',')
        names, res1_metrics = [row for row in res1_reader]
        _, res2_metrics = [row for row in res2_reader]
    _, axes = plt.subplots(nrows=2, ncols=4, constrained_layout=True)
    for i, row in enumerate(axes):
        for j, col in enumerate(row):
            metric1 = float(res1_metrics[i * 4 + j + 1])
            metric2 = float(res2_metrics[i * 4 + j + 1])
            bars = col.bar((1, 2), (metric1, metric2), color=('green', 'blue'),
                           edgecolor='black')
            col.set_xticks((1, 2))
            col.set_xticklabels(('A1', 'A2'))
            col.set_ylim(0, max(1.0, 1.15 * max(metric1, metric2)))
            col.yaxis.set_major_formatter(mtick.ScalarFormatter())
            col.ticklabel_format(axis='y', style='sci', scilimits=(0, 3))
            for rect in bars:
                height = rect.get_height()
                col.text(rect.get_x() + rect.get_width() / 2.0, height,
                         '%d' % int(height), ha='center', va='bottom')
            col.set_title(names[i * 4 + j + 1])
    if args.save:
        plt.savefig(args.save)
    plt.show()

def clean() -> None:
    """Script for cleaning current directory outcomes.

WARNING: it also erases the "platform.xml", "workload.json", "res_hierarchy.pkl" and
"job_limits.pkl" files. Its purpose is to facilitate quick experimenting and debugging.
    """

    for entry in os.scandir():
        if entry.name != 'options.json':
            os.remove(entry.name)
