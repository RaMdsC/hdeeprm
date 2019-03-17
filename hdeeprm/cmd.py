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
import evalys.jobset as ej
import evalys.utils as eu
import evalys.visu.core as evc
import evalys.visu.gantt as evg
import evalys.visu.legacy as evleg
import evalys.visu.lifecycle as evl
import evalys.visu.series as evs
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from hdeeprm.util import generate_workload, generate_platform

def launch() -> None:
    """Utility for launching HDeepRM experiments.

It takes care of creating the Platform XML file, the Workload JSON file and the Resource Hierarcy.
It also runs both Batsim and PyBatsim.

Command line arguments:
    | ``options_file`` - Options file in JSON.
    | ``customworkload`` - (Optional) Path to the custom workload in case one is used.
    | ``agent`` (Optional) File with the learning agent definition.
    | ``inmodel`` - (Optional) Path for previous model loading.
    | ``outmodel`` - (Optional) Path for saving new model.
    | ``nbruns`` - (Optional) Number of train runs for the learning agent.
    """

    parser = ap.ArgumentParser(description='Launches HDeepRM experiments')
    parser.add_argument('options_file', type=str, help='Options file defining the experiment')
    parser.add_argument('-cw', '--customworkload', type=str, help='Path for the custom workload')
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

    # If a custom workload is provided, just calculate the operations with respect to the reference
    # speed
    if args.customworkload:
        print(f'Utilizing {args.customworkload} as workload, adjusting operations')
        generate_workload(options['workload_file_path'], options['nb_resources'],
                          options['nb_jobs'], custom_workload_path=args.customworkload)
        print('Saved "workload.json" and "job_limits.pkl" in current directory')
    # Generate the Workload
    elif not present['workload.json'] or not present['job_limits.pkl'] or new_reference_speed:
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
        with open('pybatsim.log', 'w+') as out_f:
            pybs = sp.Popen(('pybatsim', '-o', json.dumps(options['pybatsim']),
                             path.join(path.dirname(path.realpath(__file__)),
                                       'entrypoints/HDeepRMWorkloadManager.py')),
                            stdout=out_f, stderr=out_f)
            sp.run(('batsim', '-E', '-w', 'workload.json', '-p', 'platform.xml'), check=True)
            pybs.communicate()

def visual() -> None:
    """Utility for analysing stats from HDeepRM outcomes.

It utilizes `Evalys <https://gitlab.inria.fr/batsim/evalys>`_ for plotting useful information from
output files. Supports *queue_size*, *utilization*, *lifecycle*, *gantt*, *gantt_no_label*,
*core_bubbles*, *mem_bubbles* and *mem_bw_overutilization*.

Command line arguments:
    | ``visualization`` - Type of visualization to be printed.
    | ``save`` - (Optional) Save the plot in the provided file path.
    """

    parser = ap.ArgumentParser(description='Plots information about the simulation outcome')
    parser.add_argument('visualization', type=str,
                        choices=('queue_size', 'utilization', 'lifecycle', 'gantt',
                                 'gantt_no_label', 'core_bubbles', 'mem_bubbles',
                                 'mem_bw_overutilization'),
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

    
    # Over-utilization visualizations
    if args.visualization in ('core_bubbles', 'mem_bubbles', 'mem_bw_overutilization'):
        with open('overutilizations.json', 'r') as in_f:
            overutilizations = json.load(in_f)
        with open('out_schedule.csv', 'r') as in_f:
            _, values = [row for row in csv.reader(in_f, delimiter=',')]
            makespan = float(values[2])
    # Job visualizations
    else:
        jobset = ej.JobSet.from_csv('./out_jobs.csv', resource_bounds=(0, options['nb_resources']))

    if args.visualization == 'core_bubbles':
        fig, ax = plt.subplots()
        ax.set_title(f'Core bubbles for {title}')
        ax.set_xlim(0, makespan)
        ax.set_ylim(0, len(overutilizations['core']))
        ax.plot(overutilizations['core'], range(len(overutilizations['core'])))
    elif args.visualization == 'mem_bubbles':
        fig, ax = plt.subplots()
        ax.set_title(f'Memory bubbles for {title}')
        ax.set_xlim(0, makespan)
        ax.set_ylim(0, len(overutilizations['mem']))
        ax.plot(overutilizations['mem'], range(len(overutilizations['mem'])))
    elif args.visualization == 'mem_bw_overutilization':
        fig, ax = plt.subplots()
        ax.set_title(f'Mem BW overutilization spans for {title}')
        ax.set_xlim(0, makespan)
        ax.set_xlabel('Simulation time')
        ax.set_ylim(0, overutilizations['mem_bw']['nb_procs'])
        ax.set_ylabel('Processors')
        ax.grid(True)
        for proc_id in overutilizations['mem_bw']['procs']:
            ax.broken_barh(overutilizations['mem_bw']['procs'][proc_id]['values'],
                          (int(proc_id), int(proc_id)))
    elif args.visualization == 'queue_size':
        _fixed_plot_series(jobset, name='queue', title=f'Queue size over time for {title}',
                           legend_label='Queue size')
        plt.xlabel('Simulation time')
        plt.ylabel('Pending jobs')
    elif args.visualization == 'utilization':
        _fixed_plot_series(jobset, name='utilization', title=f'Utilization over time for {title}',
                           legend_label='Load')
        plt.xlabel('Simulation time')
        plt.ylabel('Active cores')
    elif args.visualization == 'lifecycle':
        evl.plot_lifecycle(jobset, title=f'Job lifecycle for {title}')
    elif args.visualization == 'gantt':
        evg.plot_gantt(jobset, title=f'Gantt chart for {title}')
        plt.xlabel('Simulation time')
    elif args.visualization == 'gantt_no_label':
        evg.plot_gantt(jobset, title=f'Gantt chart for {title}', labeler=lambda _: '')
        plt.xlabel('Simulation time')
    if args.save:
        plt.savefig(args.save, format='svg', dpi=1200)
    plt.show()

def metrics() -> None:
    """Utility for comparing metrics between simulation runs.

Plots a grid with different metrics resulting from *out_schedule.csv*.

Command line arguments:
    | ``res1`` - *out_schedule.csv* file from run 1.
    | ``res2`` - *out_schedule.csv* file from run 2.
    """

    parser = ap.ArgumentParser(description='Compares metrics between simulation runs')
    parser.add_argument('res1', type=str, help='out_schedule.csv file from run 1')
    parser.add_argument('res2', type=str, help='out_schedule.csv file from run 2')
    parser.add_argument('-n1', '--name1', type=str, default='A1',
                        help='Name of the first policy or agent')
    parser.add_argument('-n2', '--name2', type=str, default='A2',
                        help='Name of the second policy or agent')
    parser.add_argument('-s', '--save', type=str, help='Save the plot in the specified file path')
    args = parser.parse_args()

    with open(args.res1, 'r') as res1_f,\
         open(args.res2, 'r') as res2_f:
        res1_reader = csv.reader(res1_f, delimiter=',')
        res2_reader = csv.reader(res2_f, delimiter=',')
        names = ('Energy (joules)', 'Makespan', 'Max. slowdown', 'Max. turnaround time',
                 'Max. waiting time', 'Mean slowdown', 'Mean turnaround time', 'Mean waiting time')
        _, res1_metrics = [row for row in res1_reader]
        _, res2_metrics = [row for row in res2_reader]
    _, axes = plt.subplots(nrows=2, ncols=4, constrained_layout=True, figsize=(12, 4))
    for i, row in enumerate(axes):
        for j, col in enumerate(row):
            metric1 = float(res1_metrics[i * 4 + j + 1])
            metric2 = float(res2_metrics[i * 4 + j + 1])
            x = (1, 2)
            bars = col.bar(x, (metric1, metric2), color=('green', 'blue'),
                           edgecolor='black', width=0.6)
            col.set_xlim(0, 3)
            col.set_ylim(0, max(1.0, 1.15 * max(metric1, metric2)))
            col.set_xticks(x)
            col.set_xticklabels((args.name1, args.name2))
            col.yaxis.set_major_formatter(mtick.ScalarFormatter())
            col.ticklabel_format(axis='y', style='sci', scilimits=(0, 3))
            for rect in bars:
                height = rect.get_height()
                if height >= 1e7:
                    text = f'{height:.2e}'
                else:
                    text = f'{int(height)}'
                col.text(rect.get_x() + rect.get_width() / 2.0, height,
                         text, ha='center', va='bottom')
            col.set_title(names[i * 4 + j])
    if args.save:
        plt.savefig(args.save, format='svg', dpi=1200)
    plt.show()

def jobstat() -> None:
    """Utility for obtaining statistics about jobs in the workload trace.

It can calculate several statistics by parsing the *workload.json* file.

Command line arguments:
    | ``workload`` - *workload.json* with the trace.
    | ``datafield`` - Job data field from which to obtain information. One of *req_time*, *size*,
      *mem* or *mem_bw*.
    | ``statistic`` - Statistic to be calculated. One of *min*, *max*, *mean*, *median*, *p95* or
      *p99*.
    """

    parser = ap.ArgumentParser(description='Compares metrics between simulation runs')
    parser.add_argument('workload', type=str, help='JSON workload file with the trace')
    parser.add_argument('datafield', type=str, choices=('req_time', 'size', 'mem', 'mem_bw'),
                        help='Job data field from which to obtain information')
    parser.add_argument('statistic', type=str,
                        choices=('min', 'max', 'mean', 'median', 'p95', 'p99'),
                        help='Statistic to be calculated')
    args = parser.parse_args()

    with open(args.workload, 'r') as in_f:
        workload = json.load(in_f)
    if args.datafield == 'size':
        data = np.array(
            [job['res'] for job in workload['jobs']]
        )
    else:
        data = np.array(
            [workload['profiles'][job['profile']][args.datafield] for job in workload['jobs']]
        )
    if args.statistic == 'min':
        print(np.min(data))
    elif args.statistic == 'max':
        print(np.max(data))
    elif args.statistic == 'mean':
        print(np.mean(data))
    elif args.statistic == 'median':
        print(np.median(data))
    elif args.statistic == 'p95':
        print(np.percentile(data, 95))
    elif args.statistic == 'p99':
        print(np.percentile(data, 99))

def clean() -> None:
    """Utility for cleaning current directory outcomes.

WARNING: it also erases the "platform.xml", "workload.json", "res_hierarchy.pkl" and
"job_limits.pkl" files, use ``-s`` or ``--soft`` to avoid this behaviour. Its purpose is to
facilitate quick experimenting and debugging.

Command line arguments:
    | ``soft`` - (Optional) Cleans only logs, does not remove platform or workload definitions.
    """

    parser = ap.ArgumentParser(description='Compares metrics between simulation runs')
    parser.add_argument('-s', '--soft', action='store_true', help='Only clean logs')
    args = parser.parse_args()

    skipped = ['options.json']
    if args.soft:
        skipped.extend(['platform.xml', 'workload.json', 'res_hierarchy.pkl', 'job_limits.pkl'])
    for entry in os.scandir():
        if entry.name not in skipped:
            os.remove(entry.name)

def _fixed_plot_series(jobset, *, name, title='Time series plot', legend_label, **kwargs):
    """
    
Should be fixed in next Evalys version, see this
`commit <https://github.com/oar-team/evalys/commit/b1d1854ec7b4085a4a6ad5d2ffbe350a44ea5edc>`_.

    """
    layout = evc.SimpleLayout(wtitle=title)
    plot = layout.inject(evs.SeriesVisualization.factory(name), spskey='all', title=title)
    eu.bulksetattr(plot, **kwargs)
    evleg.plot_load(
        load=getattr(jobset, plot._metric),
        nb_resources=jobset.MaxProcs,
        ax=plot._ax,
        time_scale=(plot.xscale == 'time'),
        load_label=legend_label
    )
    plot._ax.set_title(title)
    layout.show()
