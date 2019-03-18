HDeepRM
=======

Framework for evaluating *workload management* policies based on
*deep reinforcement learning* for *heterogeneous* clusters.

.. include-overview-start

Overview
--------

*HDeepRM* is a Python framework for evaluating workload management policies
based on deep reinforcement learning for heterogeneous clusters. It
leverages the `Batsim ecosystem <https://gitlab.inria.fr/batsim>`_
for simulating a heterogeneous workload management context. This is composed
of the *simulator*, `Batsim <https://gitlab.inria.fr/batsim/batsim>`_ and the
*decision system*, `PyBatsim <https://gitlab.inria.fr/batsim/pybatsim>`_.

HDeepRM provides a heterogeneity layer on top of PyBatsim, which adds support
for user-defined resource hierarchies. Memory capacity and bandwidth conflicts
are added along with interdependence when consolidating or scattering jobs across
the data centre.

It offers a flexible API for developing deep reinforcement learning agents.
These may be trained by providing real workload traces in
`SWF format <http://www.cs.huji.ac.il/labs/parallel/workload/swf.html>`_ along
with platforms defined in the format specified in `Platforms <TODO>`_. They can
be further evaluated and tested against classic policies.

Installation Prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~~~

HDeepRM is distributed as a Python package on
`PyPi <https://pypi.org/project/hdeeprm/>`_.
In order to download and install it, the following software is needed:

- Python3.6+, find your OS in this
  `installation guide <https://realpython.com/installing-python/>`_.
- Pip, the Python package manager. If not already available with the Python
  installation, follow the
  `official guide <https://pip.pypa.io/en/stable/installing/>`_.

Installation
~~~~~~~~~~~~

For installing HDeepRM, just download the package from PyPi:

.. code-block:: bash

  pip install --upgrade --user hdeeprm

If ``pip`` is mapped to Python 2.x, try:

.. code-block:: bash

  pip3 install --upgrade --user hdeeprm

When working with multiple Python versions, use:

.. code-block:: bash

  python3.6 -m pip install --upgrade --user hdeeprm

This should download the ``hdeeprm`` package with all its dependencies,
which are:

- ``defusedxml`` >= 0.5.0: secure XML generation and parsing.
- ``gym`` >= 0.12.0: environment, actions and observations definitions.
- ``lxml`` >= 4.3.2: generation of the XML tree. Backend for ``defusedxml``.
- ``numpy`` >= 1.16.2: efficient data structure operations.
- ``procset`` >= 1.0: closed-interval sets for resource selection.
- ``pybatsim`` >= 3.1.0: decision system and main interface to interact
  with Batsim.
- ``torch`` >= 1.0.1.post2: deep learning library for agent definition.

Usage Prerequisites
~~~~~~~~~~~~~~~~~~~

The simulation side is done by Batsim, which is needed in order to run
HDeepRM experiments. Follow the `official installation docs
<https://batsim.readthedocs.io/en/latest/installation.html>`_ for instructions.

Launching experiments
~~~~~~~~~~~~~~~~~~~~~

In order to experiment with HDeepRM, an integrated launcher is provided:

.. code-block:: bash

  hdeeprm-launch -a <agent.py> -cw <custom_workload.json> -im <saved_model.pt> -om <to_save_model.pt> <options.json>

The ``options.json`` specifies the experiment parameters. The JSON structure
is as follows:

.. code-block:: json

  {
    "seed": 0,
    "nb_resources": 0,
    "nb_jobs": 0,
    "workload_file_path": "",
    "platform_file_path": "",
    "pybatsim": {
      "log_level": "",
      "env": {
        "objective": "",
        "actions": {
          "selection": [
            {"": []}
          ],
          "void": false
        },
        "observation": "",
        "queue_sensitivity": 0.0,
      },
      "agent": {
        "type": "",
        "run": "",
        "hidden": 0,
        "lr": 0.0,
        "gamma": 0.0
      }
    }
  }

Global options:

* ``seed`` - The random seed for evaluation reproducibility.
* ``nb_resources`` - Total number of cores in the simulated platform.
* ``nb_jobs`` - Total number of jobs to generate in the workload.
* ``workload_file_path`` - Location of the original SWF formatted workload.
* ``platform_file_path`` - Location of the original
  HDeepRM JSON formatted platform.

PyBatsim options:

* ``log_level`` - Logging level for showing insights from the simulation. See `Logging <https://docs.python.org/3.6/howto/logging.html>`_ for reference on possible values.

PyBatsim - Environment options:

* ``objective`` - Metric to be optimised by the agent. See `Objectives <TODO>`_ for an explanation and recognised values.
* ``actions`` - Subset of actions for the simulation. If not specified, all 37 actions in HDeepRM are used.
* ``observation`` - Type of observation to use, one of *normal*, *small* or *minimal*.
* ``queue_sensitivity`` - Sensitivity of the observation to variations in job queue size. See `Hyperparameters - Queue Sensitivity <TODO>`_.

PyBatsim - Common agent options:

* ``type`` - Type of the scheduling agent, one of *CLASSIC* or *LEARNING*.

PyBatsim - `Learning <https://hdeeprm.readthedocs.io/en/latest/source/packages/hdeeprm.agent.html#hdeeprm.agent.LearningAgent>`_ agent options:

* ``run`` - Type of run for the learning agent, one of *train* or *test*.
  When training, the agent's inner model is updated,
  whereas testing is meant for evaluation purposes.
* ``hidden`` - Number of units in each hidden layer from the agent's inner model. See `Hyperparameters - Hidden units <TODO>`_.
* ``lr`` - Learning rate for updating the agent's inner model. See `Hyperparameters - Learning rate <TODO>`_.
* ``gamma`` - Discount factor for rewards. See `Hyperparameters - Reward Discount Factor <TODO>`_.

This is an example of an ``options.json`` file
for a classic agent:

.. code-block:: json

  {
    "seed": 2009,
    "nb_resources": 175,
    "nb_jobs": 1000,
    "workload_file_path": "/workspace/workloads/my_workload.swf",
    "platform_file_path": "/workspace/platforms/my_platform.json",
    "pybatsim": {
      "log_level": "DEBUG",
      "env": {
        "objective": "avg_utilization",
        "actions": {
          "selection": [
            {"shortest": ["high_mem_bw"]}
          ],
          "void": false
        },
        "observation": "normal",
        "queue_sensitivity": 0.05
      },
      "agent": {
        "type": "CLASSIC"
      }
    }
  }


This is another example of an ``options.json`` file,
in this case for a learning agent:

.. code-block:: json

  {
    "seed": 1995,
    "nb_resources": 175,
    "nb_jobs": 1000,
    "workload_file_path": "/workspace/workloads/my_workload.swf",
    "platform_file_path": "/workspace/platforms/my_platform.json",
    "pybatsim": {
      "log_level": "WARNING",
      "env": {
        "objective": "makespan",
        "actions": {
          "selection": [
            {"first": ["high_gflops", "high_mem_bw"]},
            {"smallest": [""]}
          ],
          "void": false
        },
        "queue_sensitivity": 0.01
      },
      "agent": {
        "type": "LEARNING",
        "run": "train",
        "hidden": 128,
        "lr": 0.001,
        "gamma": 0.99
      }
    }
  }

Optional command line arguments are available:

- ``-a`` - The file containing your developed learning agent for evaluation.
  See `agent examples <TODO>`_ for reference.

- ``-cw`` - If you are thinking about proof-of-concept experiments, you
  may need to define your own workload. Doing this in SWF is tedious, thus
  this option allows for passing a custom workload defined in Batsim JSON format.

- ``-im`` - PyTorch trained models are usually saved in ``.pt`` files. This
  option allows for loading a previously trained model to bootstrap the agent.

- ``-om`` - If you want to save the model after the simulation is finished, specify
  the output file in this option. This is usually combined with *train* runs.

.. include-overview-end
