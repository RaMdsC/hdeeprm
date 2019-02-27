HDeepRM
=======

Framework for evaluating Workload Management policies based on
Deep Reinforcement Learning for Heterogeneous Clusters.

.. include-overview-start

Overview
--------

HDeepRM is a Python framework for evaluating Workload Management policies
based on Deep Reinforcement Learning for Heterogeneous Clusters. It
leverages the `Batsim ecosystem <https://gitlab.inria.fr/batsim>`_
for simulating a heterogeneous Workload Management context. This is composed
of the Simulator, `Batsim <https://gitlab.inria.fr/batsim/batsim>`_ and the
Decision System, `PyBatsim <https://gitlab.inria.fr/batsim/pybatsim>`_.

HDeepRM provides a heterogeneity layer on top of PyBatsim, which adds support
for user-defined resource hierarchies. Memory and bandwidth conflicts are added
along with interdependence when consolidating or scattering jobs across the
data centre.

It offers a flexible API for developing Deep Reinforcement Learning agents.
These may be trained by providing real workload traces in
`SWF format <http://www.cs.huji.ac.il/labs/parallel/workload/swf.html>`_ along
with platforms defined in the format specified in `Platforms <TODO>`_. They can
be further evaluated and tested against traditional policies.

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

In order to install HDeepRM, just download the package from PyPi:

.. code-block:: bash

  pip install --user hdeeprm

If ``pip`` is mapped to Python 2.x, try:

.. code-block:: bash

  pip3 install --user hdeeprm

When working with multiple Python versions, use:

.. code-block:: bash

  python3.6 -m pip install --user hdeeprm

This should download the ``hdeeprm`` package with all its dependencies,
which are:

- ``defusedxml`` >= 0.5.0: secure XML generation and parsing.
- ``gym`` >= 0.11.0: environment, actions and observations definitions.
- ``lxml`` >= 4.3.1: generation of the XML tree. Backend for ``defusedxml``.
- ``numpy`` >= 1.16.1: efficient data structure operations.
- ``procset`` >= 1.0: closed-interval sets for resource selection.
- ``pybatsim`` >= 3.1.0: decision system and main interface to interact
  with Batsim.
- ``torch`` >= 1.0.1: deep learning library for agent definition.

Usage Prerequisites
~~~~~~~~~~~~~~~~~~~

The simulation side is done by Batsim, which is needed in order to run
HDeepRM experiments. Follow the `official installation docs
<https://batsim.readthedocs.io/en/latest/installation.html>`_ for instructions.

Usage
~~~~~

In order to experiment with HDeepRM, an integrated launcher is provided:

.. code-block:: bash

  hdeeprm-launch <agent.py> <options.json> --inmodel <saved_model.pt> --outmodel <to_save_model.pt>

The ``agent.py`` file contains your developed agent for evaluation.
See `agent examples <TODO>`_ for reference.

The ``options.json`` specifies the experiment parameters. The JSON structure
is as follows:

.. code-block:: json

  {
    "seed": "",
    "nb_resources": "",
    "nb_jobs": "",
    "workload_file_path": "",
    "platform_file_path": "",
    "pybatsim": {
      "log_level": "",
      "env": {
        "objective": "",
        "queue_sensitivity": ""
      },
      "agent": {
        "policy_pair": "",
        "run": "",
        "hidden": "",
        "lr": "",
        "gamma": ""
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
* ``queue_sensitivity`` - Sensitivity of the observation to variations in job queue size. See `Hyperparameters - Queue Sensitivity <TODO>`_.

PyBatsim - Agent options:

* ``policy_pair`` - For `classic agents <https://hdeeprm.readthedocs.io/en/latest/source/packages/hdeeprm.agent.html#hdeeprm.agent.ClassicAgent>`_ and derived only. The job and resource selection policies. Policy pairs are further described in `Environment - Action Space <TODO>`_.
* ``run`` - For `learning agents <https://hdeeprm.readthedocs.io/en/latest/source/packages/hdeeprm.agent.html#hdeeprm.agent.LearningAgent>`_ and derived only.
  Type of run for the learning agent, can be *train* or *test*.
  When training, the agent's inner model is updated,
  whereas testing is meant for evaluation purposes.
* ``hidden`` - For `learning agents <https://hdeeprm.readthedocs.io/en/latest/source/packages/hdeeprm.agent.html#hdeeprm.agent.LearningAgent>`_ and derived only. Number of units in each hidden layer from the agent's inner model. See `Hyperparameters - Hidden units <TODO>`_.
* ``lr`` - For `learning agents <https://hdeeprm.readthedocs.io/en/latest/source/packages/hdeeprm.agent.html#hdeeprm.agent.LearningAgent>`_ and derived only. Learning rate for updating the agent's inner model. See `Hyperparameters - Learning rate <TODO>`_.
* ``gamma`` - For `learning agents <https://hdeeprm.readthedocs.io/en/latest/source/packages/hdeeprm.agent.html#hdeeprm.agent.LearningAgent>`_ and derived only. Discount factor for rewards. See `Hyperparameters - Reward Discount Factor <TODO>`_.

This is an example of an ``options.json`` file
for a `classic agent <https://hdeeprm.readthedocs.io/en/latest/source/packages/hdeeprm.agent.html#hdeeprm.agent.ClassicAgent>`_:

.. code-block:: json

  {
    "seed": 2009,
    "nb_resources": 2280,
    "nb_jobs": 10000,
    "workload_file_path": "/workspace/workloads/my_workload.swf",
    "platform_file_path": "/workspace/platforms/my_platform.json",
    "pybatsim": {
      "log_level": "DEBUG",
      "env": {
        "objective": "avg_utilization",
        "queue_sensitivity": 0.05
      },
      "agent": {
        "policy_pair": "shortest-high_flops"
      }
    }
  }


This is another example of an ``options.json`` file,
in this case for a `learning agent <https://hdeeprm.readthedocs.io/en/latest/source/packages/hdeeprm.agent.html#hdeeprm.agent.LearningAgent>`_:

.. code-block:: json

  {
    "seed": 1995,
    "nb_resources": 2280,
    "nb_jobs": 10000,
    "workload_file_path": "/workspace/workloads/my_workload.swf",
    "platform_file_path": "/workspace/platforms/my_platform.json",
    "pybatsim": {
      "log_level": "WARNING",
      "env": {
        "objective": "makespan",
        "queue_sensitivity": 0.01
      },
      "agent": {
        "run": "train",
        "hidden": 128,
        "lr": 0.001,
        "gamma": 0.99
      }
    }
  }

The ``inmodel`` optional argument may be used for providing a path
to a previously trained and saved model. HDeepRM will load this model
before starting the run.

The ``outmodel`` optional argument may be specified as a path for
saving the model after the run is finished. If not provide, the model
won't be saved. This is usually combined with *train* runs.

.. include-overview-end
