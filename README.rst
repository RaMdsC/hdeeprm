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

.. include-overview-end
