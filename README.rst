MBA DS Project
##############


.. contents::

.. section-numbering::


Installation
============

This project is built as a Python package. To use it, it is recommended a fresh Python virtual environment is created and the package installed in it using the following command:


.. code-block:: bash

    pip install -e /path/to/repository_root


Data
====

The package assumes the following data files are located in the ``data`` folder at the root of the repository:

* ffp_rollout_X.csv
* ffp_train.csv
* recommendations.csv
* reviews_rollout.csv
* reviews_training.csv
* text_rollout_X.csv
* text_training.csv


Use
===

**Note:** The package relies on being able to create new directories and files in the ``models`` folder in the root of the repository.


Notebooks
=========

Although none of the notebooks included in this repository is required to run the code and use it to produce predictions, they can be examined to understand how the final code was developed.
