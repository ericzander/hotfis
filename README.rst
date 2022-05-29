HotFIS
======

**Create, alter, and visualize fuzzy inference systems in Python!**

Description
===========

HotFIS is a library designed to support
`fuzzy logic <https://en.wikipedia.org/wiki/Fuzzy_logic>`_ as well as the
streamlined creation and evaluation of fuzzy inference systems (FIS) in Python.

The library leverages `Numpy <https://numpy.org>`_ to evaluate scalar and
array-like input via `Mamdani <https://doi.org/10.1016/S0020-7373(75)80002-2.>`_
or `Takagi-Sugeno <https://ieeexplore.ieee.org/document/6313399>`_ inference.
Additionally, `Matplotlib <https://matplotlib.org>`_ is employed to quickly
visualize output and support explainability.

Note: *HotFIS is currently in early development!* Any suggested
features and improvements are welcome.

Main Features
=============

* Creation of functions capable of determining membership to an implicitly defined fuzzy set
* Organization of function in groups with a domain for Mamdani evaluation and visualization
* Creation of fuzzy rules with one or more antecedents using natural language
* Deserialization of both membership functions and fuzzy rulesets
* Mamdani and Takagi-Sugeno-Kang inference with scalar and array-like inputs
* Visualization of membership functions, function groups, and fuzzfied Mamdani output
* Networks of multiple fuzzy inference systems
* Experimental conversion of Takagi-Sugeno output functions to Mamdani outputs for explainability

Installation
============

Currently, HotFIS is hosted only on TestPyPI.

Installation with all dependencies::

    pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple hotfis

Installation with only HotFIS::

    pip install -i https://test.pypi.org/simple/ hotfis

Dependencies
============

HotFIS depends on:

* `Numpy <https://numpy.org>`_
* `Matplotlib <https://matplotlib.org>`_

Documentation
=============

Documentation can be found at:

* https://ericzander.github.io/hotfis
