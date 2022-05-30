HotFIS
======

**Create, alter, and visualize fuzzy inference systems in Python!**

Description
===========

HotFIS is a library designed to support
`fuzzy logic <https://en.wikipedia.org/wiki/Fuzzy_logic>`_ through the
streamlined creation and evaluation of fuzzy inference systems (FIS) in Python.

In order to support machine learning and data science applications,
the library leverages `Numpy <https://numpy.org>`_ [1]_ to evaluate both scalar and
array-like input via Mamdani [2]_ or Takagi-Sugeno [3]_ inference.
Additionally, `Matplotlib <https://matplotlib.org>`_ [4]_ is employed to quickly
visualize output and support explainability.

Note: *HotFIS is currently in early development!* Any suggested
features and improvements are welcome.

Features
========

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

Currently, HotFIS is hosted on TestPyPI.

Installation with all dependencies::

    pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple hotfis

Installation with only HotFIS::

    pip install -i https://test.pypi.org/simple/ hotfis

Dependencies
============

HotFIS depends on:

* `Numpy <https://numpy.org>`_
* `Matplotlib <https://matplotlib.org>`_

References
==========

.. [1] C. R. Harris et al., “Array programming with NumPy,” Nature, vol. 585, no. 7825, pp. 357–362, Sep. 2020, doi: 10.1038/s41586-020-2649-2.
.. [2] E. H. Mamdani and S. Assilian, “An experiment in linguistic synthesis with a fuzzy logic controller,” International Journal of Man-Machine Studies, vol. 7, no. 1, pp. 1–13, Jan. 1975, doi: 10.1016/s0020-7373(75)80002-2.
.. [3] T. Takagi and M. Sugeno, "Fuzzy identification of systems and its applications to modeling and control," in IEEE Transactions on Systems, Man, and Cybernetics, vol. SMC-15, no. 1, pp. 116-132, Jan.-Feb. 1985, doi: 10.1109/TSMC.1985.6313399.
.. [4] J. D. Hunter, “Matplotlib: A 2D Graphics Environment,” Computing in Science & Engineering, vol. 9, no. 3, pp. 90–95, 2007, doi: 10.1109/mcse.2007.55.

Documentation
=============

Documentation can be found at:

* https://ericzander.github.io/hotfis
