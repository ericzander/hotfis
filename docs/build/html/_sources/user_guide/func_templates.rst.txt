Guide: Functions
================

.. contents::

Intro
-----

FuzzyFuncs are membership functions corresponding to implicitly defined fuzzy
sets that form the basis for the creation and evalutation of fuzzy inference
systems.

There are three arguments required to build a FuzzyFunc: a name, a list of
parameters, and how membership should be calculated. There are three ways to
define the last.

1. Passing a template name as a string is interpreted as one of the template
   functions. See the Templates_ section below.
2. Giving a list of values from 0.0 to 1.0 for each parameter creates a
   piecewise linear function. See the Linear_ subsection below.
3. Passing a callable that takes an array-like input as the first
   parameter and a list of the function's parameters as the second creates
   a special function. See the Special_ subsection below.

Templates
---------

Linear
^^^^^^

Linear membership functions are initialized with a list of membership values
corresponding to each parameter. A piecewise function is then formed between
each parameter-membership point.

Each linear template function is equivalent to passing the stated list
of parameter memberships in place of the template name.


Triangular: 'triangular'
""""""""""""""""""""""""

Parameter Membership:

    **[0, 1, 0]**

Example:

.. code-block:: python

    FuzzyFunc("hot", [-1, 0, 1], "triangular")

.. image:: ../_static/fuzzyfuncs/triangular.png
  :width: 380

Trapezoidal: 'trapezoidal'
""""""""""""""""""""""""""

Parameter Membership:

    **[0, 1, 1, 0]**

Example:

.. code-block:: python

    FuzzyFunc("hot", [-2, -1, 1, 2], "triangular")

.. image:: ../_static/fuzzyfuncs/trapezoidal.png
  :width: 380

Left Edge: 'leftedge'
"""""""""""""""""""""

Parameter Membership:

    **[1, 0]**

Example:

.. code-block:: python

    FuzzyFunc("hot", [-1, 1], "leftedge")

.. image:: ../_static/fuzzyfuncs/leftedge.png
  :width: 380

Right Edge: 'rightedge'
"""""""""""""""""""""""

Parameter Membership:

    **[0, 1]**

Example:

.. code-block:: python

    FuzzyFunc("hot", [-1, 1], "rightedge")

.. image:: ../_static/fuzzyfuncs/rightedge.png
  :width: 380

Special
^^^^^^^

Unlike linear templates, special templates save a callable instead of a
list of membership values. The parameters are then used by the callable.

Each special template function is equivalent to passing the stated callable
in place of the template name.

Gaussian: 'gaussian'
""""""""""""""""""""

Parameters:

    **[mean, standard deviation]**

Function::

    lambda a, params: exp(-((a - params[0]) ** 2 / (2 * params[1] ** 2)))

Example:

.. code-block:: python

    FuzzyFunc("hot", [0, 1], "gaussian")

.. image:: ../_static/fuzzyfuncs/gaussian.png
  :width: 380

Takagi-Sugeno-Kang
^^^^^^^^^^^^^^^^^^

HotFIS currently only supports zeroth-order Takagi-Sugeno inference. The sole
parameter corresponds to the output constant associated with the function.

Takagi-Sugeno-Kang functions may only be used as functions in output groups.
Stated another way, they may only be stated as rule consequents evaluated
using fis.eval_tsk.

Takagi-Sugeno-Kang: 'tsk'
"""""""""""""""""""""""""

Example:

.. code-block:: python

    FuzzyFunc("hot", [0], "tsk")

.. image:: ../_static/fuzzyfuncs/tsk.png
  :width: 380
