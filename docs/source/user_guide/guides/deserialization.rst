Guide: Deserialization
======================

Membership function groupsets, fuzzy rulesets, and FISs can be defined in text
files and created by passing the relative filepath to the constructor.

FuzzyGroupset
-------------

Only template functions can be constructed via deserialization. See
:doc:`func_templates` for more info on function construction.

*example_groupset.txt*::

    group temperature
    leftedge cold 30 40
    trapezoidal warm 30 40 60 70
    rightedge hot 60 70
    domain 0 100

    group heater
    leftedge off 0.1 0.2
    trapezoidal medium 0.1 0.2 0.8 0.9
    rightedge on 0.8 0.9
    domain 0.0 1.0

Code:

.. code-block:: python

    FuzzyGroupset("example_groupset.txt")

FuzzyRuleset
------------

*example_ruleset.txt*::

    if temperature is cold then heater is on
    if temperature is warm then heater is medium
    if temperature is hot then heater is off

Code:

.. code-block:: python

    FuzzyRuleset("example_ruleset.txt")

FIS
---

.. code-block:: python

    FIS("example_groupset.txt", "example_ruleset.txt")

