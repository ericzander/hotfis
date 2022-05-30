Example: Mamdani
================

The following is a basic example of Mamdani inference with the commonly
employed temperature example.

A fuzzy inference system (FIS) is created, input values are stored in a
mapping data structure such as a dictionary, and evaluation is performed.
The outputs are returned as Tuples with both a domain and codomain as Numpy
arrays. These outputs may be both defuzzified and plotted using Matplotlib.

.. code-block:: python

    # Imports
    import hotfis as hf
    import matplotlib.pyplot as plt  # For plotting functionality

    # --------------
    # FIS Definition
    # --------------

    # Define membership function groupset
    groupset = hf.FuzzyGroupset([
        # Define the temperature input group with a domain of 0-100
        hf.FuzzyGroup("temperature", 0, 100, [
            hf.FuzzyFunc("cold", [30, 40], "leftedge"),
            hf.FuzzyFunc("warm", [30, 40, 60, 70], "trapezoidal"),
            hf.FuzzyFunc("hot", [60, 70], "rightedge")
        ]),

        # Define the heater output group with a domain of 0-1
        hf.FuzzyGroup("heater", 0.0, 1.0, [
            hf.FuzzyFunc("off", [0.1, 0.2], "leftedge"),
            hf.FuzzyFunc("medium", [0.1, 0.2, 0.8, 0.9], "trapezoidal"),
            hf.FuzzyFunc("on", [0.8, 0.9], "rightedge")
        ]),
    ])

    # Define the fuzzy ruleset using group and function names from above
    ruleset = hf.FuzzyRuleset([
        hf.FuzzyRule("if temperature is cold then heater is on"),
        hf.FuzzyRule("if temperature is warm then heater is medium"),
        hf.FuzzyRule("if temperature is hot then heater is off"),
    ])

    # Create the fuzzy inference system
    fis = hf.FIS(groupset, ruleset)

    # --------------
    # FIS Evaluation
    # --------------

    # Inputs must map the input group names as strings to scalar or array-like inputs
    # Dictionary and Pandas DataFrames are examples of valid input formats
    temp_inputs = {"temperature": 67}

    # Get dictionary of outputs for each group
    all_outputs = fis.eval_mamdani(temp_inputs)

    # Get output tuple with domain and codomain for heater group
    heater_outputs = all_outputs["heater"]

    # Split domain and codomain of fuzzified output
    domain, codomain = heater_outputs

    # Defuzzify and print output
    final_output = fis.defuzz_mamdani(domain, codomain)
    print(f"Input  : {temp_inputs['temperature']}")
    print(f"Output : {final_output}")

    # Plot both the heater group and output
    fis.groupset["heater"].plot()
    fis.plot_mamdani(domain, codomain)
    plt.show()

**Outputs**::

    Input  : 67
    Output : 0.37079499

.. image:: ../_static/mamdani_ex.png
  :width: 500
