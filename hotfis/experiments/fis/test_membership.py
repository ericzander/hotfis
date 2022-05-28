"""Basic fuzzy inference system testing.
"""

import hotfis as hf

import matplotlib.pyplot as plt


def main():
    # Fuzzy inference system
    fis = hf.FIS(
        # Define membership functions
        hf.FuzzyGroupset([
            # Input group
            hf.FuzzyGroup("temperature", 30, 70, [
                hf.FuzzyFunc("cold", [30, 40], "leftedge"),
                hf.FuzzyFunc("warm", [30, 40, 60, 70], "trapezoidal"),
                hf.FuzzyFunc("hot", [60, 70], "rightedge")
            ]),

            # Output group
            hf.FuzzyGroup("heater", 0.0, 1.0, [
                hf.FuzzyFunc("off", [0.1], "tsk"),
                hf.FuzzyFunc("medium", [0.5], "tsk"),
                hf.FuzzyFunc("on", [0.9], "tsk")
            ]),
        ]),

        # Define ruleset
        hf.FuzzyRuleset([
            hf.FuzzyRule("if temperature is cold then heater is on"),
            hf.FuzzyRule("if temperature is warm then heater is medium"),
            hf.FuzzyRule("if temperature is hot then heater is off"),
        ])
    )

    inputs = {"temperature": [32, 73]}

    memb_outputs = fis.eval_membership(inputs)

    print("done")


if __name__ == '__main__':
    main()
