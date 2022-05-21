"""Basic fuzzy inference system testing.
"""

import hotfis as hf

import matplotlib.pyplot as plt


def main():
    # Fuzzy inference system
    fis = hf.FIS(
        # Define membership functions
        hf.MembGroupset([
            # Input group
            hf.MembGroup("temperature", 30, 70, [
                hf.MembFunc("cold", [30, 40], "leftedge"),
                hf.MembFunc("warm", [30, 40, 60, 70], "trapezoidal"),
                hf.MembFunc("hot", [60, 70], "rightedge")
            ]),

            # Output group
            hf.MembGroup("heater", 0.0, 1.0, [
                hf.MembFunc("off", [0.1], "tsk"),
                hf.MembFunc("medium", [0.5], "tsk"),
                hf.MembFunc("on", [0.9], "tsk")
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
