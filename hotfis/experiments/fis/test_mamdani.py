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
                hf.FuzzyFunc("off", [0.1, 0.2], "leftedge"),
                hf.FuzzyFunc("medium", [0.1, 0.2, 0.8, 0.9], "trapezoidal"),
                hf.FuzzyFunc("on", [0.8, 0.9], "rightedge")
            ]),
        ]),

        # Define ruleset
        hf.FuzzyRuleset([
            hf.FuzzyRule("if temperature is cold then heater is on"),
            hf.FuzzyRule("if temperature is warm then heater is medium"),
            hf.FuzzyRule("if temperature is hot then heater is off"),
        ])
    )

    inputs = {"temperature": [[32, 56], [77, 0]]}
    heater_output = fis.eval_mamdani(inputs)["heater"]
    defuzzed = fis.defuzz_mamdani(heater_output)

    domain, codomains = heater_output

    fis.groupset["heater"].plot()
    fis.plot_mamdani(domain, codomains[0][0])
    plt.show()

    print("done")


if __name__ == '__main__':
    main()
