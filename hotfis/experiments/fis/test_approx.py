"""

"""

import hotfis as hf

import matplotlib.pyplot as plt

def main():
    # Fuzzy inference system
    fis = hf.FIS(
        # Define membership functions
        hf.FuzzyGroupset([
            # Input group 1
            hf.FuzzyGroup("service", 0, 10, [
                hf.FuzzyFunc("poor", [3, 5], "leftedge"),
                hf.FuzzyFunc("good", [3, 5, 7], "triangular"),
                hf.FuzzyFunc("excellent", [5, 7], "rightedge")
            ]),

            # Input group 2
            hf.FuzzyGroup("food", 0, 10, [
                hf.FuzzyFunc("rancid", [4, 6], "leftedge"),
                hf.FuzzyFunc("delicious", [4, 6], "rightedge")
            ]),

            # Output group
            hf.FuzzyGroup("tip", 0, 30, [
                hf.FuzzyFunc("cheap", [7], "tsk"),
                hf.FuzzyFunc("average", [17], "tsk"),
                hf.FuzzyFunc("generous", [26], "tsk")
            ]),
        ]),

        # Define ruleset
        hf.FuzzyRuleset([
            "if service is poor or food is rancid then tip is cheap",
            "if service is good then tip is average",
            "if service is excellent or food is delicious then tip is generous",
        ])
    )

    inputs = {
        "service": [2, 7],
        "food": [4, 9]
    }

    new_fis = fis.approximate_mamdani()

    _, ax2 = new_fis.groupset["tip"].plot(6, 27)
    fis.groupset["tip"].plot(6, 27)
    ax2.remove()

    plt.show()

    print("done")


if __name__ == '__main__':
    main()
