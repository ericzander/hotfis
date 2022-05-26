"""

"""

import hotfis as hf

import matplotlib.pyplot as plt

def main():
    # Fuzzy inference system
    fis = hf.FIS(
        # Define membership functions
        hf.MembGroupset([
            # Input group 1
            hf.MembGroup("service", 0, 10, [
                hf.MembFunc("poor", [3, 5], "leftedge"),
                hf.MembFunc("good", [3, 5, 7], "triangular"),
                hf.MembFunc("excellent", [5, 7], "rightedge")
            ]),

            # Input group 1
            hf.MembGroup("food", 0, 10, [
                hf.MembFunc("rancid", [4, 6], "leftedge"),
                hf.MembFunc("delicious", [4, 6], "rightedge")
            ]),

            # Output group
            hf.MembGroup("tip", 0, 30, [
                hf.MembFunc("cheap", [7], "tsk"),
                hf.MembFunc("average", [17], "tsk"),
                hf.MembFunc("generous", [26], "tsk")
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

    #fis.groupset["tip"].plot()
    new_fis.groupset["tip"].plot()

    plt.show()

    print("done")


if __name__ == '__main__':
    main()
