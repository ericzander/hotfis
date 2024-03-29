"""Membership function testing.
"""

from hotfis import *

import matplotlib.pyplot as plt


def main():
    sprout = create_network()

    print(sprout.req_inputs())

    sprout.display()

    # Eval

    inputs = {"time": 2, "cloudy": 3, "humidity": 9, "rain": 3}

    membs = sprout.eval_membership(inputs)
    mams = sprout.eval_mamdani(inputs)
    tsks = sprout.eval_tsk(inputs)

    sprout.groupset["growth"].plot()
    sprout.plot_mamdani(*mams["Growth"]["growth"])
    plt.show()

    print("done")


def create_network():
    groupset = FuzzyGroupset([
        FuzzyGroup("cloudy", 0, 10, [
            FuzzyFunc("clear", [4, 6], "leftedge"),
            FuzzyFunc("overcast", [4, 6], "rightedge"),
        ]),
        FuzzyGroup("time", 0, 10, [
            FuzzyFunc("day", [3, 7], "leftedge"),
            FuzzyFunc("night", [3, 7], "rightedge"),
        ]),
        FuzzyGroup("humidity", 0, 10, [
            FuzzyFunc("low", [4, 7], "leftedge"),
            FuzzyFunc("high", [4, 7], "rightedge"),
        ]),
        FuzzyGroup("rain", 0, 10, [
            FuzzyFunc("light", [1, 5], "leftedge"),
            FuzzyFunc("heavy", [1, 5], "rightedge"),
        ]),
        FuzzyGroup("sun", 0, 10, [
            FuzzyFunc("low", [2, 3], "leftedge"),
            FuzzyFunc("high", [2, 3], "rightedge"),
        ]),
        FuzzyGroup("water", 0, 10, [
            FuzzyFunc("low", [4, 5], "leftedge"),
            FuzzyFunc("high", [4, 5], "rightedge"),
        ]),
        FuzzyGroup("growth", 0, 10, [
            FuzzyFunc("low", [2, 4], "leftedge"),
            FuzzyFunc("high", [2, 4], "rightedge"),
        ])
    ])

    sun_rs = FuzzyRuleset([
        "if cloudy is clear and time is day then sun is high",
        "if cloudy is overcast then sun is low",
        "if time is night then sun is low",
    ])

    water_rs = FuzzyRuleset([
        "if humidity is high or rain is heavy then water is high",
        "if humidity is low and rain is light then water is low",
    ])

    growth_rs = FuzzyRuleset([
        "if sun is high and water is high then growth is high",
        "if sun is low then growth is low",
        "if water is low then growth is low",
    ])

    # Create fuzzy network
    sprout = FuzzyNetwork(groupset)

    # Insert growth FIS as root
    sprout.insert({"Growth": growth_rs})

    # Insert sun FIS and water FIS as branches of growth
    sprout["Growth"].insert({"Sun": sun_rs, "Water": water_rs})

    return sprout


if __name__ == "__main__":
    main()
