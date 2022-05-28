"""Membership function groupset testing.
"""

from hotfis import FuzzyGroupset

import matplotlib.pyplot as plt


def main():
    gset = FuzzyGroupset("../objects/groupset1.txt")

    for group in gset:
        print(group.name)

    gset["heater"].plot(0, 1)
    plt.show()


if __name__ == "__main__":
    main()
