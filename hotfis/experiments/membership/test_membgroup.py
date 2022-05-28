"""Membership function group testing.
"""

import matplotlib.pyplot as plt

from hotfis.membership.fuzzyfunc import FuzzyFunc
from hotfis.membership.fuzzygroup import FuzzyGroup


def main():
    group = FuzzyGroup("test", 0, 2, [
        FuzzyFunc("fn1", [0, 1], "leftedge"),
        FuzzyFunc("fn2", [0, 1, 2], "triangular"),
        FuzzyFunc("fn3", [1, 2], "rightedge"),
    ])

    for fn in group:
        print(fn.center)

    group.plot()
    plt.show()


if __name__ == "__main__":
    main()
