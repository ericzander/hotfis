"""Membership function group testing.
"""

import matplotlib.pyplot as plt

from hotfis.membership.membfunc import MembFunc
from hotfis.membership.membgroup import MembGroup


def main():
    group = MembGroup("test", 0, 2, [
        MembFunc("fn1", [0, 1], "leftedge"),
        MembFunc("fn2", [0, 1, 2], "triangular"),
        MembFunc("fn3", [1, 2], "rightedge"),
    ])

    for fn in group:
        print(fn.center)

    group.plot()
    plt.show()


if __name__ == "__main__":
    main()
