"""Membership function group testing.
"""

import matplotlib.pyplot as plt

from hotfis.membership.membfunc import MembFunc
from hotfis.membership.membgroup import MembGroup


def main():
    fns = MembGroup("test", [
        MembFunc("fn1", [0, 1], "leftedge"),
        MembFunc("fn2", [0, 1, 2], "triangular"),
        MembFunc("fn3", [1, 2], "rightedge"),
    ])

    for fn in fns:
        print(fn.center)

    fns.plot()
    plt.show()


if __name__ == "__main__":
    main()
