"""Membership function group testing.
"""

import matplotlib.pyplot as plt

from hotfis.src.fuzzy.membfunc import MembFunc
from hotfis.src.fuzzy.mfgroup import MFGroup


def main():
    fns = MFGroup([
        MembFunc([0, 1], "leftedge"),
        MembFunc([0, 1, 2], "triangular"),
        MembFunc([1, 2], "rightedge"),
        #MembFunc([0, 1, 3, 4, 8], [0.4, 0.9, 0.65, 0.9, 0.3]),
        #MembFunc([0, 2, 4, 5, 9], [0.2, 0.4, 1.0, 0.65, 1.0])
    ])

    for fn in fns:
        print(fn.center)

    fns.plot(-1, 10)
    plt.show()


if __name__ == "__main__":
    main()