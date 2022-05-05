"""Membership function testing.
"""

import matplotlib.pyplot as plt

from hotfis.src.fuzzy.membfunc import MembFunc


def main():
    # fn = MembFunc([0, 2, 4], [0, 1, 0])
    # fn = MembFunc([0, 2, 4, 6], "trapezoidal")
    # fn = MembFunc([2, 1], "gaussian")
    fn = MembFunc([0, 2, 4, 5, 9], [0.2, 0.4, 1.0, 0.65, 1.0])

    fn.plot(-2, 10)
    plt.show()


if __name__ == "__main__":
    main()
