"""Membership function testing.
"""

import matplotlib.pyplot as plt

from hotfis.src.fuzzy.membfunc import MembFunc


def main():
    fn = MembFunc([0, 2, 4], [0, 1, 0])
    # fn = MembFunc([0, 2, 4, 6], "trapezoidal")
    # fn = MembFunc([2, 1], "gaussian")

    print(fn(2.3))

    fn.plot(-2, 8, color="grey")
    plt.show()


if __name__ == "__main__":
    main()
