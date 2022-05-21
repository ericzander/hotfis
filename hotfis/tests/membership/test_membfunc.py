"""Membership function testing.
"""

import numpy as np
import matplotlib.pyplot as plt

from hotfis.membership.membfunc import MembFunc


def main():
    fn = MembFunc("fn1", [0, 2, 4], [0, 1, 0])
    # fn = MembFunc("fn2", [0, 2, 4, 6], "trapezoidal")
    # fn = MembFunc("fn3", [2, 1], "gaussian")
    # fn = MembFunc("fn4", [0, 2, 4, 5, 9], [0.2, 0.4, 1.0, 0.65, 1.0])

    output1 = fn(np.array([[1.0, 2.1], [3.0, 2.5]]))
    output2 = fn([[1.0, 1.9], [3.0, 2.5]])
    output3 = fn([1.0, 1.1, 3.2])

    fn.plot(fn.params[0], fn.params[-1])
    plt.show()


if __name__ == "__main__":
    main()
