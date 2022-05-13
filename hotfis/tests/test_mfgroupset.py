"""Membership function groupset testing.
"""

from hotfis.src.fuzzy.mfgroupset import MFGroupset

import matplotlib.pyplot as plt

def main():
    gset = MFGroupset("fuzzy_objects/ex_group1.txt")

    for group in gset:
        print(group.name)

    gset["heater"].plot(0, 1)
    plt.show()

if __name__ == "__main__":
    main()
