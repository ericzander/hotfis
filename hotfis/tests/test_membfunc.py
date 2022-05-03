"""Membership function testing.
"""

from hotfis.src.fuzzy.membfunc import MembFunc


def main():
    fn = MembFunc([0, 2, 4], [0, 1, 0])

    print(fn.name)
    print(fn(2.5))


if __name__ == "__main__":
    main()
