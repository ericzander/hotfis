"""Fuzzy rule testing.
"""

from hotfis import FuzzyGroupset, FuzzyRule


def main():
    gset = FuzzyGroupset("../objects/groupset1.txt")

    rule1 = FuzzyRule("if temperature is cold then heater is on")
    output1 = rule1.evaluate({"temperature": 35}, gset)
    output2 = rule1.evaluate({"temperature": [35, 41, 66, 32]}, gset)
    output3 = rule1.evaluate({"temperature": [[35, 41], [66, 32]]}, gset)

    print("done")


if __name__ == '__main__':
    main()
