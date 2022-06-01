import hotfis as hf

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn")

def main():
    iris = load_data()
    fis = create_fis()

    memberships = fis.eval_membership(iris)

    tsk_output = fis.eval_tsk(iris)["virginica"]

    print("done")


def load_data():
    # Load data
    data = load_iris()

    # Load X
    iris = pd.DataFrame(data=data.data, columns=data.feature_names)
    iris.columns = iris.columns.str.rstrip(" (cm)")
    iris.columns = iris.columns.str.replace(' ', '')

    # Load y
    iris["species"] = pd.Series(data.target)
    iris["species"] = pd.Categorical.from_codes(data.target, data.target_names)

    # Remove setosa
    iris = iris[iris["species"] != "setosa"]

    return iris

def create_fis():
    # Define membership function groupset
    groupset = hf.FuzzyGroupset([
        # Petal width
        hf.FuzzyGroup("sepalwidth", 1.8, 3.9, [
            hf.FuzzyFunc("small", [2.3, 3.2], "leftedge"),
            hf.FuzzyFunc("large", [2.5, 3.3], "rightedge")
        ]),

        # Petal length
        hf.FuzzyGroup("sepallength", 4.7, 8.1, [
            hf.FuzzyFunc("small", [5.5, 6.7], "leftedge"),
            hf.FuzzyFunc("large", [5.7, 6.9], "rightedge")
        ]),

        # Petal width
        hf.FuzzyGroup("petalwidth", 0.8, 2.6, [
            hf.FuzzyFunc("small", [1.4, 1.7], "leftedge"),
            hf.FuzzyFunc("large", [1.5, 1.7], "rightedge")
        ]),

        # Petal length
        hf.FuzzyGroup("petallength", 2.8, 7.1, [
            hf.FuzzyFunc("small", [4.6, 5.1], "leftedge"),
            hf.FuzzyFunc("large", [4.7, 5.2], "rightedge")
        ]),

        # Output
        hf.FuzzyGroup("virginica", -1.0, 1.0, [
            hf.FuzzyFunc("unlikely", [-1.0], "tsk"),
            hf.FuzzyFunc("likely", [1.0], "tsk")
        ]),
    ])

    # Define the fuzzy ruleset using group and function names from above
    ruleset = hf.FuzzyRuleset([
        hf.FuzzyRule("if sepalwidth is small then virginica is unlikely"),
        hf.FuzzyRule("if sepalwidth is large then virginica is likely"),

        hf.FuzzyRule("if sepallength is small then virginica is unlikely"),
        hf.FuzzyRule("if sepallength is large then virginica is likely"),

        hf.FuzzyRule("if petalwidth is small then virginica is unlikely"),
        hf.FuzzyRule("if petalwidth is large then virginica is likely"),

        hf.FuzzyRule("if petallength is small then virginica is unlikely"),
        hf.FuzzyRule("if petallength is large then virginica is likely"),
    ])

    fis = hf.FIS(groupset, ruleset)

    return fis


if __name__ == "__main__":
    main()