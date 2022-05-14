"""Contains FuzzyRule definition.
"""

from hotfis.fuzzy import MFGroupset

# ------------------------
# Nested Class Definitions
# ------------------------

class Antecedent:
    """
    Contains info describing a single antecedent and a method for
    evaluation given an input

    Attributes: #
        is_and: Indicates whether the antecedent is "and" (T) or "or" (F)
        mf_group_name: Membership group name (ex: "temp")
        mf_name: Membership function name (ex: "cold")
        eval(input, mf_groups): Evaluates an input
    """
    def __init__(self, is_and: bool, mf_group_name: str, mf_name: str):
        self.is_and = is_and
        self.mf_group_name = mf_group_name
        self.mf_name = mf_name

    def eval(self, input_val: float, mf_groupset: MFGroupset) -> float:
        # Get membership function group
        mf_group = mf_groupset[self.mf_group_name]

        # Calculate membership to respective function in group
        output = mf_group.calc_memberships(self.mf_name, input_val)

        return output

class Consequent:
    """
    Contains info describing the rule's single consequent

    Attributes:
        mf_group_name: Membership group name (ex: "heater")
        mf_name: Membership function name (ex: "on")
    """
    def __init__(self, mf_group_name: str, mf_name: str):
        self.mf_group_name = mf_group_name
        self.mf_name = mf_name