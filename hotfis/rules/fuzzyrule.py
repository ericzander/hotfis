"""Contains FuzzyRule definition.
"""

from typing import Tuple, List, Mapping

import numpy as np
from numpy.typing import ArrayLike

from hotfis.fuzzy import MFGroupset


# --------------------------
# Rule Component Definitions
# --------------------------

class _Antecedent:
    """Fuzzy rule antecedent definition.

    Contains info describing a single antecedent and a method for evaluation
    given an input.

    Attributes:
        mf_group_name (str): Membership group name (ex: "temp").
        mf_name (str): Membership function name (ex: "cold").
        is_and (bool): Whether is antecedent is "and" (T) or "or" (F).
    """
    def __init__(self, mf_group_name: str, mf_name: str, is_and: bool):
        self.mf_group_name = mf_group_name
        self.mf_name = mf_name
        self.is_and = is_and

    def eval(self, input_val: np.ndarray, mf_groupset: MFGroupset) -> float:
        # Calculate membership to respective function in group
        return mf_groupset[self.mf_group_name][self.mf_name](input_val)


class _Consequent:
    """Contains info describing the rule's single consequent.

    Attributes:
        mf_group_name (str): Membership group name (ex: "heater").
        mf_name (str): Membership function name (ex: "on").
    """
    def __init__(self, mf_group_name: str, mf_name: str):
        self.mf_group_name = mf_group_name
        self.mf_name = mf_name


# ---------------
# Rule Definition
# ---------------

class FuzzyRule:
    """Fuzzy rule used to comprise rulesets in fuzzy inference systems.

    Rules are constructed with natural language in the form of strings using
    the names of membership function objects.

    Attributes:
        antecedents (List[_Antecedent]): List of rule antecedents.
        consequent (_Consequent): Rule consequent.
    """
    # -----------
    # Constructor
    # -----------

    def __init__(self, rule_text: str):
        """Fuzzy rule constructor.

        Constructs a rule by parsing natural language as follows.

        Args:
            rule_text: A rule as a string.

        Format:
            |  *"if {group} is {membfunc} then {group} is {membfunc}"*

        Example:
            |  *rule = FuzzyRule("if temperature is cold then heater is on")*
        """
        antecedents, consequent = self._read_rule(rule_text)

        self.antecedents = antecedents
        self.consequent = consequent

    # -------
    # Methods
    # -------

    def evaluate(self, inputs: Mapping[str, np.typing.ArrayLike],
                 mf_groupset: MFGroupset) -> Tuple[str, str, float]:
        """Evaluates the rule given valid input values and compatible groupset.

        The inputs can be dictionaries or similar data structures where the
        keys are membership function group names (ex. 'temperature') and
        the values are either a float or iterable of floats representing
        values.

        Args:
            inputs: Input group names and values.
            mf_groupset: groupset of required membership function groups.

        Returns:
            Tuple with output group name, function name, and value.
        """
        prev_ms = None  # Membership of previous antecedent

        for antecedent in self.antecedents:
            # Calculate antecedent membership
            try:
                ante_input = np.asarray(inputs[antecedent.mf_group_name])
                membership = antecedent.eval(ante_input, mf_groupset)
            except KeyError:
                err_str = f"Rule - Failed to find an input or membership function " \
                          f"for the rule's '{antecedent.mf_group_name}' group."
                raise KeyError(err_str)

            # Update membership depending if antecedent is 'and' or 'or'
            if prev_ms is None:
                prev_ms = membership
            elif antecedent.is_and:
                prev_ms = min(prev_ms, membership)
            else:
                prev_ms = max(prev_ms, membership)

        return self.consequent.mf_group_name, self.consequent.mf_name, prev_ms

    # ---------------
    # Reading Methods
    # ---------------

    @staticmethod
    def _read_rule(rule_text: str) -> Tuple[List[_Antecedent], _Consequent]:
        """Reads a rule from natural language in a given string.

        Args:
            rule_text: A rule as a string.

        Returns:
            Tuple with list of rule antecedents and its consequent.
        """
        rule = rule_text.split()

        antecedents = []       # List of _Antecedent objects
        consequent = None      # _Consequent objects
        end_clause = False     # Indicates if next word is mf name
        is_and = False         # Indicates whether antecedent is and/or
        mf_g_name = None       # Membership function group name of clause
        is_consequent = False  # Indicates if processing consequent

        for word in rule:
            word = word.lower()

            if not is_consequent:
                # If reading antecedents
                if word in ("if", "or"):
                    is_and = False
                elif word == "and":
                    is_and = True
                elif word == "is":
                    end_clause = True
                elif end_clause:
                    end_clause = False
                    antecedents.append(_Antecedent(mf_g_name, word, is_and))
                elif word == "then":
                    is_consequent = True
                else:
                    mf_g_name = word
            else:
                # If reading consequent
                if word == "is":
                    end_clause = True
                elif end_clause:
                    consequent = _Consequent(mf_g_name, word)
                else:
                    mf_g_name = word

        return antecedents, consequent
