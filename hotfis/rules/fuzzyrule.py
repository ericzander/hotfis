"""Contains FuzzyRule definition.
"""

from __future__ import annotations  # Doc aliases
from typing import Tuple, List, Mapping
from numpy.typing import ArrayLike

import numpy as np

from hotfis import MembGroupset


# --------------------------
# Rule Component Definitions
# --------------------------

class _Antecedent:
    """Fuzzy rule antecedent definition.

    Contains info describing a single antecedent and a method for evaluation
    given an input.

    Attributes:
        group_name (str): Membership group name (ex: "temp").
        fn_name (str): Membership function name (ex: "cold").
        is_and (bool): Whether is antecedent is "and" (T) or "or" (F).
    """
    def __init__(self, group_name: str, fn_name: str, is_and: bool):
        self.group_name = group_name
        self.fn_name = fn_name
        self.is_and = is_and

    def eval(self, x: ArrayLike, groupset: MembGroupset) -> ArrayLike:
        # Calculate membership(s) to respective function in group
        return groupset[self.group_name][self.fn_name](x)


class _Consequent:
    """Contains info describing the rule's single consequent.

    Attributes:
        group_name (str): Membership group name (ex: "heater").
        fn_name (str): Membership function name (ex: "on").
    """
    def __init__(self, group_name: str, fn_name: str):
        self.group_name = group_name
        self.fn_name = fn_name


# ---------------
# Rule Definition
# ---------------

class FuzzyRule:
    """Fuzzy rule used to comprise rulesets in fuzzy inference systems.

    Rules can be constructed with natural language using the names of
    membership function groups and functions.

    Args:
        rule_text: A rule as a string.

    Attributes:
        antecedents (List[_Antecedent]): List of rule antecedents.
        consequent (_Consequent): Rule consequent.

    Example:
        >>> rule = FuzzyRule("if temperature is cold then heater is on")
    """
    # -----------
    # Constructor
    # -----------

    def __init__(self, rule_text: str):
        antecedents, consequent = self._read_rule(rule_text)

        self.antecedents = antecedents
        self.consequent = consequent

    # -------
    # Methods
    # -------

    def evaluate(self, x: Mapping[str, ArrayLike],
                 groupset: MembGroupset) -> Tuple[str, str, ArrayLike]:
        """Evaluates the rule given valid input values and compatible groupset.

        The inputs can be dictionaries or similar data structures where the
        keys are membership function group names (ex. 'temperature') and
        the values are either a float or iterable of floats representing
        values.

        Args:
            x: Input values as dictionary with groups as keys and a scalar or
                array-like as values. Also supports other data structures
                accessible by group name.
            groupset: Groupset of membership functions required for evaluation.

        Returns:
            Tuple with output group name, function name, and value.
        """
        prev_ms = None  # Membership of previous antecedent

        for antecedent in self.antecedents:
            # Calculate antecedent membership
            try:
                ante_input = np.asarray(x[antecedent.group_name])
                membership = antecedent.eval(ante_input, groupset)
            except KeyError:
                err_str = f"Rule - Failed to find an input or membership function " \
                          f"for the rule's '{antecedent.group_name}' group."
                raise KeyError(err_str)

            # Update membership depending if antecedent is 'and' or 'or'
            if prev_ms is None:
                prev_ms = membership
            elif antecedent.is_and:
                prev_ms = np.minimum(prev_ms, membership)
            else:
                prev_ms = np.maximum(prev_ms, membership)

        final_membership = prev_ms.item() if prev_ms.size == 1 else prev_ms

        return self.consequent.group_name, self.consequent.fn_name, final_membership

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
