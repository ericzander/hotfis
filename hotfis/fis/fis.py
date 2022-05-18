"""Contains Fuzzy Inference System (FIS) definition.
"""

from __future__ import annotations  # Doc aliases
from typing import Union, Mapping, Dict, Tuple
from numpy.typing import ArrayLike

import numpy as np

from hotfis import MembFunc, MembGroup, MembGroupset, FuzzyRule, FuzzyRuleset


class FIS:
    """Fuzzy inference system (FIS) comprised of membership functions and a ruleset.

    Args:
        groupset: Groupset of membership functions or path to file with groups.
        ruleset: Ruleset to be evaluated or path to file with rules.

    Attributes:
        groupset (MembGroupset): Groupset of membership functions required for evaluation.
        ruleset (FuzzyRuleset): Ruleset to be evaluated.

    Example:
        Method 1:

        >>> fis1 = FIS(
        >>>     # Create membership functions
        >>>     MembGroupset([
        >>>         # Create temperature group for input
        >>>         MembGroup("temperature", [
        >>>             MembFunc("cold", [30, 40], "leftedge", ),
        >>>             MembFunc("warm", [30, 40, 60, 70], "trapezoidal"),
        >>>             MembFunc("hot", [60, 70], "rightedge")
        >>>         ]),
        >>>
        >>>         # Create heater group for output
        >>>         MembGroup("heater", [
        >>>             MembFunc("off", [0.1, 0.2], "leftedge"),
        >>>             MembFunc("medium", [0.1, 0.2, 0.8, 0.9], "trapezoidal"),
        >>>             MembFunc("on", [0.8, 0.9], "rightedge")
        >>>         ]),
        >>>     ]),
        >>>
        >>>     # Create ruleset
        >>>     FuzzyRuleset([
        >>>         FuzzyRule("if temperature is cold then heater is on"),
        >>>         FuzzyRule("if temperature is warm then heater is medium"),
        >>>         FuzzyRule("if temperature is hot then heater is off"),
        >>>     ])
        >>> )

        Method 2::

            === example_groups.txt ===

            group temperature
            leftedge cold 30 40
            trapezoidal warm 30 40 60 70
            rightedge hot 60 70

            group heater
            leftedge off 0.1 0.2
            trapezoidal medium 0.1 0.2 0.8 0.9
            rightedge on 0.8 0.9

            === example_rules.txt ===

            if service is poor or food is rancid then tip is cheap
            if service is good then tip is average
            if service is excellent and food is delicioius then tip is generous

        >>> fis2 = FIS(
        >>>     "example_groups.txt",
        >>>     "example_rules.txt",
        >>> )
    """
    # -----------
    # Constructor
    # -----------

    def __init__(self, groupset: Union[str, MembGroupset],
                 ruleset: Union[str, FuzzyRuleset]):
        # Save or create membership function groupset and ruleset
        self.groupset = MembGroupset(groupset) if isinstance(groupset, str) else groupset
        self.ruleset = FuzzyRuleset(ruleset) if isinstance(ruleset, str) else ruleset

    # ------------------
    # Evaluation Methods
    # ------------------

    def eval_membership(self, x: Mapping[str, ArrayLike]) -> Dict[str, Dict[str, float]]:
        """Given input, evaluates memberships to all output functions.

        Args:
            x: Input values as dictionary with input groups names as keys and
                a scalar or array-like as values. Also supports other data
                structures similarly accessible by input group name.

        Returns:
            Dictionary with output group names as keys and values that are
            dictionaries. These have membership function names as keys and
            memberships as values.

        Example:
            >>> fis = FIS("example_groups.txt", "example_rules.txt")
            >>> memberships = fis.eval_membership({
            >>>     "temperature": [32, 56, 77, 55, 33, 21, 90]
            >>> })
        """
        all_outputs = dict()

        for rule in self.ruleset:
            # Get membership function names and corresponding membership values
            group, fn, output = rule.evaluate(x, self.groupset)

            # Create new group dictionary if not already created
            if group not in all_outputs:
                all_outputs[group] = dict()

            # Take max if old mf value exists or just save if it does not
            if fn in all_outputs[group]:
                prev = all_outputs[group][fn]
                all_outputs[group][fn] = max(prev, output)
            else:
                all_outputs[group][fn] = output

        return all_outputs

    def raw_to_mamdani(self, memberships: Dict[str, Dict[str, float]],
                       num_points: int = 100) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Converts membership values to fuzzified Mamdani output.

        Args:
            memberships: Dictionary with output group names as keys and values
                that are dictionaries. These have membership function names as
                keys and memberships as values.
            num_points: Number of points to evaluate when building output.

        Returns:
            Dictionary with output group names as keys and values that are
            dictionaries with membership function names as keys and memberships
            as values.
        """
        pass

    def raw_to_tsk(self, memberships: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Converts membership values to defuzzified Takagi-Sugeno output.

        Args:
            memberships: Dictionary with output group names as keys and values
                that are dictionaries. These have membership function names as
                keys and memberships as values.

        Returns:
            Dictionary with output group names as keys and memberships as values.
        """
        pass
