"""Contains Fuzzy Inference System (FIS) definition.
"""

from __future__ import annotations  # Doc aliases
from typing import Union, Mapping, Dict, Tuple
from numpy.typing import ArrayLike

import numpy as np
import matplotlib.pyplot as plt

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
    # Membership Methods
    # ------------------

    # Membership Evaluation

    def eval_membership(self, x: Mapping[str, ArrayLike]) -> Dict[str, Dict[str, ArrayLike]]:
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
                all_outputs[group][fn] = max(all_outputs[group][fn], output)
            else:
                all_outputs[group][fn] = output

        return all_outputs

    # ---------------
    # Mamdani Methods
    # ---------------

    def eval_mamdani(self, x: Mapping[str, ArrayLike],
                     num_points: int = 100) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Evaluates input and returns fuzzified Mamdani output.

        Args:
            x: Input values as dictionary with input groups names as keys and
                a scalar or array-like as values. Also supports other data
                structures similarly accessible by input group name.
            num_points: Number of points to evaluate when building output.

        Returns:
            Dictionary with output group names as keys and values that are
            tuples with two arrays. The first is the domain of the
            membership function. The second is a corresponding codomain
            for each input value. See the defuzz_mamdani method for
            converting a domain and codomain(s) into scalar inputs.
        """
        # Evaluate entire ruleset for all consequent memberships
        membership_outputs = self.eval_membership(x)

        # Convert to Mamdani output functions and return
        return self.convert_to_mamdani(membership_outputs, num_points)

    def convert_to_mamdani(self, memberships: Dict[str, Dict[str, ArrayLike]],
                           num_points: int = 100) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Converts membership values to fuzzified Mamdani output.

        Args:
            memberships: Dictionary with output group names as keys and values
                that are dictionaries. These have membership function names as
                keys and memberships as values.
            num_points: Number of points to evaluate when building output.

        Returns:
            Dictionary with output group names as keys and values that are
            dictionaries with membership function names as keys and an array
            with arrays of memberships as for each input value.
        """
        outputs = dict()

        # For each output group
        for group_name in memberships:
            # Get output group and domain for evaluation
            group = self.groupset[group_name]
            domain = np.linspace(group.domain[0], group.domain[1], num_points)

            # Initialize membership of Mamdani output to zero
            codomain = 0.0

            # For each output function's calculated membership
            for fn_name, fn_alpha in memberships[group_name].items():
                # Get output membership function values for each point in domain
                fn_vals = group[fn_name](domain)

                # Take whichever is less, function membership value or membership
                vals = np.vectorize(lambda x: np.minimum(fn_vals, x),
                                    signature="()->(n)")(fn_alpha)

                # Take max of output membership so far
                codomain = np.maximum(codomain, vals)

            # Save output domain and codomain
            outputs[group_name] = (domain, np.squeeze(codomain))

        return outputs

    @staticmethod
    def defuzz_mamdani(domain: np.ndarray, codomain: np.ndarray) -> ArrayLike:
        """Defuzzifies Mamdani output and returns scalar value(s).

        Args:
            domain: Domain of Mamdani output.
            codomain: Output memberships corresponding to domain.

        Returns:
            Defuzzified output(s).
        """
        top = codomain * domain
        top = np.sum(np.atleast_2d(top), axis=-1)
        bot = np.sum(np.atleast_2d(codomain), axis=-1)

        return top / bot

    @staticmethod
    def plot_mamdani(domain: np.ndarray, codomain: np.ndarray):
        """Plots output of Mamdani inference.

        Can be used in conjunction with an output membership function group's
        plot method to visualize both the group and inference output.

        Args:
            domain: Domain values of output as returned by Mamdani eval.
            codomain: Output membership values as returned by Mamdani eval.
        """
        # Plot line
        plt.plot(domain, codomain, ls=":", color="black")

        # Plot shading below function
        plt.fill_between(domain, codomain, color="grey", alpha=0.2)

        # Plot line indicating defuzzified output
        defuzzed = FIS.defuzz_mamdani(domain, codomain)
        plt.axvline(defuzzed, ls=":", color="black", ymin=0.0, ymax=0.95)

    # --------------------------
    # Takagi-Sugeno-Kang Methods
    # --------------------------

    def eval_tsk(self, x: Mapping[str, ArrayLike]) -> Dict[str, ArrayLike]:
        """Evaluates input and returns de-fuzzified Takagi Sugeno Kang output.

        Args:
            x: Input values as dictionary with input groups names as keys and
                a scalar or array-like as values. Also supports other data
                structures similarly accessible by input group name.

        Returns:
            Dictionary with output group names as keys and de-fuzzified outputs
            for each input.
        """
        # Evaluate entire ruleset for all consequent memberships
        membership_outputs = self.eval_membership(x)

        # Convert to Mamdani output functions and return
        return self.convert_to_tsk(membership_outputs)

    def convert_to_tsk(self, memberships: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Converts membership values to defuzzified Takagi-Sugeno output.

        Args:
            memberships: Dictionary with output group names as keys and values
                that are dictionaries. These have membership function names as
                keys and memberships as values.

        Returns:
            Dictionary with output group names as keys and memberships as values.
        """
        outputs_tsk = dict()

        # For each output group
        for group_name, group_memberships in memberships.items():
            top, bot = 0, 0

            # Calc top and bottom for each membership function
            for fn_name, fn_memberships in group_memberships.items():
                # Calculate TSK output value
                fn = self.groupset[group_name][fn_name]
                tsk_val = fn.center

                top += fn_memberships * tsk_val
                bot += fn_memberships

            # Save tsk output for membership group as float
            try:
                outputs_tsk[group_name] = top / bot
            except ZeroDivisionError:
                err_str = "Output memberships in TSK evaluation " \
                          f"sum to zero for group '{group_name}'."
                raise ZeroDivisionError(err_str)

        return outputs_tsk

    @staticmethod
    def plot_tsk(tsk_output: float):
        """Plots output of Takagi-Sugeno-Kang inference as a vertical line.

        Can be used in conjunction with an output membership function group's
        plot method to visualize both the group and inference output.

        Args:
            tsk_output: Single TSK output value.
        """
        plt.axvline(tsk_output, ls=":", color="black", ymax=0.95)

    # --------------
    # Helper Methods
    # --------------
