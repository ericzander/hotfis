"""Contains Fuzzy Inference System (FIS) definition.

TODO Mamdani approximation is experimental and needs further doc/testing
"""

from __future__ import annotations  # Doc aliases
from typing import Union, Mapping, Dict, Tuple, List
from numpy.typing import ArrayLike

import itertools

import numpy as np
import matplotlib.pyplot as plt

from hotfis import FuzzyFunc, FuzzyGroup, FuzzyGroupset, FuzzyRule, FuzzyRuleset


class FIS:
    """Fuzzy inference system (FIS).

    A FIS is comprised of a FuzzyGroupset containing membership functions and
    a FuzzyRuleset containing rules referring to said membership functions.

    With compatible membership functions and rules, a FIS is capable of
    determining memberships, fuzzified Mamdani output, and defuzzified
    Takagi-Sugeno-Kang (TSK) output.

    The FIS class also includes methods for defuzzifying Mamdani output,
    plotting Mamdani/TSK output, converting calculated memberships to
    Mamdani/TSK output, and other utilities.

    Args:
        groupset: Groupset of membership functions or path to file with groups.
        ruleset: Ruleset to be evaluated or path to file with rules.

    Attributes:
        groupset (FuzzyGroupset): Groupset of membership functions required for evaluation.
        ruleset (FuzzyRuleset): Ruleset to be evaluated.

    Example:
        Method 1:

        >>> fis1 = FIS(
        >>>     # Create membership functions
        >>>     FuzzyGroupset([
        >>>         # Create temperature group for input
        >>>         FuzzyGroup("temperature", 30, 70, [
        >>>             FuzzyFunc("cold", [30, 40], "leftedge"),
        >>>             FuzzyFunc("warm", [30, 40, 60, 70], "trapezoidal"),
        >>>             FuzzyFunc("hot", [60, 70], "rightedge")
        >>>         ]),
        >>>
        >>>         # Create heater group for output
        >>>         FuzzyGroup("heater", 0.0, 1.0, [
        >>>             FuzzyFunc("off", [0.1, 0.2], "leftedge"),
        >>>             FuzzyFunc("medium", [0.1, 0.2, 0.8, 0.9], "trapezoidal"),
        >>>             FuzzyFunc("on", [0.8, 0.9], "rightedge")
        >>>         ]),
        >>>     ]),
        >>>
        >>>     # Create ruleset
        >>>     FuzzyRuleset([
        >>>         "if temperature is cold then heater is on",
        >>>         "if temperature is warm then heater is medium",
        >>>         "if temperature is hot then heater is off",
        >>>     ])
        >>> )

        Method 2::

            === example_groups.txt ===

            group temperature
            leftedge cold 30 40
            trapezoidal warm 30 40 60 70
            rightedge hot 60 70
            domain 30 70

            group heater
            leftedge off 0.1 0.2
            trapezoidal medium 0.1 0.2 0.8 0.9
            rightedge on 0.8 0.9
            domain 0.0 1.0

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

    def __init__(self, groupset: Union[str, FuzzyGroupset],
                 ruleset: Union[str, FuzzyRuleset]):
        # Save or create membership function groupset and ruleset
        self.groupset = FuzzyGroupset(groupset) if isinstance(groupset, str) else groupset
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

        # Evaluate for outputs
        for rule in self.ruleset:
            # Get membership function names and corresponding membership values
            group, fn, output = rule.evaluate(x, self.groupset)

            # Create new group dictionary if not already created
            if group not in all_outputs:
                all_outputs[group] = dict()

            # Save consequent membership (add if not first instance of consequent)
            if fn not in all_outputs[group]:
                all_outputs[group][fn] = output
            else:
                all_outputs[group][fn] += output

        # Ensure all memberships add to 1.0
        for group in all_outputs:
            # Find sum of all output memberships
            sums = np.sum([all_outputs[group][fn] for fn in all_outputs[group]], axis=0)

            # Divide individual output memberships by the sum
            for fn in all_outputs[group]:
                all_outputs[group][fn] = all_outputs[group][fn] / sums

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
            tuples with two arrays. The first array is the domain of the
            output group. The second is a corresponding codomain with values
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
                # Get output function memberships for each point in domain
                fn_vals = group[fn_name](domain)

                # Take whichever is less, function value or membership
                vals = np.vectorize(lambda x: np.minimum(fn_vals, x),
                                    signature="()->(n)")(fn_alpha)

                # Take max of output membership so far
                codomain = np.maximum(codomain, vals)

            # Save output domain and codomain
            outputs[group_name] = (domain, np.squeeze(codomain))

        return outputs

    @staticmethod
    def defuzz_mamdani(mamdani_output: Tuple[np.ndarray, np.ndarray]) -> ArrayLike:
        """Defuzzifies Mamdani output and returns scalar value(s).

        Args:
            mamdani_output: Tuple in dictionary returned by eval_mamdani with
                the domain as the first element and one or more codomains
                as the second element.

        Returns:
            Defuzzified output(s) for each codomain in codomains.
        """
        domain, codomains = mamdani_output

        # Calc center of mass of fuzzified output(s)
        top = codomains * domain
        top = np.sum(np.atleast_2d(top), axis=-1)
        bot = np.sum(np.atleast_2d(codomains), axis=-1)

        # Calc final output and collapse if one element
        output = top / bot
        output = output.item() if output.size == 1 else output

        return output

    @staticmethod
    def plot_mamdani(domain: np.ndarray, codomain: np.ndarray, defuzz=True):
        """Plots output of Mamdani inference.

        Can be used in conjunction with an output membership function group's
        plot method to visualize both the group and inference output.

        Args:
            domain: Domain values of output as returned by Mamdani eval.
            codomain: Output membership values as returned by Mamdani eval.
            defuzz: Indicates of output should be defuzzified and plotted.
        """
        # Plot line
        plt.plot(domain, codomain, ls=":", color="black")

        # Plot shading below function
        plt.fill_between(domain, codomain, color="grey", alpha=0.2)

        # Plot line indicating defuzzified output
        if defuzz:
            defuzzed = FIS.defuzz_mamdani((domain, codomain))
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

    # ---------------------
    # Mamdani Approximation
    # ---------------------

    _supported_types = {
        "triangular",
        "trapezoidal",
        "leftedge",
        "rightedge"
    }

    def approximate_mamdani(self) -> FIS:
        """Creates FIS with output functions approximated through TSK evaluation.

        EXPERIMENTAL

        Returns:
            A new FIS with approximated Mamdani output functions.
        """
        # Create a dictionary of named approximated function groups
        group_funcs = self._approx_groups()

        # Create groupset with approximated functions as outputs
        approx_groupset = self._create_approx_groupset(group_funcs)

        # Create new FIS with new groupset
        approx_fis = FIS(approx_groupset, self.ruleset)

        return approx_fis

    # Approximation Helpers

    def _approx_groups(self) -> Dict[str, List[FuzzyFunc]]:
        """Approximates Mamdani output membership function groups given a FIS.

        Returns:
            A dictionary with group names as keys and a list of membership
            functions as values.
        """
        group_funcs = dict()

        for rule in self.ruleset:
            # Approximate the output function for a single rule
            group_name, memb_func = self.__approx_fn(rule)

            # Save function in group output
            if group_name not in group_funcs:
                group_funcs[group_name] = []
            group_funcs[group_name].append(memb_func)

        return group_funcs

    def __approx_fn(self, rule: FuzzyRule) -> Tuple[str, FuzzyFunc]:
        # Save left, center, and middle antecedent function values
        all_params = self.__get_antecedent_params(rule)

        # Get combos of antecedent inputs and get average outputs for output fuzzy set
        #   a = avg(rule outputs for left and center antecedent params as inputs)
        #   b = rule output for center antecedent params as inputs
        #   c = avg(rule outputs for center and right antecedent params as inputs)
        outputs = []
        for start, end in zip((0, 1, 1), (2, 2, 3)):
            # Get combinations of relevant antecedent inputs as named columns
            params = [param[start:end] for param in all_params.values()]
            combinations = np.array(list(itertools.product(*params)))
            inputs = {group_name: combinations[:, i] for i, group_name in enumerate(all_params)}

            # Evaluate the rule with each combination and save the avg result
            output_vals = self.eval_tsk(inputs)[rule.consequent.group_name]
            outputs.append(np.mean(output_vals))

        # Sort the parameters and construct the approximated output function
        final_params = np.sort(outputs)
        approx_fn = self.__create_approx_fn(rule.consequent.fn_name, final_params)

        return rule.consequent.group_name, approx_fn

    def __get_antecedent_params(self, rule) -> Dict[str, np.ndarray]:
        params = dict()

        # Save key antecedent params for relevant rule
        for ant in rule.antecedents:
            fn = self.groupset[ant.group_name][ant.fn_name]

            if fn.fn_type not in self._supported_types:
                err_str = f"Antecedent function '{fn.name}' does not support " \
                          f"Mamdani approximation."
                raise ValueError(err_str)

            params[ant.group_name] = np.array([fn.params[0], fn.center, fn.params[-1]])

        # Save params for all antecedent groups in ruleset not already addressed
        aux_params = dict()
        for aux_rule in self.ruleset:
            if aux_rule is rule:
                continue

            for ant in aux_rule.antecedents:
                fn = self.groupset[ant.group_name][ant.fn_name]
                fn_params = np.array([fn.params[0], fn.center, fn.params[-1]])

                if ant.group_name not in params:
                    if ant.group_name not in aux_params:
                        aux_params[ant.group_name] = fn_params
                    else:
                        aux_params[ant.group_name] = np.mean(
                            [aux_params[ant.group_name], fn_params], axis=0
                        )

        aux_params.update(params)
        return aux_params

    @staticmethod
    def __create_approx_fn(fn_name, params):
        if params[0] == params[1]:
            fn_type = "leftedge"
        elif params[1] == params[2]:
            fn_type = "rightedge"
        else:
            fn_type = "triangular"

        params = np.unique(params)

        return FuzzyFunc(fn_name, params, fn_type)

    # Wrap-up Helpers

    def _create_approx_groupset(self: FIS,
                                group_funcs: Dict[str, List[FuzzyFunc]]) -> FuzzyGroupset:
        approx_groupset = self.groupset.copy()

        # For each output group name
        for gname in group_funcs:
            # Update new group domain
            xmin, xmax = float("inf"), -float("inf")
            for fn in group_funcs[gname]:
                xmin = min(xmin, np.min(fn.params))
                xmax = max(xmax, np.max(fn.params))

            # Create resulting group and save in output groupset
            approx_group = FuzzyGroup(gname, xmin, xmax, group_funcs[gname])
            approx_groupset[gname] = approx_group

        return approx_groupset
