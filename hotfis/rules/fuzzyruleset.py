"""Contains FuzzyRuleset definition.
"""

from typing import List, Union, Set

import os

from hotfis import FuzzyRule


class FuzzyRuleset:
    """Ruleset to be used in fuzzy inference system evaluation.

    FuzzyRulesets are collections of rules that may be evaluated in membership
    inference. FuzzyRulesets can be defined directly with a List of str/FuzzyRules
    or by passing the path to text file with rules on separate lines.

    Args:
        source: Filepath to file with rules or list of rules as rules or FuzzyRules.

    Attributes:
        rules (List[FuzzyRule]): FuzzyRules comprising the ruleset.
        input_names (Set[str]): Names of required inputs (ex. {"food", "service"})
        output_names (Set[str]): Names of ruleset outputs (ex. {"tip"})

    Example:
        Method 1:

        >>> ruleset1 = FuzzyRuleset([
        >>>     "if temperature is cold then heater is on",
        >>>     "if temperature is warm then heater is medium",
        >>>     "if temperature is hot then heater is off",
        >>> ])

        Method 2::

            === example_rules.txt ===

            if service is poor or food is rancid then tip is cheap
            if service is good then tip is average
            if service is excellent and food is delicioius then tip is generous

        >>> ruleset2 = FuzzyRuleset("example_rules.txt")
    """
    # -----------
    # Constructor
    # -----------

    def __init__(self, source: Union[str, List[Union[str, FuzzyRule]]]):
        # Read rules
        if isinstance(source, str):
            self._read_rules(source)
        else:
            self.rules = [FuzzyRule(rule) if isinstance(rule, str) else rule for rule in source]

        # Save a list of required input names for convenience
        self.input_names: Set[str] = self.get_input_names()
        self.output_names: Set[str] = self.get_outputs_name()

    # -------
    # Methods
    # -------

    def __iter__(self):
        """Supports iteration through rules.
        """
        return iter(self.rules)

    def get_input_names(self) -> Set[str]:
        """Returns a set of required input names.
        """
        input_names = set()

        for rule in self.rules:
            for ante in rule.antecedents:
                input_names.add(ante.group_name)

        return input_names

    def get_outputs_name(self) -> Set[str]:
        """Returns a set of output names.
        """
        output_names = set()

        for rule in self.rules:
            output_names.add(rule.consequent.group_name)

        return output_names

    # ---------------
    # Reading Methods
    # ---------------

    def _read_rules(self, filepath: str):
        """Parses and saves all rules from file at given path.

        Parameters:
            filepath: Relative path of file with rules to read
        """
        full_path = os.path.join(os.getcwd(), filepath)

        with open(full_path) as file:
            rules_text = [line.rstrip() for line in file if line]
            self.rules = [FuzzyRule(rule) for rule in rules_text]
