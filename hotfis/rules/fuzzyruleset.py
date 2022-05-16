"""Contains FuzzyRuleset definition.
"""

from typing import List, Union, Set

import os

from .fuzzyrule import FuzzyRule


class FuzzyRuleset:
    """Ruleset to be used in fuzzy inference system evaluation.

    FuzzyRulesets are collections of rules that may be evaluated in fuzzy
    inference.

    Attributes:
        rules (List[FuzzyRule]): FuzzyRules comprising the ruleset.
        input_names (Set[str]): Names of required inputs (ex. {"food", "service"})
        output_names (Set[str]): Names of ruleset outputs (ex. {"tip"})
    """
    # -----------
    # Constructor
    # -----------

    def __init__(self, source: Union[str, List[FuzzyRule]]):
        """Fuzzy ruleset constructor.

        FuzzyRulesets can be defined directly with a List of FuzzyRules or by
        passing the path to text file with rules on separate lines.

        Args:
            source: Filepath to file with rules or list of FuzzyRules.

        Example input file:

            |  **rules.txt**

            |  *if service is poor or food is rancid then tip is cheap*
            |  *if service is good then tip is average*
            |  *if service is excellent and food is delicioius then tip is generous*

        Example construction:
            |  *rset1 = FuzzyRuleset("rules.txt")*
            |  *rset2 = FuzzyRuleset([rule1, rule2])*
        """
        # Read rules
        if isinstance(source, str):
            self._read_rules(source)
        else:
            self.rules = [rule for rule in source]

        # Save a list of required input names for convenience
        self.input_names: Set[str] = self.get_input_names()
        self.output_names: Set[str] = self.get_outputs_name()

    # -------
    # Methods
    # -------

    def get_input_names(self) -> Set[str]:
        """
        Returns a set of required input names.
        """
        input_names = set()

        for rule in self.rules:
            for ante in rule.antecedents:
                input_names.add(ante.mf_group_name)

        return input_names

    def get_outputs_name(self) -> Set[str]:
        """
        Returns a set of output names.
        """
        output_names = set()

        for rule in self.rules:
            output_names.add(rule.consequent.mf_group_name)

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
