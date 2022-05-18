"""Contains Fuzzy Inference System (FIS) definition.
"""

from typing import Union, Dict

from hotfis import MembGroupset, FuzzyRuleset


class FIS:
    """Fuzzy inference system (FIS) comprised of membership functions and a ruleset.

    Args:
        groupset:
        ruleset:

    Attributes:
        groupset (MembGroupset):
        ruleset (FuzzyRuleset):
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

    def eval_membership(self, x: Dict[str, float]):
        pass
