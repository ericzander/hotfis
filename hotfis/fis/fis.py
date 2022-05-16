"""Contains Fuzzy Inference System (FIS) definition.
"""

from typing import Union

from hotfis import MFGroupset, FuzzyRuleset


class FIS:
    """Fuzzy inference system (FIS) comprised of membership functions and a ruleset.


    """
    # -----------
    # Constructor
    # -----------

    def __init__(self, mf_groupset: Union[str, MFGroupset],
                 fuzzy_ruleset: Union[str, FuzzyRuleset]):
        """FIS constructor.



        Args:
            mf_groupset:
            fuzzy_ruleset:
        """
        # Save or create membership function groupset and ruleset
        self.groupset = MFGroupset(mf_groupset)
        self.ruleset = FuzzyRuleset(fuzzy_ruleset)
