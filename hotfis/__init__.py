"""Core HotFIS module.
"""

# Autodoc nested type alias rendering
from sphinx.util import inspect
inspect.TypeAliasForwardRef.__repr__ = lambda self: self.name
inspect.TypeAliasForwardRef.__hash__ = lambda self: hash(self.name)

# Membership function objects
from .membership.membfunc import MembFunc
from .membership.membgroup import MembGroup
from .membership.membgroupset import MembGroupset

# Rule objects
from .rules.fuzzyrule import FuzzyRule
from .rules.fuzzyruleset import FuzzyRuleset

# Fuzzy inference system (FIS)
from .fis.fis import FIS

# Fuzzy network of FIS
from .network.fuzzynetwork import FuzzyNetwork
