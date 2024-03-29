"""Contains membership function groupset (MFGroupset) definition.

MFGroupsets are collections of groups of membership functions to streamline
membership inference system evaluation.

For example, an MFGroupset could contain a 'temperature' group and 'heater'
group to be used in a membership inference system designed for setting heat based on
temperature.
"""

from typing import List, Dict, Union

import os
import copy

from hotfis import FuzzyGroup, FuzzyFunc


class FuzzyGroupset:
    """A collection of membership function groups for use in FIS evaluation.

    Membership function groupsets are collections of membership function groups
    used for streamlined membership inference system evaluation.

    Groupsets can be defined directly with a List of named membership function
    groups. Alternatively, they may be defined by passing the path to text file
    formatted with template names and parameters.

    Args:
        source: Path to file with membership function groups or MembGroups.

    Attributes:
        groups (Dict[str, FuzzyGroup]): Dictionary of named groups.

    Example:
        Method 1:

        >>> groupset1 = FuzzyGroupset([
        >>>     FuzzyGroup("temperature",  30, 70, [
        >>>         FuzzyFunc("cold", [30, 40], "leftedge", ),
        >>>         FuzzyFunc("warm", [30, 40, 60, 70], "trapezoidal"),
        >>>         FuzzyFunc("hot", [60, 70], "rightedge")
        >>>     ]),
        >>>     FuzzyGroup("heater", 0.0, 1.0, [
        >>>         FuzzyFunc("off", [0.1, 0.2], "leftedge"),
        >>>         FuzzyFunc("medium", [0.1, 0.2, 0.8, 0.9], "trapezoidal"),
        >>>         FuzzyFunc("on", [0.8, 0.9], "rightedge")
        >>>     ]),
        >>> ])

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

        >>> groupset2 = FuzzyGroupset("example_groups.txt")
    """
    # -----------
    # Constructor
    # -----------

    def __init__(self, source: Union[str, List[FuzzyGroup]]):
        self.groups: Dict[str, FuzzyGroup] = dict()

        # Read groups from given file or list of groups
        if isinstance(source, str):
            self._read_groups(source)
        else:
            self.groups = {group.name: group for group in source}

    # -------
    # Methods
    # -------

    def __getitem__(self, group_name) -> FuzzyGroup:
        """Supports subscripting with group name.

        Args:
            group_name: Name of the group to retrieve.
        """
        return self.groups[group_name]

    def __setitem__(self, group_name: str, group: FuzzyGroup):
        """Supports group assignment with subscripting.

        Will overwrite a group if it already exists.

        Args:
            group_name: The group name.
            group: Group to save in groupset.
        """
        self.groups[group_name] = group

    def __iter__(self):
        """Can iterate through each group.
        """
        return iter(self.groups.values())

    def keys(self):
        """Returns the names of each contained group.
        """
        return self.groups.keys()

    def items(self):
        """Returns each group name and object.
        """
        return self.groups.items()

    def values(self):
        """Returns each group.
        """
        return self.groups.values()

    def copy(self):
        """Returns deep copy of groupset
        """
        return copy.deepcopy(self)

    # ---------------
    # Reading Methods
    # ---------------

    def _read_groups(self, filepath):
        """Reads membership function groups from the given file.

        The path may be relative to the inciting script.
        """
        # Get path of original caller and append given filepath
        full_path = os.path.join(os.getcwd(), filepath)
        with open(full_path) as file:
            lines = file.readlines()

        # Initialize group name and function containers
        name = None
        functions = []

        for line in lines:
            line = line.split()

            # Evaluate line
            if not line:
                continue
            elif line[0] == "group":
                name = line[1]
            elif line[0] in FuzzyFunc.templates:
                fn = self.__read_function(line)
                functions.append(fn)
            elif line[0] == "domain":
                domain = (float(line[1]), float(line[2]))
                self.groups[name] = FuzzyGroup(name, domain[0], domain[1], functions)
                name = None
                functions = []
            else:
                raise ValueError(f"Unreadable line: {line}")

    @staticmethod
    def __read_function(line: List[str]) -> FuzzyFunc:
        """Helper function for reading a single membership function
        """
        fn_type = line[0]
        fn_name = line[1]

        # Pass function type, name, and parameters as floats to create_mf
        return FuzzyFunc(fn_name, [float(x) for x in line[2:]], fn_type)
