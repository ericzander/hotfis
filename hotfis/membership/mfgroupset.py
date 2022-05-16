"""Contains membership function groupset (MFGroupset) definition.

MFGroupsets are collections of groups of membership functions to streamline
membership inference system evaluation.

For example, an MFGroupset could contain a 'temperature' group and 'heater'
group to be used in a membership inference system designed for setting heat based on
temperature.
"""

from typing import List, Dict, Union

import os

from hotfis import MFGroup, MembFunc


class MFGroupset:
    """Collection of membership function groups for FIS evaluation.

    Membership function groupsets are collections of membership function groups
    used for streamlined membership inference system evaluation.

    Attributes:
        groups (Dict[str, MFGroup]): Dictionary of named groups.
    """
    # -----------
    # Constructor
    # -----------

    def __init__(self, source: Union[str, List[MFGroup]]):
        """Membership function groupset constructor.

        Groupsets can be defined directly with a List of named membership
        function groups (MFGroups) or by passing the path to text file
        formatted with template names and parameters:

        Args:
            source: Path to file with membership function groups or list of MFGroups.

        Example input file:

            |  **example_groups.txt**

            |  *group temperature*
            |  *leftedge cold 30 40*
            |  *trapezoidal warm 30 40 60 70*
            |  *rightedge hot 60 70*

            |  *group heater*
            |  *leftedge off 0.1 0.2*
            |  *trapezoidal medium 0.1 0.2 0.8 0.9*
            |  *rightedge on 0.8 0.9*

        Example construction:
            |  *gset1 = MFGroupset("example_groups.txt")*
            |  *gset2 = MFGroupset([group1, group2])*
        """
        self.groups: Dict[str, MFGroup] = dict()

        # Read groups from given file or list of groups
        if isinstance(source, str):
            self._read_groups(source)
        else:
            self.groups = {group.name: group for group in source}

    # -------
    # Methods
    # -------

    def __getitem__(self, group_name) -> MFGroup:
        """Supports subscripting with group name.

        Args:
            group_name: Name of the group to retrieve.

        Example:
            group = example_groupset["group_name"]
        """
        return self.groups[group_name]

    def __setitem__(self, group: MFGroup):
        """Supports group assignment with subscripting.

        Will overwrite a group if it already exists.

        Args:
            group: Group to save in groupset.

        Example:
            example_groupset["group_name"] = new_group
        """
        self.groups[group.name] = group

    def __iter__(self):
        """Can iterate through each group.

        Example:
            for group in example_groupset:
                ...
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
                self.groups[name] = MFGroup(functions, name)
                name = None
                functions = []
            elif line[0] == "group":
                name = line[1]
            elif line[0] in MembFunc.templates:
                fn = self.__read_function(line)
                functions.append(fn)
            else:
                raise ValueError(f"Unreadable line: {line}")

        # Wrap-up
        if name and functions:
            self.groups[name] = MFGroup(functions, name)

    @staticmethod
    def __read_function(line: List[str]) -> MembFunc:
        """Helper function for reading a single membership function
        """
        fn_type = line[0]
        fn_name = line[1]

        # Pass function type, name, and parameters as floats to create_mf
        return MembFunc([float(x) for x in line[2:]], fn_type, fn_name)
