"""Contains membership function group (MFGroup) definition.

MFGroups serve as named collections of membership functions and has a method
for plotting.

An example of a group could be 'temperature', comprised of membership functions
such as 'cold', 'warm', and 'hot'.
"""

from typing import List

import numpy as np
import matplotlib.pyplot as plt

from hotfis import MembFunc


class MFGroup:
    """A collection of membership functions corresponding to fuzzy sets.

    Attributes:
        fns (Dict[MembFunc]): Dictionary of MembFuncs stored in the group.
            Their names are keys and the objects themselves are values.
        name (str): The name of the group. If not given, uses a generic name.
    """
    # ----------
    # Attributes
    # ----------

    # Counter used for naming groups when names are not supplied
    _group_count = 0

    # -----------
    # Constructor
    # -----------

    def __init__(self, fns: List[MembFunc], name: str = ""):
        """Membership function group constructor.

        Supports plotting of each function on a created figure.

        Args:
            fns: A list of MembFuncs stored in the group.
            name: The name of the group. If not given, uses a generic name.
        """
        self.fns = {fn.name: fn for fn in fns}

        # Save group name
        if not name:
            self.name = f"group{MFGroup._group_count}"
            MFGroup._group_count += 1
        else:
            self.name = name

    # -------
    # Methods
    # -------

    def __getitem__(self, fn_name) -> MembFunc:
        """Supports subscripting with membership function name.

        Args:
            fn_name: Name of the function to retrieve.

        Example:
            fn = example_group["fn_name"]
        """
        return self.fns[fn_name]

    def __setitem__(self, fn: MembFunc):
        """Supports function assignment with subscripting.

        Will overwrite a function if it already exists.

        Args:
            fn: Function to save in group.

        Example:
            example_group["fn_name"] = new_fn
        """
        self.fns[fn.name] = fn

    def __iter__(self):
        """Can iterate through each membership function.

        Example:
            for fn in example_group:
                ...
        """
        return iter(self.fns.values())

    def keys(self):
        """Returns the names of each contained membership function.
        """
        return self.fns.keys()

    def items(self):
        """Returns each function name and object.
        """
        return self.fns.items()

    def values(self):
        """Returns each membership function.
        """
        return self.fns.values()

    def plot(self, start: float, stop: float, num: int = 500,
             stagger_labels: bool = False, color: str = "black",
             **plt_kwargs):
        """Plots every function in the group in a new figure.

        Args:
            start: Domain start.
            stop: Domain end.
            num: Number of points to plot for each function.
            stagger_labels: Whether to stagger function label names on top.
            color: matplotlib.pyplot color of the line representing the function.
            **plt_kwargs: matplotlib.pyplot plotting options.
        """
        # Create figure twin x axes (top one for function names)
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()

        # Create domain and xticks
        domain = np.linspace(start, stop, num)
        xlim = (start, stop)
        xticks = dict()

        # For each function, plot and update xticks
        for fn in self:
            # Plot function if not TSK
            try:
                ax1.plot(domain, fn(domain), color=color, **plt_kwargs)

                # Update function xtick labels
                xtick_val = fn.center
                if xtick_val not in xticks.values():
                    xticks[fn.name] = xtick_val
                else:
                    key = [k for k, v in xticks.items() if v == xtick_val][0]
                    xticks[f"{key}/{fn.name}"] = xtick_val
                    del xticks[key]

            # TSK center values
            except NotImplementedError:
                ax1.axvline(fn.center, **plt_kwargs)
                xticks[fn.name] = fn.center

        # Finalize x and y limits and xticks
        ax1.set_xlim(*xlim)
        ax1.set_ylim(0.01, 1.05)
        ax2.set_xticks(list(xticks.values()))
        ax2.set_xticklabels(list(xticks.keys()), fontsize=8)
        ax2.set_xlim(*xlim)
        ax2.set_ylim(-0.05, 1.05)

        # Stagger function name labels if requested
        if stagger_labels:
            for tick in ax2.xaxis.get_major_ticks()[1::2]:
                tick.set_pad(15)

        # Decorate
        plt.title(self.name, pad=16)
        ax1.grid(visible=True, axis="y", alpha=0.5, ls="--")
