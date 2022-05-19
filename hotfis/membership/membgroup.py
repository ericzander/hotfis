"""Contains membership function group (MFGroup) definition.

MFGroups serve as named collections of membership functions and has a method
for plotting.

An example of a group could be 'temperature', comprised of membership functions
such as 'cold', 'warm', and 'hot'.
"""

from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from hotfis import MembFunc


class MembGroup:
    """A collection of membership functions corresponding to fuzzy sets.

    Args:
        fns: A list of MembFuncs stored in the group.
        name: The name of the group.

    Attributes:
        fns (Dict[MembFunc]): Dictionary of MembFuncs stored in the group.
            Their names are keys and the objects themselves are values.
        name (str): The name of the group.
    """
    # -----------
    # Constructor
    # -----------

    def __init__(self, name: str, fns: List[MembFunc]):
        # Save group name and functions
        self.name = name
        self.fns = {fn.name: fn for fn in fns}

        # Get domain range used in Mamdani evaluation
        self.domain = self.get_domain()

    # -------
    # Methods
    # -------

    def __getitem__(self, fn_name) -> MembFunc:
        """Supports subscripting with membership function name.

        Args:
            fn_name: Name of the function to retrieve.
        """
        return self.fns[fn_name]

    def __setitem__(self, fn: MembFunc):
        """Supports function assignment with subscripting.

        Will overwrite a function if it already exists.

        Args:
            fn: Function to save in group.
        """
        self.fns[fn.name] = fn

    def __iter__(self):
        """Can iterate through each membership function.
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

    def get_domain(self) -> Tuple[float, float]:
        """Finds and returns tuple with domain boundaries for membership.

        Returns:
            Minimum and maximum domain values respectively.
        """
        min_val = float("inf")
        max_val = float("-inf")

        for fn in self:
            min_val = min(min_val, fn.domain[0])
            max_val = max(max_val, fn.domain[1])

        return min_val, max_val

    def plot(self, num_points: int = 500, stagger_labels: bool = False,
             line_color: str = "black", fill_alpha=0.1, **plt_kwargs):
        """Plots every function in the group in a new figure.

        Args:
            num_points: Number of points to plot for each function.
            stagger_labels: Whether to stagger function label names on top.
            line_color: matplotlib.pyplot color of the line representing the function.
            fill_alpha: Alpha of function color. Set to 0.0 for no fill.
            **plt_kwargs: matplotlib.pyplot plotting options.
        """
        # Create figure twin x axes (top one for function names)
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()

        # Create domain and xticks
        domain = np.linspace(self.domain[0], self.domain[1], num_points)
        xlim = self.domain
        xticks = dict()

        # For each function, plot and update xticks
        for fn in self:
            # Plot function if not TSK
            try:
                codomain = fn(domain)
                ax1.plot(domain, codomain, color=line_color, **plt_kwargs)
                ax1.fill_between(domain, codomain, alpha=fill_alpha)

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
