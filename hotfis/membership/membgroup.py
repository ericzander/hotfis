"""Contains membership function group (MFGroup) definition.

MFGroups serve as named collections of membership functions and has a method
for plotting.

An example of a group could be 'temperature', comprised of membership functions
such as 'cold', 'warm', and 'hot'.
"""

from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from hotfis import MembFunc


class MembGroup:
    """A collection of membership functions corresponding to fuzzy sets.

    Args:
        name: The name of the group.
        xmin: Smallest group domain value used in Mamdani evaluation and visualization.
        xmax: Largest group domain value used in Mamdani evaluation and visualization.
        fns: A list of MembFuncs stored in the group.

    Attributes:
        fns (Dict[MembFunc]): Dictionary of MembFuncs stored in the group.
            Their names are keys and the objects themselves are values.
        name (str): The name of the group.
        domain (Tuple[float, float]): Domain for Mamdani evaluation and visualization.
    """
    # -----------
    # Constructor
    # -----------

    def __init__(self, name: str, xmin: float, xmax: float, fns: List[MembFunc]):
        # Save group name and functions
        self.name = name
        self.fns = {fn.name: fn for fn in fns}

        # Get domain range used in Mamdani evaluation
        self.domain = (xmin, xmax)

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

    def plot(self, start: Optional[float] = None, stop: Optional[float] = None,
             num_points: int = 500, stagger_labels: bool = False,
             line_color: str = "black", fill_alpha=0.1, **plt_kwargs):
        """Plots every function in the group in a new figure.

        Args:
            start: Specified start of plot domain.
                Defaults to group domain start if None is passed.
            stop: Specified end of plot domain.
                Defaults to group domain end if None is passed.
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

        # Prepare to save xticks
        xticks = dict()

        all_tsk = all([fn.fn_type == "tsk" for fn in self])

        # Create domain based on given parameters or group's domain
        if start is not None and stop is not None:
            domain = np.linspace(start, stop, num_points)
        else:
            domain = np.linspace(self.domain[0], self.domain[1], num_points)

        # For each function, plot and update x-ticks
        for fn in self:
            # Plot function if not TSK
            if fn.fn_type != "tsk":
                codomain = fn(domain)
                ax1.plot(domain, codomain, color=line_color, **plt_kwargs)
                ax1.fill_between(domain, codomain, alpha=fill_alpha)

                # Update function x-tick labels
                xtick_val = fn.center
                if xtick_val not in xticks.values():
                    xticks[fn.name] = xtick_val
                else:
                    key = [k for k, v in xticks.items() if v == xtick_val][0]
                    xticks[f"{key}/{fn.name}"] = xtick_val
                    del xticks[key]

            # TSK functions
            else:
                plt.axvline(fn.center, color=line_color, ymax=0.95, **plt_kwargs)
                xticks[fn.name] = fn.center

        # Finalize x and y limits and x-ticks
        ax1.set_ylim(0.0, 1.05)
        if not all_tsk:
            ax1.margins(0.0, x=True)
        ax2.set_xticks(list(xticks.values()))
        ax2.set_xticklabels(list(xticks.keys()), fontsize=8)
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_ylim(0.0, 1.05)

        # Stagger function name labels if requested
        if stagger_labels:
            for tick in ax2.xaxis.get_major_ticks()[1::2]:
                tick.set_pad(15)

        # Decorate
        plt.title(self.name, pad=16)
        ax1.grid(visible=True, axis="y", alpha=0.5, ls="--")
