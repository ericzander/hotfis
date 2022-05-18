"""Contains membership function definition.

Membership functions calculate membership to a fuzzy set that is implicitly
defined as the function is created.

For example, membership functions for temperature could be 'cold', 'warm',
and 'hot'.
"""

from __future__ import annotations  # Doc aliases
from typing import Iterable, Union
from numpy.typing import ArrayLike

import numpy as np
import matplotlib.pyplot as plt


class MembFunc:
    """Membership function that can determine membership to a fuzzy set.

    Can be called as a function on both individual scalar inputs and
    iterable inputs such as lists and numpy arrays.

    Args:
        name: Name of membership function.
        params: Parameters of membership function.
        membership: Indicates how membership should be calculated.
            A string is interpreted as one of the :doc:`../membfunc_templates`.
            A custom callable that takes an array-like of floats (a), an iterable
            with each parameter (x), and returns membership can also be given.
            An iterable of floats of the same shape as the given domain
            parameters with corresponding membership values will be
            interpreted as a custom piecewise linear function.

    Attributes:
        name (str): Name of membership function.
        params (np.ndarray[float]): Parameters of the membership function.
            These can be domain parameters for linear functions or unique
            parameters in custom functions.
        fn_type (str): Name of template if used. Otherwise, 'custom'.
        center (float): Central or important point in domain of function
            used in built-in visualization and potentially as a zeroth order
            Takagi-Sugeno output. In linear functions, this defaults to the
            average of domain parameters corresponding to max membership values.
            For custom and Takagi-Sugeno output functions, this defaults to the
            first parameter.
        templates (Dict[str, Union[Iterable[float], callable, None]]):
            Static dictionary with template names as keys with iterables of
            membership values corresponding to parameters, a callable that
            takes input and parameters and returns membership, or None to
            indicate the function is to be a Takagi Sugeno output function.

    Examples:
            >>> # Trapezoidal function with and without templates:
            >>> fn1 = MembFunc([1, 2, 3, 4], "trapezoidal")
            >>> fn2 = MembFunc([1, 2, 3, 4], [0, 1, 1, 0])

            >>> # Standard gaussian function with and without templates:
            >>> fn3 = MembFunc([0, 1], "gaussian")
            >>> fn4 = MembFunc([0, 1], lambda a, x: exp(-((a - x[0]) ** 2 / (2 * x[1] ** 2)))

            >>> # Zeroth, first, and second order Takagi-Sugeno-Kang output functions:
            >>> fn5 = MembFunc([2.5], "tsk")      # --> 2.5
            >>> fn6 = MembFunc([8, 2], "tsk")     # --> 8 + 2^2
            >>> fn7 = MembFunc([1, 2, 3], "tsk")  # --> 1 + 2^2 + 3^3
    """
    # ----------
    # Attributes
    # ----------

    # Templates
    templates = {
        # Generic
        "triangular": [0, 1, 0],
        "trapezoidal": [0, 1, 1, 0],
        "leftedge": [1, 0],
        "rightedge": [0, 1],

        # Special
        "gaussian": lambda a, x: np.exp(-((a - x[0]) ** 2 / (2 * x[1] ** 2))),

        # Takagi-Sugeno-Kang output
        "tsk": None
    }

    # -----------
    # Constructor
    # -----------

    def __init__(self, name: str, params: Iterable[float],
                 membership: Union[str, callable, Iterable[float]]):
        # Save member function name
        self.name = name

        # Save parameters and default to linear
        self.params = np.array(params)
        self._is_linear = False

        # Depending on given membership type, construct the function
        # str      -> named template function
        # callable -> special membership function
        # iterable -> membership values for each parameter of linear function
        if isinstance(membership, str):
            self._build_template(membership)
        elif callable(membership):
            self._build_special(membership)
            self.fn_type = "custom"
        else:
            self._build_generic(membership)
            self.fn_type = "custom"

        # Create vectorized numpy function for applying sub-functions to inputs
        self._apply_sub_fns = np.vectorize(lambda x, ind: self._sub_fns[ind](x))

    # -------
    # Methods
    # -------

    def __call__(self, a: ArrayLike) -> ArrayLike:
        """Given a scalar or iterable input, returns membership values.

        Args:
            a: Input scalar, iterable, numpy array, or other array-like.

        Returns:
            Scalar or array of membership to the function depending on input.

        Raises:
            NotImplemented: TSK output functions cannot determine membership.

        Example:
            >>> fn = MembFunc("cold", [30, 40], "leftedge")
            >>> float_membership = fn(35)
            >>> list_memberships = fn([32, 35, 21, 68])
        """
        if self.fn_type == "tsk":
            raise NotImplemented("TSK output functions can't determine membership.")

        a = np.asarray(a)

        # Get sub-function indices based on input's position in domain
        if self._is_linear:
            indices = np.searchsorted(self.params, a)
        else:
            indices = np.zeros(a.shape, dtype=np.int)

        # Apply appropriate sub-functions to each input based on indices
        output = self._apply_sub_fns(a, indices)

        return output

    def plot(self, start: float, stop: float, num: int = 300,
             color: str = "black", **plt_kwargs):
        """Plots the function for a given domain using matplotlib.

        Args:
            start: Start of domain to plot.
            stop: End of domain to plot.
            num: Number of points to find membership to for plotting.
            color: matplotlib.pyplot color of the line representing the function.
            **plt_kwargs: matplotlib.pyplot plotting options.
        """
        domain = np.linspace(start, stop, num)
        plt.plot(domain, self(domain), color=color, **plt_kwargs)

        # Decorate
        plt.ylim(0.0, 1.05)
        plt.grid(visible=True, axis="y", alpha=0.5, ls="--")
        plt.title(self.name)

    # Helpers

    def _build_template(self, template_name: str):
        """Builds a membership function from a template.

        Args:
            template_name: Name of template to use.

        Raises:
            KeyError: Invalid template name.
        """
        # Get template membership
        try:
            membership = MembFunc.templates[template_name]
        except KeyError:
            raise KeyError(f"Template name {template_name} not found!")

        # Save template type
        self.fn_type = template_name

        # Build membership function
        # list     -> Membership values of generic linear function
        # callable -> Membership function
        # None     -> Takagi-Sugeno output function
        if isinstance(membership, list):
            self._build_generic(membership)
        elif callable(membership):
            self._build_special(membership)
        elif membership is None:
            self.center = self.params[0]

    def _build_generic(self, memb_vals: Iterable[float]):
        """Builds a generic linear function.

        Checks for compatible parameters and membership values before sorting
        and building piecewise sub-functions.

        Args:
            memb_vals: Values for each of the function's domain parameters.

        Raises:
            ValueError: A membership value is not a valid scalar between 0 and 1.
            ValueError: Parameter and membership dimensions don't match.
        """
        memb_vals = np.asarray(memb_vals)

        if np.any((memb_vals < 0.0) | (memb_vals > 1.0)):
            raise ValueError("A value in y is not between 0 and 1.")

        if self.params.shape[0] != memb_vals.shape[0]:
            s1, s2 = self.params.shape[0], memb_vals.shape[0]
            raise ValueError(f"Shape of domain parameters ({s1}) and function "
                             f"values ({s2}) don't match.")

        # Sort parameters if standard
        self.params.sort()

        # Create list of functions for piecewise evaluation and mark as linear
        self._sub_fns = self.__create_linear_subfunctions(memb_vals)
        self._is_linear = True

        # Save center point used in zeroth order tsk evaluation and plotting
        self.center = np.mean(self.params[np.where(memb_vals == np.max(memb_vals))])

    def _build_special(self, memb_func: callable):
        """Builds special membership function from template or given callable.

        Args:
            memb_func: Template or given callable that takes input (a) and params (x)
        """
        # Save custom function as sole sub-function
        self._sub_fns = [lambda a, x=self.params: memb_func(a, x)]

        # Save first parameter as center
        self.center = self.params[0]

    def __create_linear_subfunctions(self, memb_vals: np.ndarray) -> np.ndarray:
        """Creates an array of linear sub-functions to comprise the function.
        """
        y = memb_vals

        # Left of leftmost domain parameter
        sub_fns = [lambda xf: y[0]]

        # In between each domain parameter
        for i in range(self.params.shape[0] - 1):
            slope = (y[i + 1] - y[i]) / (self.params[i + 1] - self.params[i])
            if np.isinf(slope):
                raise ZeroDivisionError(f"Linear function parameters [{i}] and "
                                        f"[{i + 1}] are equal.")
            sub_fns.append(lambda xf, m=slope, xi=self.params[i], b=y[i]: m * (xf - xi) + b)

        # Right of rightmost domain parameter
        sub_fns.append(lambda xf: y[-1])

        return np.array(sub_fns)
