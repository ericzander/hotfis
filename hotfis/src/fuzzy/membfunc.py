"""Contains membership function definition.
"""

from typing import Iterable, Union

import numpy as np
import matplotlib.pyplot as plt


class MembFunc:
    """Function that encompasses and can calculate membership to a fuzzy set.

    Can be called as a function on both individual scalar inputs and
    iterable inputs such as lists and numpy arrays.

    Attributes:
        params (np.ndarray[float]): Parameters of the membership function.
            These can be domain parameters for linear functions or unique
            parameters in custom functions.
        center (float): Central or important point in domain of function
            used in built-in visualization and potentially as a zeroth order
            Takagi-Sugeno output if the function is used as output in a ruleset.
            In linear functions, this defaults to the average of domain
            parameters corresponding to max membership values. For custom and
            Takagi-Sugeno output functions, this defaults to the first parameter.
        name (str): Name of membership function.
        fn_type (str): Name of template if used. Otherwise, 'custom'.

    Example usages for a 'cold' temperature fuzzy set:
        |  cold_func = MembFunc([0, 10, 20], "triangular", "cold")
        |  cold_membership = cold_func(12)
        |  cold_memberships = cold_func([12, 0, 3])
        |  cold_func.plot()
        |  plt.show()
    """
    # ----------
    # Attributes
    # ----------

    # Templates
    _function_templates = {
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

    # Function count used for naming functions when names are not supplied
    _fn_count = 0

    # -----------
    # Constructor
    # -----------

    def __init__(self, params: Iterable[float],
                 membership: Union[str, callable, Iterable[float]],
                 name: str = ""):
        """Membership function constructor.

        Args:
            params: Parameters of membership function.
            membership: Indicates how membership should be calculated.
                A string is interpreted as one of the :doc:`../membfunc_templates`.
                A custom callable that takes an input value (a), an iterable
                with each parameter (x), and returns a float can also be given.
                An iterable of floats of the same shape as the given domain
                parameters with corresponding membership values will be
                interpreted as a custom piecewise linear function.
            name: Name of membership function. If not given, uses generic default.

        Examples of a trapezoidal function with and without templates:
            |  fn1 = MembFunc([1, 2, 3, 4], "trapezoidal")
            |  fn2 = MembFunc([1, 2, 3, 4], [0, 1, 1, 0])

        Example of the standard gaussian function with and without templates:
            |  fn3 = MembFunc([0, 1], "gaussian")
            |  fn4 = MembFunc([0, 1], lambda a, x: exp(-((a - x[0]) ** 2 / (2 * x[1] ** 2)))

        Example of Takagi-Sugeno-Kang output functions:
            |  fn5 = MembFunc([2.5], "tsk")     *--> 2.5*
            |  fn6 = MembFunc([8, 2], "tsk")    *--> 8 + 2^2*
            |  fn7 = MembFunc([1, 2, 3], "tsk") *--> 1 + 2^2 + 3^3*
        """
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

        # Save function name
        if not name:
            self.name = f"fn{MembFunc._fn_count}"
            MembFunc._fn_count += 1
        else:
            self.name = name

    # -------
    # Methods
    # -------

    def __call__(self, a: Union[float, Iterable[float]]) -> Union[float, np.ndarray]:
        """Given a scalar or iterable input, returns membership values.

        Args:
            a: Input scalar or array.

        Returns:
            Scalar or array of membership to the function depending on input.

        Raises:
            NotImplemented: TSK output functions cannot determine membership.
        """
        if self.fn_type == "tsk":
            raise NotImplemented("TSK output functions can't determine membership.")

        a = np.asarray(a)

        # Get sub-function indices based on input's position in domain
        if self._is_linear:
            indices = np.searchsorted(self.params, a)
        else:
            indices = np.zeros(a.shape, dtype=np.int)

        # If scalar input, apply corresponding sub-function
        if indices.ndim == 0:
            output = self._sub_fns[indices](a)
            return np.squeeze(output)

        # If array input, apply corresponding sub-function for each value
        output = np.array([self._sub_fns[ind](val) for val, ind in zip(a, indices)])
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
            membership = MembFunc._function_templates[template_name]
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
            raise ValueError(f"Shape of domain parameters ({s1}) and values "
                             f"({s2}) don't match.")

        # Sort parameters if standard
        self.params.sort()

        # Create list of functions for piecewise evaluation and mark as linear
        self._sub_fns = self.__create_linear_subfunctions(memb_vals)
        self._is_linear = True

        # Save center point used in zeroth order tsk evaluation and plotting
        self.center = np.mean(self.params[np.where(memb_vals == memb_vals.max())])

    def _build_special(self, memb_func: callable):
        """Builds special function from template or given callable.

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
