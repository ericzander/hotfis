"""Contains membership function definition.
"""

from typing import Iterable, Union

import numpy as np


class MembFunc:
    """Function that can, given input, calculate membership to a fuzzy set.

    Supports scalar and iterable inputs.

    Attributes:
        x: Domain parameters of membership function.
        y: Membership values at each point in domain parameters (x).
        name: Name of membership function.
    """
    # ----------
    # Attributes
    # ----------

    # Templates with membership values (y) and whether function is standard
    _function_templates = {
        # Standard fuzzy (standard=True)
        "triangular": (np.array([0, 1, 0]), True),
        "trapezoidal": (np.array([0, 1, 1, 0]), True),
        "leftedge": (np.array([1, 0]), True),
        "rightedge": (np.array([0, 1]), True),

        # Takagi-Sugeno-Kang outputs (y=None, standard=False)
        "tsk": (None, False)
    }

    # Function count used for naming functions when names are not supplied
    _fn_count = 0

    # -------
    # Methods
    # -------

    def __init__(self, x: Iterable[float], y: Union[str, Iterable[float]],
                 name: str = ""):
        """Membership function constructor.

        Examples of a trapezoidal function with and without templates:
            |  *fn1 = MembFunc([1, 2, 3, 4], [0, 1, 1, 0])*
            |  *fn2 = MembFunc([1, 2, 3, 4], "trapezoidal")*

        Example of a zeroth-order Takagi-Sugeno-Kang output function:
            |  *tsk_fn = MembFunc([2.5], "tsk")*

        Args:
            x: Domain parameters of membership function.
            y: Either membership values at each point in params or template name.
            name: Name of membership function. If not given, uses generic default.
        """
        # Save domain and membership values of key points
        self.x = np.array(x)

        #
        if isinstance(y, str):
            self._build_from_template(y)
        else:
            self.x.sort()
            self.y = np.array(y)
            self._standard_wrapup()

        # Ensure function has a name
        if not name:
            self.name = f"fn{MembFunc._fn_count}"
            MembFunc._fn_count += 1
        else:
            self.name = name

    def __call__(self, a: Union[float, Iterable[float]]) -> Union[float, np.ndarray]:
        """Given a scalar or iterable input, returns membership values.

        Args:
            a: Input scalar or array.

        Returns:
            Scalar or array of membership to the function depending on input.

        Raises:
            NotImplemented: TSK functions cannot determine membership.
        """
        if self.y is None:
            raise NotImplemented("TSK output functions can't determine membership.")

        a = np.asarray(a)

        # Get sub-function indices based on input's position in domain
        indices = np.searchsorted(self.x, a)

        # If scalar input, apply corresponding sub-function
        if indices.ndim == 0:
            output = self._sub_fns[indices](a)
            return np.squeeze(output)

        # If array input, apply corresponding sub-function for each value
        output = np.array([self._sub_fns[ind](val) for val, ind in zip(a, indices)])
        return output

    # Helpers

    def _build_from_template(self, template_name: str):
        """Builds a membership function when given a template name.

        Args:
            template_name: Name of template to use.

        Raises:
            KeyError: Invalid template name.
        """
        try:
            self.y, is_standard = MembFunc._function_templates[template_name]
        except KeyError:
            raise KeyError(f"Template name {template_name} not found!")

        # Sort, perform error-checking, and create sub-functions if standard
        if is_standard:
            self.x.sort()
            self._standard_wrapup()

        # Save relevant values if TSK
        if self.y is None:
            self._tsk_wrapup()

    def _tsk_wrapup(self):
        """Performs initialization wrap-up for TSK output functions
        """
        # Save value as tsk output
        self._tsk = self.x[0]

    def _standard_wrapup(self):
        """Performs initialization wrap-up for standard, non-TSK functions

        Raises:
            ValueError: A value in y is not a valid scalar between 0 and 1.
            ValueError: x and y dimensions don't match.
        """
        if np.any((self.y < 0.0) | (self.y > 1.0)):
            raise ValueError("A value in y is not between 0 and 1.")

        # Ensure x and y shapes match
        if self.x.shape[0] != self.y.shape[0]:
            s1, s2 = self.x.shape[0], self.y.shape[0]
            raise ValueError(f"Shape of domain ({s1}) and values ({s2}) don't match.")

        # Create list of functions for piecewise evaluation
        self._sub_fns = self._create_subfunctions()

    def _create_subfunctions(self):
        """Creates an array of sub-functions comprising the piecewise function.
        """
        # Left of leftmost domain parameter
        sub_fns = [lambda xi: self.y[0]]

        # In between each domain parameter
        for i in range(self.x.shape[0] - 1):
            slope = (self.y[i+1] - self.y[i]) / (self.x[i+1] - self.x[i])
            sub_fns.append(lambda xi: slope * (xi - self.x[i]) + self.y[i])

        # Right of rightmost domain parameter
        sub_fns.append(lambda xi: self.y[-1])

        return np.array(sub_fns)
