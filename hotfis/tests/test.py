"""
This is a documentation test.

**TEST**
"""


def test_fn(a: int, b: int) -> int:
    """Test function.

    This is a test function for docstrings.

    Parameters:
        a (int): First integer.
        b (int): Second integer.

    Returns:
        int: Sum of a and b.
    """
    return a + b

class TestClass:
    """Test class."""
    def __init__(self):
        self.a = 1

    def test_method(self, add: int) -> None:
        """Test method.

        Parameters:
            add (int): Number to add to self.a.
        """
        self.a += 1
