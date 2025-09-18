class OutOfSupportError(ValueError):
    """Exception raised for values that are out of support."""

    def __init__(self, message="This value is out of support."):
        self.message = message
        super().__init__(self.message)


def check_matplotlib(has_mpl: bool) -> None:
    """Check if matplotlib is available."""
    if not has_mpl:
        raise ImportError(
            "Matplotlib is required for plotting. "
            "Please install it with 'pip install matplotlib'."
        )
