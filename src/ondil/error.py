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


def check_scoringrules(has_scoringrules: bool) -> None:
    """Check if scoringrules is available."""
    if not has_scoringrules:
        raise ImportError(
            "scoringrules is required for scoring rules functionality. "
            "Please install it with 'pip install scoringrules'."
        )
