# Author: Simon Hirsch
# License: GPL-3.0
import warnings


def test_import():
    try:
        import ondil  # noqa

        failed = False
    except Exception:
        failed = True

    assert not failed, "Import failed with Exception."


def test_no_syntaxwarnings_on_import():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", SyntaxWarning)
        import ondil  # noqa

    assert not any(issubclass(warn.category, SyntaxWarning) for warn in w), w
