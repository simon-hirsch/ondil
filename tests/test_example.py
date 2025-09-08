import pytest

# I have originally written this using subprocess
# but this does not work in some environments in the github actions CI/CD
# using the exec function works, but is less clean in terms of error reporting


EXAMPLE_MD = [
    "README.md",
    "docs/index.md",
]

EXAMPLE_SCRIPTS = [
    "examples/estimation_methods.py",
    "examples/batch_and_online_estimation.py",
]


@pytest.mark.parametrize("filepath", EXAMPLE_MD)
def test_examples_in_md(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        file = f.read()

    python_code = file.split("```python")[1].split("```")[0]
    exec(python_code, {}, {})


@pytest.mark.parametrize("script", EXAMPLE_SCRIPTS)
def test_examples_in_py(script):
    exec(open(script).read(), {}, {})
