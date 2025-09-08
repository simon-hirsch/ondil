import pytest

# I have originally written this using subprocess
# but this does not work in some environments in the github actions CI/CD
# using the exec function works, but is less clean in terms of error reporting


EXAMPLE_MD = [
    "README.md",
    "docs/index.md",
    "docs/estimators_and_methods.md",
    "docs/methods.md",
]

EXAMPLE_SCRIPTS = [
    "examples/estimation_methods.py",
    "examples/batch_and_online_estimation.py",
]


@pytest.mark.parametrize("filepath", EXAMPLE_MD)
def test_examples_in_md(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        file = f.read()

    python_blocks = []
    parts = file.split("```python")
    for part in parts[1:]:
        code = part.split("```")[0]
        python_blocks.append(code.strip())
    python_code = "\n\n".join(python_blocks)
    exec(python_code, {}, {})


@pytest.mark.parametrize("script", EXAMPLE_SCRIPTS)
def test_examples_in_py(script):
    with open(script) as f:
        exec(f.read(), {}, {})
