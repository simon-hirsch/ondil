import os
import subprocess
import sys

import pytest

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
    # proc = subprocess.run([sys.executable, "-c", python_code], env=os.environ.copy())
    # assert proc.returncode == 0, "The extracted code did not run successfully"


@pytest.mark.parametrize("script", EXAMPLE_SCRIPTS)
def test_examples_in_py(script):
    exec(open(script).read(), {}, {})
    # proc = subprocess.run([sys.executable, script], env=os.environ.copy())
    # assert proc.returncode == 0, f"The example script {script} did not run successfully"
