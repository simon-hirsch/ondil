import subprocess

import pytest

EXAMPLES = [
    "README.md",
    "docs/index.md",
]


@pytest.mark.parametrize("filepath", EXAMPLES)
def test_examples(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        file = f.read()

    python_code = file.split("```python")[1].split("```")[0]
    proc = subprocess.run(["python", "-c", python_code])
    assert proc.returncode == 0, "The extracted code did not run successfully"
