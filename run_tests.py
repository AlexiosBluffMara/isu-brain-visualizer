#!/usr/bin/env python
import subprocess
import sys

result = subprocess.run([
    sys.executable, "-m", "pytest",
    "tests/test_gaussian_residual.py", "-q"
], cwd="/home/soumitty/isu-brain-mri")
sys.exit(result.returncode)
