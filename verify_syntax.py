#!/usr/bin/env python
"""Quick syntax check of the updated gaussian_residual module."""
import sys
from pathlib import Path

# Check syntax
try:
    import py_compile
    py_compile.compile(
        str(Path(__file__).parent / "src/fcd_prototype/gaussian_residual.py"),
        doraise=True
    )
    print("✓ gaussian_residual.py: syntax OK")
except py_compile.PyCompileError as e:
    print(f"✗ Syntax error: {e}")
    sys.exit(1)

# Try importing
sys.path.insert(0, str(Path(__file__).parent / "src"))
try:
    from fcd_prototype.gaussian_residual import (
        compare_gaussian_fields,
        _evaluate_gaussian_field_on_grid,
        _field_method_heatmap,
    )
    print("✓ All new functions importable")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Check that field-mode is available
import inspect
sig = inspect.signature(compare_gaussian_fields)
if "heatmap_method" in sig.parameters:
    print("✓ heatmap_method parameter present")
else:
    print("✗ heatmap_method parameter missing")
    sys.exit(1)

if "field_chunk_size" in sig.parameters:
    print("✓ field_chunk_size parameter present")
else:
    print("✗ field_chunk_size parameter missing")
    sys.exit(1)

print("\n✅ All syntax and import checks passed!")
