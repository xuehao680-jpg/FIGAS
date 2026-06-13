#!/home/xuehao/miniconda3/envs/figas/bin/python3
"""Test pyvinecopulib in the figas conda environment."""
import sys
print(f"Python: {sys.version}")
try:
    import pyvinecopulib as pv
    print(f"pyvinecopulib: {pv.__version__}  -- INSTALLED!")
    print("Ready to use full R-Vine, C-Vine, D-Vine structure selection.")
except ImportError as e:
    print(f"pyvinecopulib NOT available: {e}")
    sys.exit(1)
