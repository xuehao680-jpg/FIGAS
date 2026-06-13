#!/usr/bin/env python3
"""Quick test of Optuna-based FIGAS on a single edge."""
import sys
import numpy as np
sys.path.insert(0, '/mnt/d/zcc/PythonCode')
import warnings
warnings.filterwarnings('ignore')

from figas_filter import estimate_figas_params

# Edge 1 Tree 1: ydlMIB (=col 2) vs nfFTSE (=col 3), family 14
np.random.seed(123)
u = np.genfromtxt('/mnt/d/zcc/data/11_u_list.csv', delimiter=',', skip_header=1)
u1, u2 = u[:, 1], u[:, 2]

print('Testing FIGAS Optuna: Edge (ydlMIB, nfFTSE), family=14')
best, fres = estimate_figas_params(u1, u2, 14)
print()
print('===== COMPARISON =====')
print(f'Static LL:       322.68')
print(f'FIGAS+Optuna LL: {fres["loglik"]:.2f}')
print(f'Improvement:     {fres["loglik"] - 322.68:.2f}')
