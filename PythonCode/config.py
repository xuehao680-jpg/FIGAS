#!/usr/bin/env python3
"""
Global configuration for the GARCH-Copula refactored project.
Replaces hard-coded paths and magic numbers in the original R code.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/mnt/d/zcc")
DATA_DIR    = PROJECT_ROOT / "data"
OUTPUT_DIR  = DATA_DIR                     # intermediate outputs go here
SCRIPT_DIR  = Path(__file__).resolve().parent

DATA_CSV     = DATA_DIR / "yidaiyilu(1).csv"
U_LIST_CSV   = DATA_DIR / "11_u_list.csv"
U_TEST_CSV   = DATA_DIR / "11_u_test.csv"
MODEL_PKL    = DATA_DIR / "11Marginal_Full_Data.pkl"  # replaces .RData

# ── Reproducibility ────────────────────────────────────────────────────
SEED = 123

# ── Train / Test split ──────────────────────────────────────────────────
TRAIN_RATIO = 0.8   # chrono 80-20 split

# ── Assets ─────────────────────────────────────────────────────────────
ASSETS = ["hushen300", "ydlMIB", "xxlNZ50", "nfFTSE", "bxBOVESPA"]

# ── GJR-GARCH specs (from R code auto.arima + GARCH selection) ────────
# Format: {asset: {"arma": (p,q), "garch": (p,q), "dist": "std"}}
GARCH_SPECS = {
    "hushen300":  {"arma": (0, 2), "garch": (1, 1)},
    "ydlMIB":     {"arma": (0, 2), "garch": (1, 1)},
    "xxlNZ50":    {"arma": (2, 2), "garch": (1, 1)},
    "nfFTSE":     {"arma": (3, 1), "garch": (1, 1)},
    "bxBOVESPA":  {"arma": (0, 1), "garch": (1, 1)},
}
DISTRIBUTION = "t"  # Student-t

# ── Copula families ────────────────────────────────────────────────────
#  1=Gaussian  2=t  3=Clayton  4=Gumbel  5=Frank  14=Survival Gumbel
# 23=Clayton 90-rotated
FAMILY_SET = [1, 2, 3, 4, 5, 10]  # for Vine structure selection

# D-Vine tree structure (reordered: ydlMIB=1, nfFTSE=2, bxBOVESPA=3, hushen300=4, xxlNZ50=5)
DVINE_ORDER = [2, 4, 5, 1, 3]  # column indices (1-based, matching R)
DVINE_NAMES = ["ydlMIB", "nfFTSE", "bxBOVESPA", "hushen300", "xxlNZ50"]

# ── FIGAS optimization bounds ──────────────────────────────────────────
FIGAS_BOUNDS = {
    "mu":    (-20.0, 20.0),
    "alpha": (0.001, 0.50),
    "beta":  (0.001, 0.999),
    "d":     (0.001, 0.49),
    "kappa": (2.1, 50.0),
}

FIGAS_N_RESTARTS = 10  # number of random init points for FIGAS estimation
FIGAS_OPTUNA_TRIALS = 80  # Optuna trials for family=2 (t-Copula); 60 for others

GAS_BOUNDS = {
    "mu":    (-15.0, 15.0),
    "alpha": (0.001, 0.4),
    "beta":  (0.01, 0.99),
    "kappa": (2.1, 30.0),
}

# ── Numerical stability ────────────────────────────────────────────────
PDF_FLOOR   = 1e-10      # min PDF value before log
SCORE_CLAMP = 10.0        # clamp scores to [-10, 10]
F_DIFF_H    = 1e-5        # step size for finite-difference score

# ── Display ────────────────────────────────────────────────────────────
PRINT_WIDTH = 70

def ensure_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
