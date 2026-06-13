#!/usr/bin/env python3
"""
D-Vine Copula structure selection, training, and out-of-sample evaluation.

Implements:
  1. kendall_tau_matrix(u_data)         -- Pairwise Kendall's tau
  2. select_vine_structure(u_data)      -- R-Vine/C-Vine/D-Vine selection
  3. DVINE_STRUCTURE                    -- Hard-coded 4-tree D-Vine structure
  4. train_dvine(u_data, model_type)    -- Core training (static/figas/gas)
  5. eval_test_dvine(...)               -- OOS evaluation
  6. compare_models(u_train, u_test)    -- Full comparison pipeline

References:
  R code: /mnt/d/zcc/Rcode/代码(1).txt  lines 339-483 (selection), 492-1071 (training)
  figas_filter.py  -- FIGAS filter, static copula fit, PDF, h-functions
  gas_filter.py    -- GAS filter
  config.py        -- DVINE_ORDER, DVINE_NAMES, FAMILY_SET, SEED, PRINT_WIDTH
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

# ── Import project config ──────────────────────────────────────────────────
try:
    from config import (
        DVINE_ORDER, DVINE_NAMES, FAMILY_SET, ASSETS, SEED, PRINT_WIDTH
    )
except ImportError:
    DVINE_ORDER = [2, 4, 5, 1, 3]
    DVINE_NAMES = ["ydlMIB", "nfFTSE", "bxBOVESPA", "hushen300", "xxlNZ50"]
    FAMILY_SET = [1, 2, 3, 4, 5, 10]
    ASSETS = ["hushen300", "ydlMIB", "xxlNZ50", "nfFTSE", "bxBOVESPA"]
    SEED = 123
    PRINT_WIDTH = 70

# ── Import from sibling modules ────────────────────────────────────────────
from figas_filter import (
    _static_copula_fit,
    _bicop_pdf,
    _bicop_hfunc,
    _safe_bicop_hfunc,
    filter_figas,
    estimate_figas_params,
)
from gas_filter import filter_gas, estimate_gas_params


# ============================================================================
# 1. Kendall's tau matrix
# ============================================================================

def kendall_tau_matrix(u_data):
    """
    Compute pairwise Kendall's tau correlation matrix.

    Parameters
    ----------
    u_data : pd.DataFrame or np.ndarray  (n x d)
        Data array.

    Returns
    -------
    tau_mat : np.ndarray  (d x d)
        Kendall's tau matrix.
    """
    if isinstance(u_data, pd.DataFrame):
        arr = u_data.values
    else:
        arr = np.asarray(u_data, dtype=float)

    n_cols = arr.shape[1]
    tau_mat = np.zeros((n_cols, n_cols))

    print("Kendall's tau 矩阵:")
    print("-" * PRINT_WIDTH)

    for i in range(n_cols):
        for j in range(n_cols):
            if i == j:
                tau_mat[i, j] = 1.0
            elif i < j:
                tau_val, _ = kendalltau(arr[:, i], arr[:, j])
                tau_mat[i, j] = tau_val
                tau_mat[j, i] = tau_val

    # Print formatted matrix
    row_labels = (ASSETS if len(ASSETS) >= n_cols
                  else [f"Var{i + 1}" for i in range(n_cols)])
    row_labels = row_labels[:n_cols]

    header = "        " + "  ".join(f"{name:>10s}" for name in row_labels)
    print(header)
    for i, name in enumerate(row_labels):
        vals = "  ".join(f"{tau_mat[i, j]:10.4f}" for j in range(n_cols))
        print(f"{name:>8s} {vals}")
    print("-" * PRINT_WIDTH)

    return tau_mat


# ============================================================================
# 2. Vine copula structure selection
# ============================================================================

def select_vine_structure(u_data):
    """
    Select optimal Vine Copula structure (R-Vine, C-Vine, D-Vine).

    Tries pyvinecopulib for full selection first.  Falls back to a
    simplified D-Vine selection if the library is unavailable.

    Parameters
    ----------
    u_data : pd.DataFrame or np.ndarray  (n x d)
        Raw data (not yet pseudo-observations).

    Returns
    -------
    comparison : pd.DataFrame
        Columns: [Model, LogLik, AIC, BIC] -- best model by AIC printed.
    """
    # Convert to array
    if isinstance(u_data, pd.DataFrame):
        arr = u_data.values
    else:
        arr = np.asarray(u_data, dtype=float)

    # Drop rows with any NaN
    finite_mask = np.isfinite(arr).all(axis=1)
    arr = arr[finite_mask]

    # Check sample size
    n, n_cols = arr.shape
    if n < 30:
        raise ValueError(f"Sample size too small ({n} rows) for Vine model.")

    # Remove constant columns
    variances = np.var(arr, axis=0)
    if np.any(variances == 0):
        print("发现方差为0的列, 正在自动剔除...")
        arr = arr[:, variances > 0]
        n_cols = arr.shape[1]
        if n_cols < 2:
            raise ValueError("After dropping constant columns, fewer than 2 remain.")

    # Compute pseudo-observations
    pobs = np.zeros_like(arr)
    for j in range(n_cols):
        pobs[:, j] = (np.argsort(np.argsort(arr[:, j])) + 1.0) / (n + 1.0)

    # Print variable names
    if isinstance(u_data, pd.DataFrame):
        var_names = u_data.columns.tolist()
        if len(var_names) > n_cols:
            var_names = [f"Var{i + 1}" for i in range(n_cols)]
    else:
        var_names = [f"Var{i + 1}" for i in range(n_cols)]
    print(f"当前用于建模的变量: {', '.join(var_names)}")

    # ========================================================================
    # Attempt pyvinecopulib-based full selection
    # ========================================================================
    try:
        import pyvinecopulib as pv
        print("=== 正在自动构建最优 Vine Copula 结构 (pyvinecopulib) ... ===")
        results = {}
        family_set = [1, 2, 3, 4, 5, 10]

        for vtype in ["RVine", "CVine", "DVine"]:
            print(f"  --- {vtype} 选择中 ... ---")
            if vtype == "DVine":
                # D-Vine with TSP ordering
                tau_mat = np.zeros((n_cols, n_cols))
                for i in range(n_cols):
                    for j in range(n_cols):
                        if i < j:
                            t, _ = kendalltau(pobs[:, i], pobs[:, j])
                            tau_mat[i, j] = t
                            tau_mat[j, i] = t
                        elif i == j:
                            tau_mat[i, j] = 1.0
                dist_mat = 1.0 - np.abs(tau_mat)
                try:
                    import python_tsp.distances as tsp_dist
                    import python_tsp.heuristics as tsp_heur
                    dist_sq = ((dist_mat + dist_mat.T) / 2.0).astype(np.float64)
                    dist_sq[np.diag_indices_from(dist_sq)] = 0.0
                    perm, _ = tsp_heur.solve_tsp_local_search(
                        tsp_dist.TSPDistanceMatrix(dist_sq)
                    )
                    order = [int(p + 1) for p in perm]
                    print(f"  D-vine 最优变量顺序: {order}")
                except ImportError:
                    order = list(range(1, n_cols + 1))
                    print(f"  TSP library unavailable; using sequential order: {order}")

                mat, family_arr, theta_arr, nu_arr = pv.D2RVine(order)
                ctrl = pv.FitControlsVinecop(
                    family_set=family_set,
                    selection_criterion="aic"
                )
                vc = pv.Vinecop(data=pv.to_pseudo_obs(pobs), controls=ctrl, matrix=mat)
                results[vtype] = {
                    "logLik": vc.loglik() if hasattr(vc, 'loglik') else vc.get_loglik(),
                    "AIC": vc.aic() if hasattr(vc, 'aic') else vc.get_aic(),
                    "BIC": vc.bic() if hasattr(vc, 'bic') else vc.get_bic(),
                }
            else:
                structure = getattr(pv, f"{vtype}StructureSelect")
                result = structure(
                    data=pobs,
                    familyset=family_set,
                    selectioncrit="AIC"
                )
                if vtype == "RVine":
                    results[vtype] = {
                        "logLik": result.logLik,
                        "AIC": result.AIC,
                        "BIC": result.BIC,
                    }
                elif vtype == "CVine":
                    results[vtype] = {
                        "logLik": result.logLik,
                        "AIC": result.AIC,
                        "BIC": result.BIC,
                    }

        # Build comparison dataframe
        comparison = pd.DataFrame([
            {"Model": "R-vine", "LogLik": results["RVine"]["logLik"],
             "AIC": results["RVine"]["AIC"], "BIC": results["RVine"]["BIC"]},
            {"Model": "C-vine", "LogLik": results["CVine"]["logLik"],
             "AIC": results["CVine"]["AIC"], "BIC": results["CVine"]["BIC"]},
            {"Model": "D-vine", "LogLik": results["DVine"]["logLik"],
             "AIC": results["DVine"]["AIC"], "BIC": results["DVine"]["BIC"]},
        ])

    except (ImportError, Exception) as e:
        print(f"pyvinecopulib 不可用或出错 ({e}), 使用简化 D-Vine 结构选择.")
        print("=== 简化 D-Vine 结构选择 ===")

        # --------------------------------------------------------------------
        # Simplified D-Vine selection (no external Vine library)
        # --------------------------------------------------------------------
        # Compute Kendall's tau matrix -> distance matrix
        tau_mat = kendall_tau_matrix(pobs)
        dist_mat = 1.0 - np.abs(tau_mat)
        print(f"\n距离矩阵 (1 - |tau|):")
        print("-" * PRINT_WIDTH)
        row_labels = var_names[:n_cols]
        header = "        " + "  ".join(f"{name:>10s}" for name in row_labels)
        print(header)
        for i, name in enumerate(row_labels):
            vals = "  ".join(f"{dist_mat[i, j]:10.4f}" for j in range(n_cols))
            print(f"{name:>8s} {vals}")
        print("-" * PRINT_WIDTH)

        # Simplified ordering: use DVINE_ORDER from config (1-based)
        # or default to 1,2,3,4,5
        order_1b = DVINE_ORDER if len(DVINE_ORDER) == n_cols else list(range(1, n_cols + 1))
        print(f"D-vine 变量顺序: {order_1b}")

        # ====================================================================
        # Fit all pair-copulas independently and select best family by AIC
        # ====================================================================
        families_to_try = [1, 2, 3, 4, 5]
        n_pairs = n_cols * (n_cols - 1) // 2
        print(f"\n拟合 {n_pairs} 对 Copula, 候选族: {families_to_try}")
        print("-" * PRINT_WIDTH)

        total_ll = 0.0
        total_params = 0
        pair_idx = 0
        for i in range(n_cols):
            for j in range(i + 1, n_cols):
                pair_idx += 1
                uu1 = pobs[:, i]
                uu2 = pobs[:, j]

                best_fam = None
                best_aic = np.inf
                best_ll = None

                for fam in families_to_try:
                    try:
                        par, par2 = _static_copula_fit(uu1, uu2, fam)
                        pdfs = _bicop_pdf(uu1, uu2, fam, par, par2)
                        ll = float(np.sum(np.log(np.maximum(pdfs, 1e-10))))
                        # AIC = -2*LL + 2*k  where k = num params
                        k = 2 if fam == 2 else 1
                        aic = -2.0 * ll + 2.0 * k
                        if aic < best_aic:
                            best_aic = aic
                            best_fam = fam
                            best_ll = ll
                    except Exception:
                        continue

                if best_fam is not None:
                    total_ll += best_ll
                    total_params += (2 if best_fam == 2 else 1)
                    family_names = {1: "Gaussian", 2: "t", 3: "Clayton",
                                   4: "Gumbel", 5: "Frank"}
                    print(f"  Pair ({var_names[i]}, {var_names[j]}): "
                          f"{family_names.get(best_fam, str(best_fam))} "
                          f"(family={best_fam}), LL={best_ll:.2f}, AIC={best_aic:.2f}")

        aic_total = -2.0 * total_ll + 2.0 * total_params
        bic_total = -2.0 * total_ll + total_params * np.log(n)

        comparison = pd.DataFrame([
            {"Model": "D-vine (simplified)",
             "LogLik": total_ll, "AIC": aic_total, "BIC": bic_total}
        ])

    # ========================================================================
    # Print comparison and identify best model
    # ========================================================================
    print("\n" + "=" * PRINT_WIDTH)
    print("Vine Copula 模型比较:")
    print("=" * PRINT_WIDTH)
    print(comparison.to_string(index=False))
    print("-" * PRINT_WIDTH)

    best_idx = comparison["AIC"].idxmin()
    best_name = comparison.loc[best_idx, "Model"]
    best_aic = comparison.loc[best_idx, "AIC"]
    print(f"\n最优模型 (按AIC): {best_name}")
    print(f"AIC = {best_aic:.2f}")
    print("=" * PRINT_WIDTH + "\n")

    return comparison


# ============================================================================
# 3. Hard-coded D-Vine structure (matching R code dvine_structure)
# ============================================================================

DVINE_STRUCTURE = [
    # Tree 1: 4 edges
    [{"v1": 1, "v2": 2, "family": 14},    # (1,2) Survival Gumbel
     {"v1": 2, "v2": 3, "family": 2},     # (2,3) t
     {"v1": 3, "v2": 4, "family": 2},     # (3,4) t
     {"v1": 4, "v2": 5, "family": 14}],   # (4,5) Survival Gumbel
    # Tree 2: 3 edges
    [{"v1": 1, "v2": 3, "family": 2},     # (1,3|2) t
     {"v1": 2, "v2": 4, "family": 2},     # (2,4|3) t
     {"v1": 3, "v2": 5, "family": 2}],    # (3,5|4) t
    # Tree 3: 2 edges
    [{"v1": 1, "v2": 4, "family": 2},     # (1,4|2,3) t
     {"v1": 2, "v2": 5, "family": 23}],   # (2,5|3,4) Clayton 90-rotated
    # Tree 4: 1 edge
    [{"v1": 1, "v2": 5, "family": 3}]      # (1,5|2,3,4) Clayton
]


# ============================================================================
# 4. D-Vine training (core function)
# ============================================================================

def train_dvine(u_data, model_type):
    """
    Train a D-Vine Copula model on pseudo-observation data.

    Traverses the 4-tree DVINE_STRUCTURE sequentially, using h-functions
    from earlier trees as inputs to later trees.

    Parameters
    ----------
    u_data : np.ndarray or pd.DataFrame  (n x 5)
        Pseudo-observation data.  Must have 5 columns.
        Data should already be reordered by DVINE_ORDER.
    model_type : str
        One of {"static", "figas", "gas"}.

    Returns
    -------
    dict with keys:
        total_loglik : float
        edges        : list of lists  (4 trees, each with edge result dicts)
    """
    if model_type not in ("static", "figas", "gas"):
        raise ValueError(f"Unknown model_type: {model_type}. "
                         f"Use 'static', 'figas', or 'gas'.")

    # Convert to array
    if isinstance(u_data, pd.DataFrame):
        arr = u_data.values.astype(float)
    else:
        arr = np.asarray(u_data, dtype=float)

    n = arr.shape[0]
    results = [[], [], [], []]   # one slot per tree
    total_ll = 0.0

    model_label = {"static": "静态 D-Vine",
                   "figas": "D-Vine-FIGAS",
                   "gas": "D-Vine-GAS(1,1)"}[model_type]
    print(f"\n{'=' * PRINT_WIDTH}")
    print(f"===== 训练 {model_label} =====")
    print(f"{'=' * PRINT_WIDTH}")

    for tree_idx in range(4):
        tree_edges = DVINE_STRUCTURE[tree_idx]
        print(f"\n{'=' * 49}")
        print(f"-> 正在构建 D-Vine Tree {tree_idx + 1} ...")
        print(f"{'=' * 49}")

        for e, edge in enumerate(tree_edges):
            v1 = edge["v1"]
            v2 = edge["v2"]
            fam = edge["family"]

            # ----------------------------------------------------------------
            # Obtain u1, u2 for this edge
            # ----------------------------------------------------------------
            if tree_idx == 0:
                uu1 = arr[:, v1 - 1]
                uu2 = arr[:, v2 - 1]
            elif tree_idx == 1:
                uu1 = results[0][e]["h1"]
                uu2 = results[0][e + 1]["h2"]
            elif tree_idx == 2:
                if e == 0:
                    uu1 = results[1][0]["h1"]
                    uu2 = results[1][1]["h2"]
                else:
                    uu1 = results[1][1]["h1"]
                    uu2 = results[1][2]["h2"]
            elif tree_idx == 3:
                uu1 = results[2][0]["h1"]
                uu2 = results[2][1]["h2"]

            print(f"   Edge {e + 1} (vars {v1}-{v2}, family {fam})...")

            # ----------------------------------------------------------------
            # Model estimation
            # ----------------------------------------------------------------
            if model_type == "static":
                par_static, par2_static = _static_copula_fit(uu1, uu2, fam)
                h1, h2 = _safe_bicop_hfunc(uu1, uu2, fam, par_static, par2_static)
                pdf_vals = _bicop_pdf(uu1, uu2, fam, par_static, par2_static)
                loglik = float(np.sum(np.log(np.maximum(pdf_vals, 1e-10))))

                if fam == 2:
                    print(f"      --> [静态参数] par(rho): {par_static:.4f}, "
                          f"kappa(df): {par2_static:.2f}")
                else:
                    print(f"      --> [静态参数] par(rho/theta): {par_static:.4f}")
                print(f"      --> [模型拟合] 该 Edge 对数似然: {loglik:.2f}")

                res = {
                    "loglik": loglik,
                    "h1": h1,
                    "h2": h2,
                    "params": {"par": par_static, "par2": par2_static},
                    "fam": fam,
                }

            elif model_type == "figas":
                best_params, fres = estimate_figas_params(
                    uu1, uu2, fam, verbose=False
                )
                # Re-print with edge context
                if fam == 2:
                    print(f"      -->[动态参数] mu: {best_params[0]:.4f}, "
                          f"alpha: {best_params[1]:.4f}, "
                          f"beta: {best_params[2]:.4f}, "
                          f"d: {best_params[3]:.4f}, "
                          f"kappa(df): {best_params[4]:.2f}")
                else:
                    print(f"      -->[动态参数] mu: {best_params[0]:.4f}, "
                          f"alpha: {best_params[1]:.4f}, "
                          f"beta: {best_params[2]:.4f}, "
                          f"d: {best_params[3]:.4f}")
                print(f"      -->[模型拟合] 该 Edge 对数似然: {fres['loglik']:.2f}")

                res = {
                    "loglik": fres["loglik"],
                    "h1": fres["h1"],
                    "h2": fres["h2"],
                    "params": best_params,
                    "fam": fam,
                    "par_seq": fres.get("par_t", None),
                }

            elif model_type == "gas":
                best_params, fres = estimate_gas_params(
                    uu1, uu2, fam, verbose=False
                )
                # Re-print with edge context
                if fam == 2:
                    print(f"      -->[动态参数] mu: {best_params[0]:.4f}, "
                          f"alpha: {best_params[1]:.4f}, "
                          f"beta: {best_params[2]:.4f}, "
                          f"kappa(df): {best_params[3]:.2f}")
                else:
                    print(f"      -->[动态参数] mu: {best_params[0]:.4f}, "
                          f"alpha: {best_params[1]:.4f}, "
                          f"beta: {best_params[2]:.4f}")
                print(f"      -->[模型拟合] 该 Edge 对数似然: {fres['loglik']:.2f}")

                res = {
                    "loglik": fres["loglik"],
                    "h1": fres["h1"],
                    "h2": fres["h2"],
                    "params": best_params,
                    "fam": fam,
                    "par_seq": fres.get("par_t", None),
                }

            results[tree_idx].append(res)
            total_ll += res["loglik"]

        # --------------------------------------------------------------------
        # Tree-level summary
        # --------------------------------------------------------------------
        tree_ll = sum(r["loglik"] for r in results[tree_idx])
        print(f"==> Tree {tree_idx + 1} 结构完成 | 本层累积对数似然: {tree_ll:.2f}")

    print(f"\n{model_label} 模型训练完成, 总对数似然: {total_ll:.4f}")
    return {"total_loglik": total_ll, "edges": results}


# ============================================================================
# 5. Out-of-sample evaluation
# ============================================================================

def eval_test_dvine(u_test_data, trained_model, model_type):
    """
    Evaluate a trained D-Vine model on out-of-sample (test) data.

    Uses the same sequential tree traversal as training, but:
      - "static": uses pre-estimated parameters directly (no re-estimation).
      - "figas": calls filter_figas with trained params (no re-estimation).
      - "gas":   calls filter_gas with trained params (no re-estimation).

    Parameters
    ----------
    u_test_data : np.ndarray or pd.DataFrame  (n_test x 5)
        OOS pseudo-observation data, already reordered by DVINE_ORDER.
    trained_model : dict
        Output of train_dvine(), with "edges" key.
    model_type : str
        One of {"static", "figas", "gas"}.

    Returns
    -------
    total_ll : float
        Total OOS log-likelihood summed across all 40 edges.
    """
    if isinstance(u_test_data, pd.DataFrame):
        arr = u_test_data.values.astype(float)
    else:
        arr = np.asarray(u_test_data, dtype=float)

    total_ll = 0.0
    results_test = [[], [], [], []]

    for tree_idx in range(4):
        tree_edges = DVINE_STRUCTURE[tree_idx]
        for e, edge in enumerate(tree_edges):
            v1 = edge["v1"]
            v2 = edge["v2"]
            fam = edge["family"]

            # ----------------------------------------------------------------
            # Obtain u1, u2 (from test data or previous tree h-functions)
            # ----------------------------------------------------------------
            if tree_idx == 0:
                uu1 = arr[:, v1 - 1]
                uu2 = arr[:, v2 - 1]
            elif tree_idx == 1:
                uu1 = results_test[0][e]["h1"]
                uu2 = results_test[0][e + 1]["h2"]
            elif tree_idx == 2:
                if e == 0:
                    uu1 = results_test[1][0]["h1"]
                    uu2 = results_test[1][1]["h2"]
                else:
                    uu1 = results_test[1][1]["h1"]
                    uu2 = results_test[1][2]["h2"]
            elif tree_idx == 3:
                uu1 = results_test[2][0]["h1"]
                uu2 = results_test[2][1]["h2"]

            # ----------------------------------------------------------------
            # Compute log-likelihood and h-functions on test data
            # ----------------------------------------------------------------
            if model_type == "static":
                params = trained_model["edges"][tree_idx][e]["params"]
                par_static = params["par"]
                par2_static = params.get("par2", 0) if isinstance(params, dict) else 0
                h1, h2 = _safe_bicop_hfunc(uu1, uu2, fam, par_static, par2_static)
                pdf_vals = _bicop_pdf(uu1, uu2, fam, par_static, par2_static)
                loglik = float(np.sum(np.log(np.maximum(pdf_vals, 1e-10))))

            elif model_type == "figas":
                params = trained_model["edges"][tree_idx][e]["params"]
                fres = filter_figas(params, uu1, uu2, fam)
                loglik = fres["loglik"]
                h1 = fres["h1"]
                h2 = fres["h2"]

            elif model_type == "gas":
                params = trained_model["edges"][tree_idx][e]["params"]
                fres = filter_gas(params, uu1, uu2, fam)
                loglik = fres["loglik"]
                h1 = fres["h1"]
                h2 = fres["h2"]

            results_test[tree_idx].append({"h1": h1, "h2": h2})
            total_ll += loglik

    return total_ll


# ============================================================================
# 6. Full model comparison (train + OOS eval)
# ============================================================================

def compare_models(u_train, u_test):
    """
    Train all three D-Vine models on training data, evaluate on test data,
    and print a comparison table.

    Applies DVINE_ORDER reordering to both train and test sets.

    Parameters
    ----------
    u_train : pd.DataFrame or np.ndarray  (n_train x 5)
        Training pseudo-observations.
    u_test : pd.DataFrame or np.ndarray  (n_test x 5)
        Test pseudo-observations.

    Returns
    -------
    best_model_name : str
        Name of the best-performing model ("static", "figas", or "gas").
    results_dict : dict
        Dictionary with keys:
          - "static": trained static model
          - "figas":  trained FIGAS model
          - "gas":    trained GAS model
          - "comparison": pd.DataFrame with OOS results
    """
    # Reorder by DVINE_ORDER (1-based -> 0-based)
    order_0b = [x - 1 for x in DVINE_ORDER]

    if isinstance(u_train, pd.DataFrame):
        u_train_re = u_train.iloc[:, order_0b].values.astype(float)
    else:
        u_train_re = np.asarray(u_train, dtype=float)[:, order_0b]

    if isinstance(u_test, pd.DataFrame):
        u_test_re = u_test.iloc[:, order_0b].values.astype(float)
    else:
        u_test_re = np.asarray(u_test, dtype=float)[:, order_0b]

    print(f"\n{'#' * PRINT_WIDTH}")
    print(f"#  D-Vine Copula 模型训练与样本外 (OOS) 评估")
    print(f"#  训练集样本数: {u_train_re.shape[0]}, 测试集样本数: {u_test_re.shape[0]}")
    print(f"#  D-Vine 顺序: {DVINE_NAMES}")
    print(f"{'#' * PRINT_WIDTH}")

    # ── Train all three models ─────────────────────────────────────────────
    static_model = train_dvine(u_train_re, "static")
    figas_model = train_dvine(u_train_re, "figas")
    gas_model = train_dvine(u_train_re, "gas")

    # ── OOS evaluation ─────────────────────────────────────────────────────
    print(f"\n{'=' * PRINT_WIDTH}")
    print(f"===== 计算测试集 OOS 对数似然 =====")
    print(f"{'=' * PRINT_WIDTH}")

    static_test_ll = eval_test_dvine(u_test_re, static_model, "static")
    figas_test_ll = eval_test_dvine(u_test_re, figas_model, "figas")
    gas_test_ll = eval_test_dvine(u_test_re, gas_model, "gas")

    # ── Comparison table ───────────────────────────────────────────────────
    comparison = pd.DataFrame([
        {"Model": "静态 D-Vine",
         "Train LogLik": static_model["total_loglik"],
         "OOS LogLik": static_test_ll},
        {"Model": "D-Vine + FIGAS",
         "Train LogLik": figas_model["total_loglik"],
         "OOS LogLik": figas_test_ll},
        {"Model": "D-Vine + GAS(1,1)",
         "Train LogLik": gas_model["total_loglik"],
         "OOS LogLik": gas_test_ll},
    ])

    print(f"\n{'=' * PRINT_WIDTH}")
    print("样本外 (OOS) 结果比较:")
    print(f"{'=' * PRINT_WIDTH}")
    for _, row in comparison.iterrows():
        print(f"{row['Model']:<22s} "
              f"Train LogLik: {row['Train LogLik']:>12.4f}  "
              f"OOS LogLik: {row['OOS LogLik']:>12.4f}")
    print(f"{'=' * PRINT_WIDTH}")

    best_idx = comparison["OOS LogLik"].idxmax()
    best_name = comparison.loc[best_idx, "Model"]
    best_ll = comparison.loc[best_idx, "OOS LogLik"]

    model_map = {
        "静态 D-Vine": "static",
        "D-Vine + FIGAS": "figas",
        "D-Vine + GAS(1,1)": "gas",
    }
    best_model_name = model_map.get(best_name, "static")

    print(f"\n最优模型是: {best_name}")
    print(f"OOS LogLik = {best_ll:.4f}")
    print(f"{'=' * PRINT_WIDTH}\n")

    results_dict = {
        "static": static_model,
        "figas": figas_model,
        "gas": gas_model,
        "comparison": comparison,
    }

    return best_model_name, results_dict


# ============================================================================
# 7. Quick smoke-test (runs when executed directly)
# ============================================================================

if __name__ == "__main__":
    print("=" * PRINT_WIDTH)
    print("vine_copula.py -- self-test")
    print("=" * PRINT_WIDTH)

    rng = np.random.RandomState(SEED)
    n_train = 500
    n_test = 150

    # ── Generate synthetic data with known dependence structure ─────────────
    # Use a simple Gaussian copula with positive correlations
    from scipy.stats import norm

    # Correlation matrix matching the 5-asset structure
    rho = 0.4
    cov = np.array([
        [1.0, rho, rho / 2, rho / 2, rho / 3],
        [rho, 1.0, rho, rho / 2, rho / 3],
        [rho / 2, rho, 1.0, rho, rho / 2],
        [rho / 2, rho / 2, rho, 1.0, rho],
        [rho / 3, rho / 3, rho / 2, rho, 1.0],
    ])

    # Generate multivariate normal, then convert to uniforms
    L = np.linalg.cholesky(cov)
    z_train = rng.randn(n_train, 5) @ L.T
    z_test = rng.randn(n_test, 5) @ L.T

    u_train_sim = norm.cdf(z_train)
    u_test_sim = norm.cdf(z_test)

    # Convert to DataFrames for realistic testing
    u_train_df = pd.DataFrame(u_train_sim, columns=ASSETS)
    u_test_df = pd.DataFrame(u_test_sim, columns=ASSETS)

    # ── Test 1: Kendall's tau matrix ───────────────────────────────────────
    print("\n[Test 1] kendall_tau_matrix")
    tau_mat = kendall_tau_matrix(u_train_df)
    print(f"  tau_mat shape: {tau_mat.shape}")

    # ── Test 2: Vine structure selection ───────────────────────────────────
    print("\n[Test 2] select_vine_structure")
    comparison = select_vine_structure(u_train_df)

    # ── Test 3: Train static D-Vine ────────────────────────────────────────
    print("\n[Test 3] train_dvine (static)")
    order_0b = [x - 1 for x in DVINE_ORDER]
    u_train_re = u_train_df.iloc[:, order_0b].values.astype(float)
    static_model = train_dvine(u_train_re, "static")
    print(f"  Static total loglik: {static_model['total_loglik']:.2f}")

    # ── Test 4: OOS eval for static ────────────────────────────────────────
    print("\n[Test 4] eval_test_dvine (static)")
    u_test_re = u_test_df.iloc[:, order_0b].values.astype(float)
    oos_ll = eval_test_dvine(u_test_re, static_model, "static")
    print(f"  Static OOS loglik: {oos_ll:.2f}")

    # ── Test 5: Train FIGAS (smoke test on 1 edge worth of data) ───────────
    print("\n[Test 5] train_dvine (figas) -- edge 1 only (quick smoke)")
    # Test a single edge to avoid long optimization times
    uu1 = u_train_re[:, 0]
    uu2 = u_train_re[:, 1]
    best_p, fres = estimate_figas_params(uu1, uu2, 14, verbose=True)
    print(f"  FIGAS edge 1 loglik: {fres['loglik']:.2f}")

    # ── Test 6: Train GAS (smoke test) ─────────────────────────────────────
    print("\n[Test 6] train_dvine (gas) -- edge 1 only (quick smoke)")
    best_p2, fres2 = estimate_gas_params(uu1, uu2, 14, verbose=True)
    print(f"  GAS edge 1 loglik: {fres2['loglik']:.2f}")

    # ── Test 7: Full compare_models ────────────────────────────────────────
    print("\n[Test 7] compare_models (full pipeline)")
    best_name, results_dict = compare_models(u_train_df, u_test_df)
    print(f"  Best model: {best_name}")
    print(f"  Comparison table:\n{results_dict['comparison']}")

    print(f"\n{'=' * PRINT_WIDTH}")
    print("All self-tests completed.")
    print(f"{'=' * PRINT_WIDTH}")
