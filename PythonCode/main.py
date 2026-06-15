#!/home/xuehao/miniconda3/envs/figas/bin/python3
"""
Main pipeline: 一带一路沿线国家股市风险溢出分析
D-Vine-FIGAS Copula 模型

Refactored from R code at /mnt/d/zcc/Rcode/代码(1).txt
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Add project root to path ────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import *
import utils
import marginal_models
import vine_copula


def main():
    """Run the complete analysis pipeline."""
    ensure_dirs()

    # ====================================================================
    # 模块 0: 数据读取与划分
    # ====================================================================
    print("\n" + "=" * PRINT_WIDTH)
    print("模块 0: 数据读取与划分")
    print("=" * PRINT_WIDTH)

    (returns_full, returns_train, returns_test,
     date_all, date_train, date_test) = utils.load_and_split_data()

    print(f"  全样本: {len(returns_full)} 天")
    print(f"  训练集: {len(returns_train)} 天")
    print(f"  测试集: {len(returns_test)} 天")

    # ====================================================================
    # 模块 1: 描述性统计
    # ====================================================================
    print("\n" + "=" * PRINT_WIDTH)
    print("模块 1: 描述性统计")
    print("=" * PRINT_WIDTH)

    utils.descriptive_stats(returns_full)

    # ====================================================================
    # 模块 2: 收益率时序图
    # ====================================================================
    print("\n" + "=" * PRINT_WIDTH)
    print("模块 2: 收益率时序图")
    print("=" * PRINT_WIDTH)

    utils.plot_return_series(returns_full, date_all, ASSETS)

    # ====================================================================
    # 模块 3: 诊断检验 (JB, ADF, ARCH-LM)
    # ====================================================================
    print("\n" + "=" * PRINT_WIDTH)
    print("模块 3: 诊断检验")
    print("=" * PRINT_WIDTH)

    utils.run_diagnostic_tests(returns_train, ASSETS)

    # ====================================================================
    # 模块 4: ARCH-LM 检验 (基于 ARMA 残差)
    # ====================================================================
    print("\n" + "=" * PRINT_WIDTH)
    print("模块 4: ARCH-LM 检验 (基于 ARMA 残差)")
    print("=" * PRINT_WIDTH)

    utils.run_arch_lm_on_arma_residuals(returns_train, ASSETS)

    # ====================================================================
    # 模块 5: Ljung-Box 白噪声检验
    # ====================================================================
    print("\n" + "=" * PRINT_WIDTH)
    print("模块 5: Ljung-Box 白噪声检验")
    print("=" * PRINT_WIDTH)

    utils.ljung_box_matrix(returns_full, ASSETS)

    # ====================================================================
    # 模块 6: ARMA-GARCH 阶数选择
    # ====================================================================
    print("\n" + "=" * PRINT_WIDTH)
    print("模块 6: ARMA-GARCH 阶数选择")
    print("=" * PRINT_WIDTH)

    utils.select_arma_garch_orders(returns_train, ASSETS)

    # ====================================================================
    # 模块 7: GJR-GARCH(1,1) + Student-t 边缘建模
    # ====================================================================
    print("\n" + "=" * PRINT_WIDTH)
    print("模块 7: GJR-GARCH(1,1) + Student-t 边缘建模")
    print("=" * PRINT_WIDTH)

    # 7.1 Fit GJR-GARCH on training set
    print("\n--- 7.1 拟合训练集 GJR-GARCH ---")
    fit_dict = marginal_models.fit_gjr_garch(returns_train, ASSETS, GARCH_SPECS)

    # 7.2 Extract standardized residuals
    print("\n--- 7.2 提取标准化残差 ---")
    z_dict, shape_dict = marginal_models.extract_std_residuals(fit_dict)

    # 7.3 Residual diagnostics
    print("\n--- 7.3 残差诊断 ---")
    marginal_models.residual_diagnostics(z_dict)

    # 7.4 PIT transform
    print("\n--- 7.4 PIT 变换 ---")
    if PIT_METHOD == "ecdf":
        print(f"  [ECDF mode] Using empirical CDF on raw returns (preserving long memory)")
        u_train_dict = marginal_models.pit_transform(
            z_dict, shape_dict, returns_df=returns_train
        )
    else:
        u_train_dict = marginal_models.pit_transform(z_dict, shape_dict)

    # 7.5 PIT uniformity check
    print("\n--- 7.5 PIT 均匀性检验 ---")
    marginal_models.pit_uniformity_check(u_train_dict, show_plots=False)

    # 7.6 Filter test set
    print("\n--- 7.6 检验集过滤 ---")
    z_test_dict, u_test_dict, test_filter_results = marginal_models.filter_test_set(
        fit_dict, returns_test, GARCH_SPECS, shape_dict,
        returns_train_df=(returns_train if PIT_METHOD == "ecdf" else None),
    )

    # 7.7 Save outputs
    print("\n--- 7.7 保存边缘模型输出 ---")
    marginal_models.save_marginal_outputs(
        u_train_dict, u_test_dict, fit_dict, None
    )

    # ====================================================================
    # 模块 8: Vine Copula 结构选择
    # ====================================================================
    print("\n" + "=" * PRINT_WIDTH)
    print("模块 8: Vine Copula 结构选择")
    print("=" * PRINT_WIDTH)

    # Build u_train DataFrame (1944 x 5)
    u_train_df = pd.DataFrame(u_train_dict)[ASSETS]

    # Kendall's tau
    tau_mat = vine_copula.kendall_tau_matrix(u_train_df)

    # Vine structure selection (R/C/D-Vine comparison via pyvinecopulib)
    comparison = vine_copula.select_vine_structure(u_train_df)

    # Save module 8 comparison
    comp_csv = DATA_DIR / "08_vine_comparison.csv"
    comparison.to_csv(comp_csv, index=False)
    print(f"\n模块 8 输出 → {comp_csv}")

    # Determine best vine type from comparison (by AIC)
    best_aic_idx = comparison["AIC"].idxmin()
    best_vine_str = comparison.loc[best_aic_idx, "Model"]  # "R-vine", "C-vine", "D-vine"
    vine_type_map = {"R-vine": "RVine", "C-vine": "CVine", "D-vine": "DVine"}
    best_vine_type = vine_type_map.get(best_vine_str, "DVine")
    print(f"最优 Vine 结构: {best_vine_str} → 模块 9 将使用 {best_vine_type}")

    # ====================================================================
    # 模块 9: R/C/D-Vine 动态 Copula (静态 vs FIGAS vs GAS)
    # ====================================================================
    print("\n" + "=" * PRINT_WIDTH)
    print(f"模块 9: {best_vine_str} 动态 Copula 建模 (含 R/C/D 对比)")
    print("=" * PRINT_WIDTH)

    u_test_df = pd.DataFrame(u_test_dict)[ASSETS]

    all_results = {}
    for vt, vt_label in [("DVine", "D-Vine"), ("CVine", "C-Vine"), ("RVine", "R-Vine")]:
        print(f"\n{'~' * PRINT_WIDTH}")
        print(f"  训练 {vt_label} 结构 ...")
        print(f"{'~' * PRINT_WIDTH}")
        best_model, results = vine_copula.compare_models(u_train_df, u_test_df, vine_type=vt)
        vine_copula.save_vine_outputs(results, vt, str(DATA_DIR))
        all_results[vt] = results

    # Use best vine type as final
    final_results = all_results.get(best_vine_type, all_results["DVine"])
    best_model = max(
        [(vt, r["comparison"].loc[r["comparison"]["OOS LogLik"].idxmax(), "OOS LogLik"])
         for vt, r in all_results.items()],
        key=lambda x: x[1]
    )[0]

    # ====================================================================
    # 完成
    # ====================================================================
    print("\n" + "=" * PRINT_WIDTH)
    print("分析完成!")
    print(f"最优 Vine 结构: {best_vine_str}")
    print(f"最优模型: {best_model}")
    print(f"中间数据已保存至: {DATA_DIR}")
    print("=" * PRINT_WIDTH)

    return final_results


if __name__ == "__main__":
    main()
