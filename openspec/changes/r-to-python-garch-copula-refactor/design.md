## Context

当前有一份完整的 R 代码（1074行）实现了 D-Vine-FIGAS Copula 模型来分析一带一路沿线 5 个股市指数的风险溢出效应。代码在 Windows+R 环境下可运行，但存在路径硬编码、单文件过长、R 性能瓶颈等问题。

参照 `D:\SCI投稿论文\model\gas_copula_simulation.py` 中已有的 Python GAS-Copula 实现（二元高斯 Copula + GAS 得分驱动），需要将其扩展为完整的 5 变量 D-Vine Copula 系统，并引入 FIGAS（分数阶积分 GAS）作为核心创新方法。

**约束**：
- Python 3.10+，依赖 `arch`>=6.0, `scipy`>=1.10, `numpy`>=1.24, `pandas`>=2.0
- 输出结果需与原 R 代码一致（相同随机种子 set.seed(123) → np.random.seed(123)）
- 中间数据文件格式从 CSV/RData 迁移到 CSV/Pickle

## Goals / Non-Goals

**Goals:**
- 完整复现 R 代码的分析流程：数据读取 → 描述性统计 → GJR-GARCH 边缘建模 → D-Vine Copula 选择 → FIGAS/GAS 动态建模 → OOS 评估
- 模块化设计，每个模块可独立测试和复用
- FIGAS 过滤函数需精确实现分数阶差分权重和 t/Gaussian/Clayton/Gumbel 族的得分函数
- 与 R 代码输出结果数值一致（可复现性）

**Non-Goals:**
- 不实现 CoVaR/风险溢出计算（`risk_measures.py` 留作后续扩展）
- 不实现 Adam 优化变体（参考代码中的 SimpleGASAdam，非论文核心方法）
- 不实现 R-Vine/C-Vine（D-Vine 是最终选择，但保留结构选择代码）
- 不实现实时/流式数据处理

## Decisions

### 1. 边缘 GARCH 建模：`arch` 库 vs 手写
- **选择**: 使用 `arch` 库 (`arch.univariate.GARCH`)
- **理由**: `arch` 库原生支持 GJR-GARCH + Student-t，并提供 `fixed_pars` 用于检验集过滤，与 R 的 `rugarch` 功能完全对齐
- **备选**: 手写 GJR-GARCH 需实现极大似然优化和约束条件，开发量过大且容易出错

### 2. Vine Copula: `pyvinecopula` vs 手写
- **选择**: 混合策略 — `pyvinecopula` 用于静态结构和参数估计，手写动态过滤函数（FIGAS/GAS）
- **理由**: `pyvinecopula` 已实现 BiCopEst、RVineStructureSelect 等核心功能，但时变参数过滤需 100% 控制，必须手写
- **备选**: 全部手写 Vine Copula 结构（开发量过大，且容易出错）

### 3. FIGAS 实现：逐元素循环 vs 向量化
- **选择**: 逐元素 for 循环（`figas_filter.py` 中的 `filter_figas_correct`）
- **理由**: FIGAS 的分数阶差分项（$\sum_{j=1}^L \psi_j y_{t-j}$）依赖过去路径，天然是递归的无法向量化。参考 R 代码也是逐期迭代
- **备选**: 用 `numba.jit` 加速循环（后续优化，首批不引入额外依赖）

### 4. 数值优化: `scipy.optimize.minimize` vs `nlminb`
- **选择**: `scipy.optimize.minimize(method='L-BFGS-B')`
- **理由**: L-BFGS-B 支持参数边界约束，与 R 的 `nlminb(lower, upper)` 功能一致
- **备选**: `nlminb` 在 Python 中无直接等价物，L-BFGS-B 是最接近的替代

### 5. 项目结构：7 个模块文件
- `config.py`: 集中管理所有路径、参数
- `utils.py`: 纯函数工具集
- `marginal_models.py`: 面向过程的边缘建模流水线
- `figas_filter.py`: 核心 FIGAS 过滤函数
- `gas_filter.py`: GAS(1,1) 过滤函数（对比基准）
- `vine_copula.py`: D-Vine 结构 + 训练/eval 框架
- `main.py`: 编排脚本（无类，纯函数调用）

### 6. 文件命名与路径管理
- **选择**: `pathlib.Path` 替代硬编码字符串
- **理由**: 跨平台兼容（Linux/WSL/Windows），不再需要类似 "C:/Users/12936/Desktop/" 的硬编码

## Risks / Trade-offs

- **[精度差异]** R 和 Python 的浮点运算实现不同，FIGAS 递归过程中细小数值差异可能逐期累积 → 在核心函数中设置与 R 一致的截断阈值（如 `max(min(x, 10), -10)`），关键输出用 `np.allclose` 验证
- **[pyvinecopula 不可用]** 如果 `pyvinecopula` 在 WSL 环境下编译失败 → 回退方案：用 `scipy.stats` + 手写 BiCopEst/BiCopPDF/BiCopHfunc
- **[FIGAS 收敛问题]** 分数阶参数 d 在边界（0.05, 0.49）处可能导致似然面平坦 → 使用多重初始值（R 代码中的 `runif` 随机初始化）并选择最优收敛结果
- **[性能]** D-Vine Tree 的 FIGAS 训练涉及 10 条边 × 每边参数优化，总计算量约是参考代码的 10 倍 → 预期训练时间 ~5-10 分钟（R 版本类似），可在后续用 `numba` 加速

## Open Questions

- `pyvinecopula` 在 WSL Python 3.10 环境下能否直接 `pip install`？（需在实际环境验证）
- 是否需要保留 R 代码中的 CoVaR/风险溢出部分？（当前标记为 Non-Goal，待确认）
- FIGAS 的参数 d 是否需要与 alpha/beta 联合优化，还是可以固定为某个经验值？
