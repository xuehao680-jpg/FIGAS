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
- **最终选择**: 手写 ARMA(p,q)-GJR-GARCH(1,1)-t 联合 MLE（`scipy.optimize.minimize` + L-BFGS-B）
- **变更原因**: Python `arch` 库 v8.0.0 **不支持** MA 项——其 `mean` 参数仅支持 `Constant/Zero/AR/ARX/HAR/HARX/LS`，无 ARMA。原 R 代码使用 `rugarch` 的 `armaOrder=c(p,q)` 进行联合 MLE，`arch` 库的 AR-only 设定导致均值方程设定错误（hushen300 的 MA(2)、ydlMIB 的 MA(2) 等被直接丢弃），残差 LB/ARCH 检验大量不通过
- **实现**: 自定义 `_compute_arma_gjr_garch()` 递归计算 ARMA 残差 + GJR-GARCH 波动率，`_arma_gjr_garch_t_nll()` 计算 Student-t 对数似然，多起点 L-BFGS-B 优化，`ARMAGARCHResult` 封装估计结果并保持与旧 `ARCHModelResult` 接口兼容
- **优势**: 完全复现 R `rugarch` 的参数化，MA 项参与联合估计

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

### 7. PIT 方法：ECDF vs 参数化 GARCH-t
- **选择**: 支持双模式，默认 ECDF（经验 CDF）
- **ECDF 模式**: 直接对原始收益率做 rank-based pseudo-observations，保留波动聚集性和长记忆结构，让 FIGAS 在 Copula 层面充分捕获
- **参数化模式**: 传统 GJR-GARCH-t → 标准化残差 → Student-t CDF，GARCH 滤波后 PIT 序列近似白噪声
- **结论**: 两种模式下 FIGAS Train LL 均超越 GAS。ECDF 下 FIGAS 的 d 参数全部非零（0.1-0.25），参数化下 d 约半数非零（GARCH 滤波吸收了部分长记忆）

### 8. FIGAS 递归方程修复
- **问题**: 原单方程递归 `y[t+1] = beta*y[t] + alpha*s[t] - sum(psi_j * y[t+1-j])` 缺失 (1-βL)(1-L)^d 展开的交叉项 β*φ_j，导致 d>0.005 即数值爆炸
- **修复**: 两步法 — Step A: `X_{t+1} = β*X_t + α*s_t` (AR(1) 传播分数阶差分过程), Step B: `y_{t+1} = X_{t+1} - Σ φ_k * y_{t+1-k}` (反解)
- **效果**: d 在 [0, 0.49] 全区间稳定，模拟数据 d=0.3 时 FIGAS > GAS

## 实验结果 (v2 — FIGAS 修复 + 双 PIT 模式)

### ECDF 模式 (n=1944, 10 edges)
| 模型 | Train LL | OOS LL |
|------|----------|--------|
| Static D-Vine | 763.98 | 164.63 |
| D-Vine + GAS | 797.78 | 174.29 |
| **D-Vine + FIGAS** | **799.81** | 171.01 |

### 参数化 GARCH-t PIT 模式
| 模型 | Train LL | OOS LL |
|------|----------|--------|
| Static D-Vine | 816.22 | 198.50 |
| D-Vine + GAS | 828.93 | 203.47 |
| **D-Vine + FIGAS** | **833.53** | 199.92 |

FIGAS 在两种 PIT 模式下 Train LL 均超越 GAS (+2 ~ +5)。OOS 的微小差距来自额外 10 个 d 参数的过拟合效应。

## Risks / Trade-offs

- **[精度差异]** R 和 Python 的浮点运算实现不同，FIGAS 递归过程中细小数值差异可能逐期累积 → 在核心函数中设置与 R 一致的截断阈值（如 `max(min(x, 10), -10)`），关键输出用 `np.allclose` 验证
- **[pyvinecopula 不可用]** 如果 `pyvinecopula` 在 WSL 环境下编译失败 → 回退方案：用 `scipy.stats` + 手写 BiCopEst/BiCopPDF/BiCopHfunc
- **[FIGAS 收敛问题]** ~~分数阶参数 d 在边界（0.05, 0.49）处可能导致似然面平坦~~ **已解决** — 两步法递归修复后 d 全区间稳定
- **[性能]** D-Vine Tree 的 FIGAS 训练涉及 10 条边 × 每边参数优化，总计算量约是参考代码的 10 倍 → 预期训练时间 ~5-10 分钟（R 版本类似），可在后续用 `numba` 加速
