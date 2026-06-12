## Why

毕业论文的 R 代码（`Rcode/代码(1).txt`）实现了 D-Vine-FIGAS Copula 模型来做一带一路沿线国家股市风险溢出分析，但代码中存在诸多问题：路径硬编码、R 语言运行速度慢、可维护性差。需要将其重构为 Python，利用 Python 生态（`arch`、`scipy`、`numpy`）实现更快、更模块化、更易维护的版本，同时保持与原 R 代码的学术方法论一致。

## What Changes

- **新建** Python 模块化代码结构，替代单一 R 脚本（1074 行）
- **新建** `marginal_models.py`: GJR-GARCH + Student-t 边缘分布拟合（替代 R 的 `rugarch` 调用）
- **新建** `vine_copula.py`: D-Vine Copula 静态/动态（GAS、FIGAS）建模（替代 R 的 `VineCopula` 调用）
- **新建** `figas_filter.py`: FIGAS(1,1,d) 分数阶积分 GAS 过滤函数，含 t/Clayton/Gumbel/Survival Gumbel 多种 Copula 族的得分函数与 Fisher 信息
- **新建** `gas_filter.py`: 标准 GAS(1,1) 过滤函数作为对比基准
- **新建** `utils.py`: 描述性统计、JB/ADF/ARCH-LM/Ljung-Box 检验、PIT 诊断、可视化
- **新建** `config.py`: 统一管理路径、模型参数、随机种子
- **新建** `main.py`: 主流程编排脚本（数据读取→边缘建模→Copula 选择→FIGAS 动态建模→OOS 评估）
- **新建** `risk_measures.py`: CoVaR 和风险溢出指数计算（论文最终输出）
- **复用** 参考代码 `D:\SCI投稿论文\model\gas_copula_simulation.py` 中的 GAS-Copula 得分函数推导和 Fisher 信息计算

## Capabilities

### New Capabilities
- `marginal-garch`: GJR-GARCH(1,1) + Student-t 分布边缘建模，含训练集拟合、检验集过滤、标准化残差提取、PIT 变换
- `vine-copula-select`: R-Vine/C-Vine/D-Vine 三种结构自动选择（按 AIC/BIC），含 Kendall's tau 矩阵、TSP 最优顺序
- `figas-copula-dynamic`: FIGAS(1,1,d) 分数阶积分 GAS 驱动的动态 D-Vine Copula 参数估计，支持 Gaussian/t/Clayton/Gumbel/Survival Gumbel/旋转 Clayton 族
- `gas-copula-dynamic`: GAS(1,1) 驱动的动态 D-Vine Copula（对比基准）
- `oos-evaluation`: 样本外(OOS)评估框架，训练集/测试集对数似然比较
- `descriptive-statistics`: 描述性统计、平稳性检验、ARCH效应检验、白噪声检验

### Modified Capabilities
<!-- No existing specs to modify -->

## Impact

- **语言迁移**: R → Python 3.10+，依赖 `arch`, `scipy`, `numpy`, `pandas`, `pyvinecopula`, `matplotlib`
- **代码结构**: 单文件 1074 行 → 多模块 ~1500 行（含完整注释和文档字符串）
- **输出兼容**: 生成的中间文件（CSV、RData → Pickle/HDF5）路径统一在 `data/` 目录下
- **参考代码**: 复用 `D:\SCI投稿论文\model\gas_copula_simulation.py` 中的 GAS-Copula 理论实现，扩展到 5 变量 D-Vine + FIGAS
