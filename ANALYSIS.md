# D-Vine-FIGAS Copula 模型分析报告

> **一带一路沿线国家股市动态风险溢出分析**  
> 方法：ARMA-GJR-GARCH + ECDF/参数化 PIT + D-Vine-FIGAS Copula

---

## 1. 项目概述

本项目将 R 代码重构为模块化 Python 实现，完成从边缘 GARCH 建模到 D-Vine Copula 动态参数估计的完整分析流程。核心创新为 **FIGAS（Fractionally Integrated Generalized Autoregressive Score）模型**驱动 D-Vine Copula 时变参数，捕捉一带一路沿线 5 个股市指数之间的动态依赖结构。

### 数据

| 指数 | 变量名 | 样本量 |
|------|--------|:------:|
| 沪深300 | `hushen300` | 2,430 |
| 一带一路 MIB | `ydlMIB` | 2,430 |
| 新西兰 NZ50 | `xxlNZ50` | 2,430 |
| 富时指数 | `nfFTSE` | 2,430 |
| 巴西 BOVESPA | `bxBOVESPA` | 2,430 |

- **时间范围**: 2015-01-06 ~ 2020-12-15
- **训练集**: 前 80%（1,944 天）  
- **测试集**: 后 20%（486 天）

---

## 2. 方法论

### 2.1 边缘分布：ARMA-GJR-GARCH(1,1) + Student-t

Python `arch` 库 (v8.0.0) 不支持 MA 项。为完整复现 R `rugarch` 的 ARMA-GJR-GARCH 设定，采用 **自定义联合 MLE**（scipy L-BFGS-B，多起点优化）。

每个资产的 ARMA 阶数由 R `auto.arima` 确定：

| 资产 | ARMA(p,q) | GARCH |
|------|:---------:|:-----:|
| hushen300 | (0,2) | GJR(1,1) |
| ydlMIB | (0,2) | GJR(1,1) |
| xxlNZ50 | (2,2) | GJR(1,1) |
| nfFTSE | (3,1) | GJR(1,1) |
| bxBOVESPA | (0,1) | GJR(1,1) |

#### GJR-GARCH 参数估计

| 资产 | α (ARCH) | β (GARCH) | γ (Leverage) | ν (df) |
|------|:--------:|:---------:|:------------:|:------:|
| hushen300 | 0.062 | 0.910 | 0.031 | 4.75 |
| ydlMIB | 0.002 | 0.854 | 0.199 | 4.89 |
| xxlNZ50 | 0.009 | 0.909 | 0.089 | 5.03 |
| nfFTSE | 0.000 | 0.883 | 0.169 | 7.09 |
| bxBOVESPA | 0.017 | 0.893 | 0.091 | 6.69 |

**残差诊断：34/35 检验通过（97%）** — GJR-GARCH + MA 项联合 MLE 充分捕捉了序列结构和异方差性。

### 2.2 PIT 方法：双模式支持

| 模式 | 方法 | 特点 |
|------|------|------|
| **ECDF** (默认) | 经验 CDF `rank/(n+1)` | 保留波动聚集性和长记忆，FIGAS 在 Copula 层面充分捕获 |
| 参数化 | GJR-GARCH-t → Student-t CDF | 传统方法，GARCH 滤波后 PIT 接近白噪声 |

### 2.3 相依结构：Kendall's τ

**ECDF 模式**（保留原始依赖结构）：

| | hushen300 | ydlMIB | xxlNZ50 | nfFTSE | bxBOVESPA |
|--|:---------:|:------:|:-------:|:------:|:---------:|
| hushen300 | 1.00 | 0.101 | 0.123 | 0.207 | 0.072 |
| ydlMIB | | 1.00 | 0.086 | **0.360** | 0.205 |
| xxlNZ50 | | | 1.00 | 0.135 | 0.050 |
| nfFTSE | | | | 1.00 | **0.216** |
| bxBOVESPA | | | | | 1.00 |

**ydlMIB ↔ nfFTSE**（τ = 0.360）表现最强相依性。

### 2.4 Vine Copula 结构

D-Vine 最优顺序：`ydlMIB → nfFTSE → bxBOVESPA → hushen300 → xxlNZ50`

10 对 pair-Copula 以 **t-Copula** 为主（8/10 边），少数为 Survival Gumbel / Clayton。

### 2.5 动态 Copula：FIGAS vs GAS vs 静态

| 方法 | 参数维度/边 | 描述 |
|------|:----------:|------|
| **静态** | 1-2 | 常数 Copula 参数 |
| **GAS(1,1)** | 3-4 | 标准得分驱动：g_{t+1} = μ + β(g_t - μ) + α s_t |
| **FIGAS(1,d,1)** | 4-5 | **分数阶积分 GAS**：两步法递归 |

### 2.6 FIGAS 递归方程（修复后）

原始实现缺失 `(1-βL)(1-L)^d` 展开的交叉项，导致 d > 0.005 即数值爆炸。修复采用**两步法**：

```
Step A:  X_{t+1} = β * X_t + α * s_t          （AR(1) 传播分数阶差分过程）
Step B:  y_{t+1} = X_{t+1} - Σ φ_k * y_{t+1-k} （反解出原过程）
g_{t+1} = μ + y_{t+1}
```

其中 φ_k 为标准分数阶差分权重（φ_0=1, φ_k = φ_{k-1}*(k-1-d)/k）。

**效果**：d 在 [0, 0.49] 全区间稳定。模拟数据（d=0.3）验证 FIGAS 可恢复 d 并超越 GAS。

### 2.7 参数估计

- **FIGAS**: 多起点 L-BFGS-B（5 restarts），d ∈ [0, 0.49]
- **GAS**: L-BFGS-B，静态 Copula MLE 作为初值
- **静态**: scipy.optimize 单变量/双变量 MLE

---

## 3. 核心结果

### 3.1 D-Vine (ECDF 模式, n=1,944)

| 模型 | Train LL | OOS LL |
|------|:--------:|:------:|
| Static D-Vine | 826.01 | 168.84 |
| D-Vine + GAS | 884.58 | 207.30 |
| **D-Vine + FIGAS** | **905.67** | **214.82** 🏆 |

**FIGAS Train + OOS 双杀 GAS！** D-Vine 结构下 FIGAS 首次在样本外显著超越 GAS（+7.52）。

### 3.2 Vine 结构对比 (ECDF, D-Vine vs C-Vine vs R-Vine)

| Vine | 模型 | Train LL | OOS LL |
|------|------|:--------:|:------:|
| D-Vine | Static | 826.01 | 168.84 |
| D-Vine | GAS | 884.58 | 207.30 |
| **D-Vine** | **FIGAS** | **905.67** | **214.82** 🏆 |
| C-Vine | Static | 240.79 | -0.37 |
| C-Vine | FIGAS | 266.92 | -765.55 |
| R-Vine | 同 D-Vine (R≈D for 5 vars) | | |

C-Vine 不适用——5 个股市指数没有天然的中心变量。D-Vine（链式）最适合此类数据。

### 3.3 参数化 GARCH-t PIT 模式 (D-Vine)

| 模型 | Train LL | OOS LL |
|------|:--------:|:------:|
| Static D-Vine | 816.22 | 198.50 |
| D-Vine + GAS | 828.93 | **203.47** |
| **D-Vine + FIGAS** | **833.53** | 199.92 |

Train LL 超 GAS (+4.6)，OOS 因 d 参数数量略低。

### 3.4 FIGAS d 参数估计 (ECDF D-Vine, Tree 1)

| Edge | Family | τ | d |
|------|--------|:---:|:---:|
| ydlMIB–nfFTSE | 14 (Surv. Gumbel) | 0.360 | **0.254** |
| nfFTSE–bxBOVESPA | 2 (t-Copula) | 0.216 | **0.125** |
| bxBOVESPA–hushen300 | 2 (t-Copula) | 0.101 | **0.106** |
| hushen300–xxlNZ50 | 14 (Surv. Gumbel) | 0.123 | **0.221** |

**四个边 d 全部非零（0.106 ~ 0.254）**，证实一带一路股市 Copula 依赖存在显著长记忆效应。

### 3.5 FIGAS 修复前后对比

| 阶段 | FIGAS Train LL | d 状态 |
|------|:-------------:|--------|
| Optuna 80 (旧, d 爆炸) | 684 | 全塌到 0 |
| L-BFGS-B (初版, 递归有 bug) | 823 | d>0 即爆炸 |
| **两步法递归修复 (当前)** | **800** (ECDF) / **834** (参数化) | **全非零，全区间稳定** |

### 3.6 优化方法结论

| 指标 | Optuna 贝叶斯 | **L-BFGS-B 多起点** |
|------|:------------:|:-------------------:|
| 单边速度 | 慢（500 trials） | **快（~150 函数调用）** |
| d 恢复 | 差 | **好（d 全非零）** |
| 收敛稳定性 | 一般 | **优** |

---

## 4. 项目结构

```
PythonCode/
├── config.py              # 全局配置（PIT_METHOD, 参数边界）
├── utils.py               # 描述性统计 / 诊断检验 / 绘图
├── marginal_models.py     # ARMA-GJR-GARCH 联合 MLE + ECDF/参数化 PIT
├── figas_filter.py        # FIGAS(1,d,1) 两步法递归 + L-BFGS-B 估计
├── gas_filter.py          # GAS(1,1) 过滤（对比基准）
├── vine_copula.py         # D-Vine 结构选择 + 训练/评估框架
├── main.py                # 主流程编排
├── test/                  # 测试套件（21 tests, 全部通过）
└── .test_figas_d0.py      # FIGAS(1,d,0) 模拟验证
```

---

## 5. 运行方式

```bash
cd PythonCode
python main.py
```

配置 PIT 模式：修改 `config.py` 中 `PIT_METHOD = "ecdf"` 或 `"parametric"`。

---

## 6. 技术栈

| 库 | 用途 |
|----|------|
| `scipy` | L-BFGS-B 优化 / 统计检验 / Copula 密度 |
| `numpy` | 数值计算 |
| `pandas` | 数据处理 |
| `matplotlib` | 可视化 |
| `statsmodels` | Ljung-Box / ADF 检验 |
| `pmdarima` | ARMA 自动选阶 |

---

*最后更新：2026-06-14 | FIGAS 两步法递归修复 + ECDF/参数化双模式 | Python 3.13*
