# D-Vine-FIGAS Copula 模型分析报告

> **一带一路沿线国家股市动态风险溢出分析**  
> 方法：GJR-GARCH(1,1) + Student-t 边缘建模 + D-Vine Copula 动态参数估计

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

### 2.1 边缘分布：GJR-GARCH(1,1) + Student-t

每个资产的收益率序列通过 **GJR-GARCH(1,1)** 建模，捕捉：
- 波动率聚集（GARCH 效应）
- 杠杆效应（GJR 非对称项）
- 厚尾分布（Student-t 新息）

标准化残差通过 **PIT（概率积分变换）** 映射到 Uniform(0,1)，作为 Copula 建模的输入。

#### GJR-GARCH 参数估计（训练集，5 个资产）

| 资产 | α (ARCH) | β (GARCH) | γ (Leverage) | ν (df) |
|------|:--------:|:---------:|:------------:|:------:|
| hushen300 | 0.062 | 0.911 | 0.031 | 4.74 |
| ydlMIB | 0.003 | 0.845 | 0.212 | 4.88 |
| xxlNZ50 | 0.004 | 0.917 | 0.087 | 5.16 |
| nfFTSE | 0.000 | 0.884 | 0.167 | 7.18 |
| bxBOVESPA | 0.017 | 0.890 | 0.095 | 6.72 |

#### 残差诊断

所有资产的标准化残差均通过 Ljung-Box 和 ARCH-LM 检验（p > 0.05），表明 GJR-GARCH 模型充分捕捉了序列结构和异方差性。

### 2.2 相依结构：Kendall's τ

|  | hushen300 | ydlMIB | xxlNZ50 | nfFTSE | bxBOVESPA |
|--|:---------:|:------:|:-------:|:------:|:---------:|
| hushen300 | 1.00 | 0.096 | 0.118 | 0.198 | 0.072 |
| ydlMIB | | 1.00 | 0.081 | **0.341** | 0.196 |
| xxlNZ50 | | | 1.00 | 0.127 | 0.055 |
| nfFTSE | | | | 1.00 | **0.207** |
| bxBOVESPA | | | | | 1.00 |

**ydlMIB ↔ nfFTSE**（τ = 0.341）表现出最强的相依性，反映了欧洲与一带一路市场的联动。

### 2.3 Vine Copula 结构选择

通过 TSP（旅行商问题）求解最优 D-Vine 变量顺序：

```
ydlMIB → nfFTSE → bxBOVESPA → hushen300 → xxlNZ50
```

10 对 pair-Copula 最优选型以 **t-Copula**（自由度高）为主，少数边为 Survival Gumbel / Clayton，兼顾上下尾依赖。

### 2.4 动态 Copula：FIGAS vs GAS vs 静态

比较三种动态参数化方法：

| 方法 | 描述 | 参数维度/边 |
|------|------|:----------:|
| **静态** | 常数 Copula 参数 | 1-2 |
| **GAS(1,1)** | 标准得分驱动更新 | 3-4 |
| **FIGAS(1,1,d)** | 分数阶积分 GAS | **4-5** |

FIGAS 的核心创新在于引入**分数阶差分参数 d ∈ (0, 0.5)**，使模型能捕捉长期记忆效应：

$$y_{t+1} = \beta y_t + \alpha s_t - \sum_{j=1}^{L} \psi_j y_{t+1-j}$$

其中 $\psi_j$ 为分数阶差分权重，$s_t$ 为缩放得分函数。

### 2.5 参数估计方法

FIGAS 参数通过 **Optuna 贝叶斯优化**（TPE 采样器）估计：

- 搜索空间：mu ∈ [-20, 20], α ∈ [0.001, 0.5], β ∈ [0.001, 0.999], **d ∈ (0, 0.5)**
- t-Copula 边：80 次试验 | 非 t-Copula 边：60 次试验
- 目标函数：最大化对数似然

相比传统 L-BFGS-B + 随机初值，贝叶斯优化避免了参数卡边界和局部最优问题。

---

## 3. 核心结果

### 3.1 模型比较

| 模型 | 训练集 LogLik | **测试集 OOS LogLik** | 排名 |
|------|:------------:|:--------------------:|:----:|
| 静态 D-Vine | 805.39 | -275.38 | 🥉 |
| D-Vine + GAS(1,1) | **818.50** | -214.34 | 🥈 |
| **D-Vine + FIGAS** | 791.41 | **-202.87** | 🥇 |

### 3.2 关键发现

1. **FIGAS 在样本外表现最优**（OOS LogLik = -202.87），优于 GAS（-214.34）和静态模型（-275.38）

2. **泛化能力**：FIGAS 在训练集上似然最低，但在测试集上最优 — 经典的 "Less is More" 模式。FIGAS 通过分数阶差分捕捉长期记忆，避免了 GAS 的过拟合倾向

3. **参数不卡边界**：Optuna 贝叶斯优化后，所有 FIGAS 参数均在合理范围内自然分布，d 参数服从(0, 0.5)的强约束

4. **Tree 层面的提升**：FIGAS 在 Tree 3（条件相依）上显著优于静态模型（LL=26.03 vs 12.10），说明动态建模在条件相依层面效果更明显

### 3.3 优化方法对比

| 指标 | L-BFGS-B（初版） | **Optuna 贝叶斯优化** |
|------|:-----------------:|:---------------------:|
| FIGAS Train LL | -12,874 | **791.41** |
| FIGAS OOS LL | -3,567 | **-202.87** |
| 参数卡边界 | 3/10 边 | **0 边** |
| d 分布 | 全在 0.05 边界 | **0.025 ~ 0.462** |

---

## 4. 项目结构

```
PythonCode/
├── config.py              # 全局配置（路径、参数边界）
├── utils.py               # 描述性统计 / 诊断检验 / 绘图
├── marginal_models.py     # GJR-GARCH + Student-t 边缘建模 + PIT
├── figas_filter.py        # FIGAS(1,1,d) 核心过滤 + Optuna 参数估计
├── gas_filter.py          # GAS(1,1) 过滤（对比基准）
├── vine_copula.py         # D-Vine 结构选择 + 训练/评估框架
├── main.py                # 主流程编排
├── requirements.txt       # 依赖清单
├── test/                  # 测试套件（173 快速 + 25 慢速测试）
└── test_imports.py        # 集成验证
```

---

## 5. 运行方式

```bash
pip install -r requirements.txt
python main.py
```

---

## 6. 技术栈

| 库 | 用途 |
|----|------|
| `arch` | GJR-GARCH 模型 |
| `scipy` | 统计检验 / Copula 密度 |
| `numpy` | 数值计算 |
| `pandas` | 数据处理 |
| `Optuna` | 贝叶斯参数优化 |
| `matplotlib` | 可视化 |
| `statsmodels` | Ljung-Box / ADF 检验 |
| `pmdarima` | ARMA 自动选阶 |

---

*生成日期：2026-06-13 | 基于 D-Vine-FIGAS Copula 模型 | Python 3.13*
