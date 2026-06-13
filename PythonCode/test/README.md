# GARCH-Copula Test Suite

针对 FIGAS 项目的 pytest 测试套件。覆盖 config、utils、marginal_models、
figas_filter、gas_filter、vine_copula 六个模块。

---

## 目录结构

```
test/
├── conftest.py                       # 共享 fixtures（合成数据生成器）
├── config/
│   ├── test_config.py                # 常量类型、路径、边界、ensure_dirs
│   └── README.md
├── utils/
│   ├── test_utils.py                 # 数据加载、统计量、诊断检验、np.roll bug 检测
│   └── README.md
├── marginal_models/
│   ├── test_marginal_models.py       # GJR-GARCH 拟合、残差提取、PIT、OOS 过滤
│   └── README.md
├── figas_filter/
│   ├── test_figas_filter.py          # 分数权重、Copula PDF/h-func、FIGAS 滤波、参数估计
│   └── README.md
├── gas_filter/
│   ├── test_gas_filter.py            # GAS 滤波、参数估计、与 FIGAS 一致性
│   └── README.md
├── vine_copula/
│   ├── test_vine_copula.py           # Kendall tau、D-Vine 训练、OOS 评估、模型比较
│   └── README.md
├── pyproject.toml                    # pytest 配置
└── README.md                         # 本文档
```

---

## 运行方式

```bash
cd /mnt/d/zcc/PythonCode

# 运行全部测试（约 3-5 分钟，含 FIGAS/GAS 优化）
python -m pytest

# 仅运行快速测试（跳过 FIGAS/GAS 参数估计耗时测试）
python -m pytest -m "not slow"

# 仅运行单个模块
python -m pytest test/figas_filter/

# 仅运行单个测试类
python -m pytest test/utils/test_utils.py::TestSignifStars

# 显示详细输出
python -m pytest -v

# 运行所有测试（含 print 输出）并报告最长耗时测试
python -m pytest -v --durations=10
```

---

## 测试策略

### 数据依赖
所有测试使用 `conftest.py` 中的 **合成数据**（Gaussian、Clayton、t-Copula），
不依赖外部 CSV 文件。保证可重现性和 CI 兼容性。

### 分模块要点

| 模块 | 核心测试点 | 已知风险 |
|------|-----------|----------|
| **config** | 常量类型、路径、边界有效性、ensure_dirs 幂等性 | — |
| **utils** | `_signif_stars` 边界、**`arch_lm_test_custom` 的 np.roll bug 检测**、描述统计、诊断检验 | `np.roll` 导致 ARCH-LM 统计量错误 |
| **marginal_models** | GARCH 拟合、条件波动率 >0、PIT ∈ (0,1)、OOS 过滤一致性 | MA 项被静默丢弃 |
| **figas_filter** | 分数权重已知值、Copula PDF 解析值、h-func 范围、FIGAS 优于静态 | L=100 硬编码、异常吞噬 |
| **gas_filter** | 滤波输出完整性、参数边界、与 FIGAS 链接函数一致性 | 与 FIGAS 代码重复 |
| **vine_copula** | Kendall tau 对称性、DVINE_STRUCTURE 校验、OOS loglik 有限性 | 结构硬编码不依赖数据 |

### FIGAS/GAS 优化测试注意事项

`estimate_figas_params` 和 `estimate_gas_params` 调用 L-BFGS-B 优化，
在合成数据上通常需 50-150 次迭代。测试使用 `verbose=False` 抑制输出，
并使用中小样本（200-400 obs）控制耗时。

---

## 覆盖率重点

测试已覆盖的风险点（来自代码审计）：

1. **`utils.py:arch_lm_test_custom` 的 np.roll bug**  
   `test_np_roll_is_wrong_for_arch_lm` 直接演示 np.roll 与 NaN-padding 的差异。

2. **MA 项丢弃**  
   `test_marginal_models.py` 的 pipeline 测试验证 GARCH 拟合能运行，
   但不验证 MA 阶数被正确传入（当前代码忽略 `_ma_order`）。

3. **异常默认兜底**  
   `test_figas_filter.py` 的极端参数测试确保 `_safe_bicop_hfunc` 在
   无效族代码时回退到独立 copula。

4. **参数边界合规**  
   `test_config.py` 验证 FIGAS_BOUNDS 和 GAS_BOUNDS 的上下限值。

---

## 添加新测试

```python
# 1. 在相应模块下的 test_*.py 中添加
# 2. 使用 conftest.py 中的共享 fixtures
# 3. 避免依赖外部 CSV / 网络
# 4. 耗时测试加 @pytest.mark.slow 标记

@pytest.mark.slow
def test_expensive_optimization(self, clayton_synthetic):
    ...
```
