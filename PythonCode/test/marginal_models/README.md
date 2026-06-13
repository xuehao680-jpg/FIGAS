# Test: marginal_models.py

GJR-GARCH(1,1)-t 拟合、标准化残差提取、PIT 变换、残差诊断、
OOS 测试集过滤和模型持久化。

## 测试类

| 类 | 内容 |
|----|------|
| `TestArchLm` | **正确版** ARCH-LM（np.concatenate+NaN），白噪声/ARCH 效应 |
| `TestFitGjrGarch` | 返回所有资产、条件波动率>0、参数有限 |
| `TestExtractStdResiduals` | 形状、有限值、nu>2、均值≈0 |
| `TestPitTransform` | PIT ∈ (0,1)、KS 检验≈Uniform |
| `TestDiagnostics` | 残差诊断 + PIT 直方图的冒烟测试 |
| `TestFilterTestSet` | OOS 形状、PIT ∈ (0,1)、训练/测试 PIT 不重复 |
| `TestSaveMarginalOutputs` | CSV + PKL 写入、pickle bundle 键集合 |
| `TestMarginalPipeline` | 端到端 pipeline 冒烟测试 |

### 已知风险
- MA 阶数被静默丢弃（`_ma_order` 未使用）—— 无测试可弥补此问题，
  因为 Python arch 库不支持均值方程中的 MA。
