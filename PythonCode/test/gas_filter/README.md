# Test: gas_filter.py

GAS(1,1) 滤波器和参数估计 — FIGAS 的基准对照模型。

## 测试类

| 类 | 内容 |
|----|------|
| `TestFilterGas` | 输出键完整性、有限 loglik、Clayton/t 参数范围、长度匹配、常量极限 |
| `TestEstimateGasParams` | Clayton + t-copula 估计收敛、边界合规 |
| `TestGasFigasConsistency` | 与 FIGAS 共享 `_inverse_link` 和 `_bicop_pdf` |
| `TestGasIntegration` | GAS 动态 loglik > 静态 loglik（时间变化 Clayton） |

### 一致性检查
GAS 和 FIGAS 使用来自 `figas_filter.py` 的相同链接函数和 Copula PDF：
- `test_link_function_identical` 验证 `_inverse_link` 结果一致
- `test_same_pdf_backend` 验证 `_bicop_pdf` 结果一致
