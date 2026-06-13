# Test: figas_filter.py

FIGAS(1,d,1) 滤波器的核心实现：分数差分权重、内联 Copula PDF/h-functions、
静态 MLE、FIGAS 滤波器和完整参数估计。

## 测试类

| 类 | 内容 |
|----|------|
| `TestFracWeights` | 已知权重值（d=0.25, 前 4 个权重）、d=0 全零、求和≈0 |
| `TestBicopPdf` | Clayton 解析密度、对称性；t-copula 大 nu 极限；Gumbel；旋转族 | 
| `TestBicopHfunc` | 值域 (0,1)、族间旋转一致、数组输入 |
| `TestSafeBicopHfunc` | 正常操作、无效族兜底、极端参数 |
| `TestInverseLink` | rho=0→mu=0、双向可逆性 |
| `TestStaticCopulaFit` | Clayton/t/Survival Gumbel/90-rotated MLE 恢复已知参数 |
| `TestFilterFigas` | 输出键、有限 loglik、参数范围、长度匹配、alpha=beta=0→常量 |
| `TestEstimateFigasParams` | 收敛、参数在边界内、t-copula kappa 范围 |
| `TestFigasIntegration` | FIGAS 动态 loglik > 静态 loglik（时间变化 Clayton） |

### 数值验证
- `frac_weights(d=0.25, max_lag=4)` 与手工计算 `[1, -0.25, -0.09375, -0.0546875]` 一致
- Clayton PDF `c(0.5, 0.5; theta=2)` 与解析公式一致
