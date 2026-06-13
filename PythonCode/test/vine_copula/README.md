# Test: vine_copula.py

Vine copula 结构选择、Kendall tau 矩阵、D-Vine 训练（静态/FIGAS/GAS）、
OOS 评估和模型比较。

## 测试类

| 类 | 内容 |
|----|------|
| `TestKendallTauMatrix` | 形状、对角=1、对称性、弱/强相关验证、DataFrame 输入 |
| `TestSelectVineStructure` | 返回 DataFrame、AIC 有限、小样本拒绝 |
| `TestDVineStructure` | 4 棵树、边计数 [4,3,2,1]、键完整性、族代码支持 |
| `TestTrainDvineStatic` | total_loglik/edges 键、4 棵树、loglik 有限、h1/h2 ∈ (0,1) |
| `TestTrainDvineFigas` | FIGAS 训练有限 loglik、par_seq 存在 |
| `TestTrainDvineGas` | GAS 训练有限 loglik |
| `TestEvalTestDvine` | 静态/FIGAS/GAS OOS loglik 有限、非极端 |
| `TestCompareModels` | best_name ∈ {static,figas,gas}、结果 dict 键、3 行比较表 |
| `TestEdgeCases` | 无效 model_type→ValueError、错误列数、常量输入 |

### 已知风险
- `DVINE_STRUCTURE` 是硬编码的，不从数据中选择
- `select_vine_structure` 中的 pyvinecopulib 路径在当前环境中不可用，
  始终走 fallback 简化 D-Vine 路径
