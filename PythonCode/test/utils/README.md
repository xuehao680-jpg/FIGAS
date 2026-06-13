# Test: utils.py

数据加载、描述性统计、诊断检验（JB、ADF、ARCH-LM、Ljung-Box）、
ARMA-GARCH 阶数选择和辅助函数。

## 测试类

| 类 | 内容 |
|----|------|
| `TestSignifStars` | 在 p=0.01, 0.05, 0.10 边界处的显著性星号 |
| `TestArchLmCustom` | **np.roll bug 检测** — 与正确实现对比 |
| `TestLoadAndSplitData` | 临时 CSV 的 80/20 切分、时间顺序、列数 |
| `TestDescriptiveStats` | 输出形状、列名、峰度≥3、标准差>0 |
| `TestPlotReturnSeries` | PNG 文件保存和空列表不崩溃 |
| `TestDiagnosticTests` | JB + ADF 返回 DataFrame、ADF 拒绝单位根 |
| `TestLjungBox` | 形状匹配 (n_lags × n_assets)、p 值在 [0,1] |
| `TestArmaGarchOrderSelection` | 返回 dict 结构、GARCH 阶数在候选集中 |

### 已知 Bug：`arch_lm_test_custom` 的 `np.roll`
- `test_np_roll_is_wrong_for_arch_lm` 直接证明 np.roll 错误地用末尾值填充
  而非 NaN，导致前几行的回归矩阵包含无效回绕值。
