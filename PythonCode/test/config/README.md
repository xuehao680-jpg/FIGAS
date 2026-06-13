# Test: config.py

验证配置常量的类型、路径解析、优化边界、数值稳定性常量和 `ensure_dirs`。

## 测试类

| 类 | 内容 |
|----|------|
| `TestConstants` | SEED、TRAIN_RATIO、ASSETS、GARCH_SPECS 的存在性和类型 |
| `TestPaths` | PROJECT_ROOT、DATA_DIR、CSV/PKL 文件名模式 |
| `TestBounds` | FIGAS_BOUNDS/GAS_BOUNDS 的键和取值范围 |
| `TestNumericalStability` | PDF_FLOOR >0、SCORE_CLAMP >0、F_DIFF_H 合理性 |
| `TestEnsureDirs` | 目录创建和幂等性（使用 tmp_path + monkeypatch） |
