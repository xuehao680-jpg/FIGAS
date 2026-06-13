#!/usr/bin/env python3
"""Fix vine_copula.py pyvinecopulib integration."""

with open('/mnt/d/zcc/PythonCode/vine_copula.py', 'r') as f:
    content = f.read()

# Find the pyvinecopulib block and replace it
old_marker = '        import pyvinecopulib as pv\n        print("=== 正在自动构建最优 Vine Copula 结构 (pyvinecopulib) ... ===")\n        results = {}\n        family_set = [1, 2, 3, 4, 5, 10]'

# Find where the simplified fallback starts
fallback_marker = '        print("pyvinecopulib 不可用或出错'

idx_start = content.find(old_marker)
idx_fallback = content.find(fallback_marker, idx_start)

if idx_start < 0 or idx_fallback < 0:
    print('ERROR: Cannot find markers')
    print('idx_start:', idx_start, 'idx_fallback:', idx_fallback)
    exit(1)

new_block = '''        import pyvinecopulib as pv\n        print("=== 正在自动构建最优 Vine Copula 结构 (pyvinecopulib) ... ===")\n        results = {}\n        family_set = [1, 2, 3, 4, 5, 10]\n        pobs_data = pv.to_pseudo_obs(pobs)\n        structure_names = ["R-Vine (无约束)", "C-Vine (星型)", "D-Vine (链式)"]\n\n        for vtype, title in zip(["RVine", "CVine", "DVine"], structure_names):\n            print(f"  --- {title} 选择中 ... ---")\n            ctrl = pv.FitControlsVinecop(\n                family_set=family_set,\n                select_truncation=True,\n                select_threshold=0.05,\n                show_trace=False\n            )\n            if vtype == "CVine":\n                struct = pv.CVineStructure(list(range(n_cols)))\n                vc = pv.Vinecop(data=pobs_data, controls=ctrl, structure=struct)\n            elif vtype == "DVine":\n                struct = pv.DVineStructure(list(range(n_cols)))\n                vc = pv.Vinecop(data=pobs_data, controls=ctrl, structure=struct)\n            else:  # RVine\n                vc = pv.Vinecop(data=pobs_data, controls=ctrl)\n            \n            ll = vc.get_loglik()\n            aic_val = vc.get_aic()\n            bic_val = vc.get_bic()\n            results[vtype] = {\"logLik\": ll, \"AIC\": aic_val, \"BIC\": bic_val}\n            print(f"    {vtype}: LogLik={ll:.2f}, AIC={aic_val:.2f}, BIC={bic_val:.2f}")\n'''

content = content[:idx_start] + new_block + content[idx_fallback:]

with open('/mnt/d/zcc/PythonCode/vine_copula.py', 'w') as f:
    f.write(content)

print('vine_copula.py updated with correct pyvinecopulib API')
