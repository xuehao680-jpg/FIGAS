#!/usr/bin/env python3
with open('/mnt/d/zcc/PythonCode/vine_copula.py', 'r') as f:
    content = f.read()

old = 'print(f"pyvinecopulib 不可用或出错 ({e}), 使用简化 D-Vine 结构选择.")'

new = '''        comparison = pd.DataFrame([
            {"Model": "R-vine", "LogLik": results["RVine"]["logLik"],
             "AIC": results["RVine"]["AIC"], "BIC": results["RVine"]["BIC"]},
            {"Model": "C-vine", "LogLik": results["CVine"]["logLik"],
             "AIC": results["CVine"]["AIC"], "BIC": results["CVine"]["BIC"]},
            {"Model": "D-vine", "LogLik": results["DVine"]["logLik"],
             "AIC": results["DVine"]["AIC"], "BIC": results["DVine"]["BIC"]},
        ])
    except Exception as e:
        print(f"pyvinecopulib 不可用或出错 ({e}), 使用简化 D-Vine 结构选择.")'''

if old in content:
    content = content.replace(old, new)
    with open('/mnt/d/zcc/PythonCode/vine_copula.py', 'w') as f:
        f.write(content)
    print('Fixed!')
else:
    print('ERROR: old string not found')
    # Print context
    idx = content.find('pyvinecopulib')
    if idx >= 0:
        for i in range(max(0,idx-50), min(len(content), idx+100)):
            print(content[i], end='' if content[i] != '\n' else '\n')
