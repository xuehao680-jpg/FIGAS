#!/usr/bin/env python3
"""Fix figas_filter.py: add h-function fallback warning count."""

with open('/mnt/d/zcc/PythonCode/figas_filter.py', 'r') as f:
    content = f.read()

old = '''    # ----- 5. 计算 h 函数
    h1 = np.zeros(T_len)
    h2 = np.zeros(T_len)
    for t in range(T_len):
        h1[t], h2[t] = _safe_bicop_hfunc(u1[t], u2[t], family=fam_id,
                                           par=par_t[t], par2=kappa)'''

new = '''    # ----- 5. 计算 h 函数
    h1 = np.zeros(T_len)
    h2 = np.zeros(T_len)
    n_h_fallback = 0
    for t in range(T_len):
        h1[t], h2[t] = _safe_bicop_hfunc(u1[t], u2[t], family=fam_id,
                                           par=par_t[t], par2=kappa)
        if h1[t] == u1[t] and h2[t] == u2[t]:
            n_h_fallback += 1'''

count1 = content.count(old.strip())
if count1 == 0:
    print('Old string not found - checking variants...')
    # Try to find the actual text
    idx = content.find('h1 = np.zeros(T_len)')
    print(content[idx:idx+200])
else:
    content = content.replace(old.strip(), new.strip())
    print(f'Replaced {count1} occurrences')

# Add warning to return dict
old_ret = "return dict(\n        loglik=float(np.sum(ll_seq)),"
new_ret = "if n_h_fallback > 0:\n        import warnings\n        warnings.warn(f\"FIGAS filter: {n_h_fallback}/{T_len} h-function evaluations fell back to independence copula\", RuntimeWarning)\n    return dict(\n        loglik=float(np.sum(ll_seq)),"

count2 = content.count(old_ret)
if count2 == 0:
    print('Return string not found')
else:
    content = content.replace(old_ret, new_ret)
    print(f'Replaced return {count2} occurrences')

with open('/mnt/d/zcc/PythonCode/figas_filter.py', 'w') as f:
    f.write(content)

print('Done')
