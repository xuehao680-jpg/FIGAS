with open('/mnt/d/zcc/PythonCode/figas_filter.py') as f:
    c = f.read()

# Find and fix: add n_h_fallback=0 before the loop, and increment inside
old = '    h1 = np.zeros(T_len)\n    h2 = np.zeros(T_len)\n    for t in range(T_len):\n        h1[t], h2[t] = _safe_bicop_hfunc(u1[t], u2[t], family=fam_id,\n                                           par=par_t[t], par2=kappa)'

new = '    h1 = np.zeros(T_len)\n    h2 = np.zeros(T_len)\n    n_h_fallback = 0\n    for t in range(T_len):\n        h1[t], h2[t] = _safe_bicop_hfunc(u1[t], u2[t], family=fam_id,\n                                           par=par_t[t], par2=kappa)\n        if h1[t] == u1[t] and h2[t] == u2[t]:\n            n_h_fallback += 1'

if old in c:
    c = c.replace(old, new)
    open('/mnt/d/zcc/PythonCode/figas_filter.py','w').write(c)
    print('Fixed h-function counter')
else:
    # Check if already fixed
    if 'n_h_fallback = 0' in c and 'n_h_fallback += 1' in c:
        print('Already fixed')
    else:
        print('Pattern not found - showing context:')
        idx = c.find('h1 = np.zeros(T_len)')
        print(c[idx:idx+300])
