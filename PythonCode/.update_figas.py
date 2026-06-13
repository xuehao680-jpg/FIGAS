#!/usr/bin/env python3
"""Update estimate_figas_params to use Optuna Bayesian optimization."""
import re

with open('/mnt/d/zcc/PythonCode/figas_filter.py', 'r') as f:
    content = f.read()

# New function body using Optuna
new_func = '''
def estimate_figas_params(u1, u2, fam_id, verbose=True):
    """
    Estimate FIGAS parameters via Optuna Bayesian optimization.
    d is constrained to (0, 0.5) per domain rule.
    """
    import optuna

    u1a, u2a = np.asarray(u1, float).ravel(), np.asarray(u2, float).ravel()

    # Static initial estimates
    start_rho, start_kappa = _static_copula_fit(u1a, u2a, fam_id)
    start_mu = _inverse_link(fam_id, start_rho)

    if verbose:
        suffix = f", kappa={start_kappa:.2f}" if fam_id == 2 else ""
        print(f"  [FIGAS] family={fam_id}, static_par={start_rho:.4f}{suffix}")

    def objective(trial):
        mu = trial.suggest_float('mu', -20, 20)
        alpha = trial.suggest_float('alpha', 0.001, 0.5)
        beta = trial.suggest_float('beta', 0.001, 0.999)
        d = trial.suggest_float('d', 0.001, 0.499)  # must be in (0, 0.5)

        n_pars = 4
        theta = [mu, alpha, beta, d]
        if fam_id == 2:
            kappa = trial.suggest_float('kappa', 2.1, 50.0)
            theta.append(kappa)
            n_pars = 5

        try:
            tmp = filter_figas(np.array(theta), u1a, u2a, fam_id)
            ll = tmp.get('loglik', -1e10)
            return float(ll) if np.isfinite(ll) else float(-1e10)
        except Exception:
            return float(-1e10)

    n_trials = 80 if fam_id == 2 else 60

    study = optuna.create_study(direction='maximize')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params_list = [study.best_params['mu'], study.best_params['alpha'],
                        study.best_params['beta'], study.best_params['d']]
    if fam_id == 2:
        best_params_list.append(study.best_params['kappa'])
    best = np.array(best_params_list)

    fres = filter_figas(best, u1a, u2a, fam_id)

    if verbose:
        print(f"    FIGAS best: mu={best[0]:.4f}, a={best[1]:.4f}, "
              f"b={best[2]:.4f}, d={best[3]:.4f}" +
              (f", k={best[4]:.2f}" if fam_id == 2 else ""))
        print(f"    FIGAS loglik: {fres['loglik']:.2f} (n={len(u1a)})")

    return best, fres
'''

# Replace old function
pattern = r'def estimate_figas_params\(.*?(?=\n# =+\n#  )'
old_match = re.search(r'def estimate_figas_params\(.*?(?=\n# [=\-]{10,})', content, re.DOTALL)

if old_match:
    new_content = content[:old_match.start()] + new_func + '\n' + content[old_match.end():]
    with open('/mnt/d/zcc/PythonCode/figas_filter.py', 'w') as f:
        f.write(new_content)
    print('FIGAS estimation function updated successfully.')
else:
    print('ERROR: Could not find old function to replace.')
