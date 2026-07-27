# quant_engine/optimization_lib.py
import numpy as np
from scipy.optimize import minimize, differential_evolution # Example optimizers
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV # For ML models later

def maximum_likelihood_estimation(log_likelihood_func, initial_guess, data, method='BFGS'):
    """Generic MLE using scipy.optimize.minimize (maximizing log-likelihood is minimizing -log-likelihood)."""
    def neg_log_likelihood(params):
        return -log_likelihood_func(params, data)

    result = minimize(neg_log_likelihood, initial_guess, method=method)
    if result.success:
        return result.x, result.fun # parameters, -log_likelihood_value
    else:
        raise RuntimeError(f"MLE failed: {result.message}")

def grid_search(objective_func, param_grid, data):
    """Generic grid search."""
    best_params = None
    best_score = float('inf') # Assuming minimization
    for params in param_grid:
        score = objective_func(params, data)
        if score < best_score:
            best_score = score
            best_params = params
    return best_params, best_score

def bayesian_optimization(objective_func, bounds, n_calls=50):
    """Example using differential evolution as a proxy for Bayesian Optimization (requires more sophisticated lib like scikit-optimize for true BO)."""
    def neg_objective(params):
         return -objective_func(params, data) # Assuming maximization, negate for minimization

    result = differential_evolution(neg_objective, bounds, maxiter=n_calls)
    if result.success:
        return result.x, -result.fun # parameters, objective_value
    else:
        raise RuntimeError(f"Optimization failed: {result.message}")

# Add more optimization functions as needed...
