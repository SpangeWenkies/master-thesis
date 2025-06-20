import numpy as np
from typing import Iterable, Dict
from .structure_defs import DiffKey

def div_by_stdev(name, vec):
    """Standardize a vector by its standard deviation."""
    if vec.size == 0 or not np.isfinite(vec).all():
        print(f"Warning: vector '{name}' has non-finite values or is empty")
        return vec  # or return np.full_like(vec, np.nan)

    std = vec.std()
    if std == 0:
        print(f"Warning: Standard deviation is zero for vector '{name}' — possibly constant values.")
        return vec  # or np.full_like(vec, np.nan)

    return vec / std

def make_score_dicts(diffs: dict, keys: Iterable[DiffKey], score_names: Iterable[str]) -> Dict[DiffKey, Dict[str, np.ndarray]]:
    """Organize difference vectors into nested dictionaries"""
    score_dicts: Dict[DiffKey, Dict[str, np.ndarray]] = {}
    for key in keys:
        score_dicts[key] = {
            score: diffs[score][key]
            for score in score_names
            if key in diffs[score]
        }
    return score_dicts

def t_test_per_replication(diff_matrix):
    """Return p-values from t-tests applied row-wise.

    Parameters
    ----------
    diff_matrix : array-like, shape (reps, n)
        Each row contains the score differences for a single replication.

    Returns
    -------
    numpy.ndarray
        Array of p-values for the null of zero mean difference per replication.
    """
    from scipy.stats import ttest_1samp

    diff_matrix = np.asarray(diff_matrix)
    res = ttest_1samp(diff_matrix, popmean=0, axis=1, nan_policy="omit")
    return res.pvalue


def perform_size_tests(p_values, alpha_grid=None):
    """Compute size-discrepancy curve using per-replication p-values.

    Parameters
    ----------
    p_values : array-like
        P-values from individual replications.
    alpha_grid : array-like, optional
        Significance levels over which to evaluate the test size.
    """
    if alpha_grid is None:
        alpha_grid = np.linspace(0.01, 0.2, 20)

    p_values = np.asarray(p_values)
    # For each alpha compute the empirical rejection rate.  The comprehension
    # returns a vector of the same length as ``alpha_grid``.
    rejection_rates = np.array([(p_values < a).mean() for a in alpha_grid])
    discrepancies = rejection_rates - alpha_grid
    return alpha_grid, rejection_rates, discrepancies
