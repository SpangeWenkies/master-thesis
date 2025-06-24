import numpy as np
from typing import Iterable, Dict
from .structure_defs import DiffKey

def div_by_stdev(name: str, vec: np.ndarray) -> np.ndarray:
    """Standardize ``vec`` by its standard deviation.

    Parameters
    ----------
    name : str
        Identifier used in warning messages.
    vec : ndarray of shape (n,)
        Data vector to standardize.

    Returns
    -------
    ndarray of shape (n,)
        Standardized vector.
    """
    if vec.size == 0 or not np.isfinite(vec).all():
        print(f"Warning: vector '{name}' has non-finite values or is empty")
        return vec  # or return np.full_like(vec, np.nan)

    std = vec.std()
    if std == 0:
        print(f"Warning: Standard deviation is zero for vector '{name}' — possibly constant values.")
        return vec  # or np.full_like(vec, np.nan)

    return vec / std

def make_score_dicts(
    diffs: dict,
    keys: Iterable[DiffKey],
    score_names: Iterable[str],
) -> Dict[DiffKey, Dict[str, np.ndarray]]:
    """Organize difference vectors into nested dictionaries.

    Parameters
    ----------
    diffs : dict
        Mapping ``score_name`` -> ``{DiffKey: array}``.
    keys : Iterable[DiffKey]
        Keys identifying the differences to collect.
    score_names : Iterable[str]
        Names of the scores stored in ``diffs``.

    Returns
    -------
    Dict[DiffKey, Dict[str, ndarray]]
        Nested mapping ``DiffKey -> {score_name: array}``.
    """
    score_dicts: Dict[DiffKey, Dict[str, np.ndarray]] = {}
    for key in keys:
        score_dicts[key] = {
            score: diffs[score][key]
            for score in score_names
            if key in diffs[score]
        }
    return score_dicts

def t_test_per_replication(diff_matrix: np.ndarray) -> np.ndarray:
    """Row-wise t-tests of zero mean difference.

    Parameters
    ----------
    diff_matrix : ndarray of shape (reps, n)
        Each row contains the score differences for one replication.

    Returns
    -------
    ndarray of shape (reps,)
        P-values for ``H0: mean(diff) = 0`` per replication.
    """
    from scipy.stats import ttest_1samp

    diff_matrix = np.asarray(diff_matrix)
    res = ttest_1samp(diff_matrix, popmean=0, axis=1, nan_policy="omit")
    return res.pvalue


def perform_size_tests(p_values: np.ndarray, alpha_grid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute size-discrepancy curve using per-replication p-values.

    Parameters
    ----------
    p_values : ndarray of shape (reps,)
        P-values from individual replications.
    alpha_grid : ndarray of shape (m,), optional
        Significance levels over which to evaluate the test size. If ``None``,
        a default grid between 0.01 and 0.2 is used.

    Returns
    -------
    tuple of ndarray
        ``(alpha_grid, rejection_rates, discrepancies)`` each of shape ``(m,)``.
    """
    if alpha_grid is None:
        alpha_grid = np.linspace(0.01, 0.2, 20)

    p_values = np.asarray(p_values)
    # For each alpha compute the empirical rejection rate.  The comprehension
    # returns a vector of the same length as ``alpha_grid``.
    rejection_rates = np.array([(p_values < a).mean() for a in alpha_grid])
    discrepancies = rejection_rates - alpha_grid
    return alpha_grid, rejection_rates, discrepancies

def dm_statistic(diff_vec: np.ndarray) -> float:
    """Compute the Diebold--Mariano test statistic for ``diff_vec``.

    Parameters
    ----------
    diff_vec : ndarray of shape (n,)
        Difference vector of two score sequences.

    Returns
    -------
    float
        Diebold--Mariano statistic based on the sample mean and variance.
    """
    diff_vec = np.asarray(diff_vec)
    if diff_vec.size == 0 or not np.isfinite(diff_vec).all():
        print("Warning: DM statistic undefined for empty or non-finite vector")
        return float("nan")

    variance = diff_vec.var(ddof=1)
    if variance == 0:
        print("Warning: variance is zero in DM statistic computation")
        return float("nan")

    return float(diff_vec.mean() / np.sqrt(variance / diff_vec.size))
