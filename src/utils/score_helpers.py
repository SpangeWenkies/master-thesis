import numpy as np

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

def extract_suffix_from_key(key, score_names):
    """
    Extracts the suffix (e.g. 'ecdf_f_g') from a key like 'CLS_diffs_ecdf_f_g'
    """
    for score in score_names:
        prefix = f"{score}_diffs_"
        if key.startswith(prefix):
            return key[len(prefix):]
    raise ValueError(f"Key '{key}' does not start with a known score prefix.")

def make_score_dicts(diffs, tag_suffixes, score_names):
    """
    Organize flattened diff dictionary into nested dict: {suffix: {score: array}}

    Parameters:
        diffs: dict like {"LogS_diffs_ecdf_f_g": array, ...}
        tag_suffixes: list of suffixes to include (e.g. ['ecdf_f_g', 'oracle_f_g'])
        score_names: list of score types (e.g. ['LogS', 'CS', 'CLS'])

    Returns:
        dict of form {suffix: {score: array}}
    """
    score_dicts = {}
    for suffix in tag_suffixes:
        score_dicts[suffix] = {
            score: diffs[f"{score}_diffs_{suffix}"]
            for score in score_names
        }
    return score_dicts

def make_pair_labels(suffixes):
    """
    Converts suffixes like 'oracle_bb1_localized_f_for_KL_matching' to 'bb1_localized - f_for_KL_matching' labels.

    Parameters:
        suffixes: list of string suffixes (e.g. ['oracle_bb1_localized_f_for_KL_matching'])

    Returns:
        dict mapping each suffix to a pair label (e.g. {'oracle_bb1_localized_f_for_KL_matching': 'bb1_localized - f_for_KL_matching'})
    """
    pair_labels = {}
    for suffix in suffixes:
        parts = suffix.split("_")
        pit = parts[0]
        model_a = parts[1]
        # Join the remaining parts, then split the last underscore group into model_b
        model_and_target = "_".join(parts[2:])
        if "_" in model_and_target:
            split_pos = model_and_target.rfind("_")
            model_b = model_and_target[split_pos+1:]
            model_a = model_and_target[:split_pos]
        else:
            model_b = model_and_target
        pair_labels[suffix] = f"{model_a} - {model_b}"
    return pair_labels

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
    """Compute size-discrepancy curve using per-replication p-values."""
    if alpha_grid is None:
        alpha_grid = np.linspace(0.01, 0.1, 10)

    p_values = np.asarray(p_values)
    rejection_rates = np.array([(p_values < a).mean() for a in alpha_grid])
    discrepancies = rejection_rates - alpha_grid
    return alpha_grid, rejection_rates, discrepancies