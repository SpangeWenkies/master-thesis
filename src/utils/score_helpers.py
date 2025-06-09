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
    Converts suffixes like 'oracle_f_g' to 'f - g' labels.

    Parameters:
        suffixes: list of string suffixes (e.g. ['oracle_f_g'])

    Returns:
        dict mapping each suffix to a pair label (e.g. {'oracle_f_g': 'f - g'})
    """
    pair_labels = {}
    for suffix in suffixes:
        parts = suffix.split("_")  # e.g., ["oracle", "bb1", "g_for_KL_matching"]
        pit = parts[0]
        model_a = parts[1]
        model_b = "_".join(parts[2:])  # allow underscores in model_b
        pair_labels[suffix] = f"{model_a} - {model_b}"
    return pair_labels