"""Scoring rules and related helpers for copula evaluation."""

from __future__ import annotations

import numpy as np
from scipy.stats import ttest_1samp

from .copula_utils import (
    sGumbel_copula_pdf_from_PITs,
    bb1_copula_pdf_from_PITs,
    student_t_copula_pdf_from_PITs,
)

def _fw_bar(mF: np.ndarray, w: np.ndarray) -> float:
    """Return P(Y not in R) given densities mF and 0/1 mask w."""
    F_total = max(np.sum(mF), 1e-100)      # ≈ ∫ f(y) dy  over sample points
    F_outside = np.sum(mF * (1 - w))       # mass outside region
    F_outside = max(F_outside, 1e-100)
    F_outside = min(F_outside, F_total)    # numerical guard
    return F_outside / F_total            # right-tail probability

def outside_prob_from_sample(sample: np.ndarray, q_val: float) -> float:
    """Return ``P(U1 + U2 > q_val)`` estimated from PIT samples."""
    if sample.ndim != 2 or sample.shape[1] != 2:
        raise ValueError("sample must be of shape (n, 2)")
    return float(np.mean(sample[:, 0] + sample[:, 1] > q_val))


def LogS(mF: np.ndarray) -> np.ndarray:
    """Return elementwise logarithmic score ``log f(y)``.

    Parameters
    ----------
    mF : ndarray
        Density evaluations ``f(y)``.

    Returns
    -------
    ndarray
        ``log f(y)`` with zeros avoided by clipping ``mF``.
    """
    mF = np.asarray(mF).copy()
    mF[mF == 0] = 1e-100  # avoid numerical zeros
    return np.log(mF)


def CS(
    mF: np.ndarray,
    u: np.ndarray,
    q_val: float,
    Fw_bar: float | None = None,
) -> np.ndarray:
    """Return censored logarithmic score contributions.

    Parameters
    ----------
    mF : ndarray
        Density evaluations ``f(y)``.
    u : ndarray
        PIT pairs associated with the densities.
    q_val : float
        Threshold used to construct the binary weight ``w`` via
        ``u[:,0] + u[:,1] <= q_val``.
    Fw_bar : float, optional
        Right-tail probability ``\bar F_w``. If ``None`` it is estimated from
        ``mF`` and the internally computed ``w``.

    Returns
    -------
    ndarray
        Censored log score ``w * log f(y) + (1-w) * log \bar F_w``.
    """
    w = (u[:, 0] + u[:, 1] <= q_val).astype(float)
    if Fw_bar is None:
        Fw_bar = _fw_bar(mF, w)
    mF = np.asarray(mF).copy()
    mF[mF == 0] = 1e-100  # avoid numerical zeros
    return w * np.log(mF) + (1 - w) * np.log(Fw_bar)


def CLS(
    mF: np.ndarray,
    u: np.ndarray,
    q_val: float,
    Fw_bar: float | None = None,
) -> np.ndarray:
    """Return conditional logarithmic score contributions.

    Parameters
    ----------
    mF : ndarray
        Density evaluations ``f(y)``.
    u : ndarray
        PIT pairs associated with the densities.
    q_val : float
        Threshold used to construct the binary weight ``w`` via
        ``u[:,0] + u[:,1] <= q_val``.
    Fw_bar : float, optional
        Right-tail probability ``\bar F_w``. If ``None`` it is estimated from
        ``mF`` and the internally computed ``w``.

    Returns
    -------
    ndarray
        Conditional log score ``w * (log f(y) - log(1-\bar F_w))``.
    """
    w = (u[:, 0] + u[:, 1] <= q_val).astype(float)
    if Fw_bar is None:
        Fw_bar = _fw_bar(mF, w)
    # ensure strictly positive density values to avoid log(0)
    mF = np.asarray(mF).copy()
    mF[mF == 0] = 1e-100
    return w * (np.log(mF) - np.log(1.0 - Fw_bar))


def estimate_localized_kl(u_samples: np.ndarray, pdf_p, pdf_f, region_mask: np.ndarray) -> float:
    """Estimate KL divergence restricted to ``region_mask``.

    Parameters
    ----------
    u_samples : ndarray of shape (n, 2)
        Sample pairs from ``p``.
    pdf_p : Callable[[np.ndarray], np.ndarray]
        Density of ``p``.
    pdf_f : Callable[[np.ndarray], np.ndarray]
        Density of reference distribution ``f``.
    region_mask : ndarray of shape (n,)
        Boolean mask selecting samples in the region.

    Returns
    -------
    float
        Localized KL divergence.
    """
    p = pdf_p(u_samples)
    f = pdf_f(u_samples)
    w = region_mask.astype(float)

    log_ratio = np.log(p) - np.log(f)
    return float((w * p * log_ratio).mean())


def estimate_local_kl(u_samples: np.ndarray, pdf_p, pdf_f, region_mask: np.ndarray) -> float:
    """Estimate a weighted KL divergence using ``region_mask`` as weights.

    Parameters
    ----------
    u_samples : ndarray of shape (n, 2)
        Sample pairs from ``p``.
    pdf_p : Callable[[np.ndarray], np.ndarray]
        Density of ``p``.
    pdf_f : Callable[[np.ndarray], np.ndarray]
        Density of reference ``f``.
    region_mask : ndarray of shape (n,)
        Non-negative weights for each sample.

    Returns
    -------
    float
        Weighted KL divergence.
    """
    p_vals = pdf_p(u_samples)
    f_vals = pdf_f(u_samples)
    log_ratio = np.log(p_vals) - np.log(f_vals)

    w = region_mask.astype(float)  # 0/1 indicator
    mass_in_R = w.mean()  # ≈ P(R) because E_P[w] = P(R)

    if mass_in_R == 0.0:
        raise ValueError("Region mask selects no samples; cannot estimate KL_R.")

    # conditional expectation
    return float((w * log_ratio).mean() / mass_in_R)


def evaluate_mass_in_region(density: np.ndarray, W: np.ndarray) -> float:
    """Return fraction of total probability mass contained in a region.

    Parameters
    ----------
    density : ndarray of shape (m, n)
        Density evaluated on a grid.
    W : ndarray of shape (m, n)
        Binary mask selecting the region.

    Returns
    -------
    float
        Fraction ``sum(density * W) / sum(density)``.
    """
    weighted_mass = np.sum(density * W)
    total_mass = np.sum(density)
    return float(weighted_mass / total_mass) if total_mass != 0 else float('nan')


def compute_scores_over_region(density: np.ndarray, W: np.ndarray, eps: float = 1e-12) -> tuple[float, float]:
    """Compute CS and CLS scores over a given region mask.

    Parameters
    ----------
    density : ndarray of shape (m, n)
        Density evaluated on a grid.
    W : ndarray of shape (m, n)
        Binary region mask.
    eps : float, default 1e-12
        Small constant to avoid log of zero.

    Returns
    -------
    tuple of float of shape (2,)
        ``(CS, CLS)`` scores.
    """
    density = density + eps
    log_density = np.log(density)
    CS = float(np.sum(log_density * density * W))
    p_A = np.sum(density * W)
    p_cond = (density * W) / p_A
    CLS = float(np.sum(log_density * p_cond))
    return CS, CLS


def compute_score_differences(pdf_f: np.ndarray, pdf_g: np.ndarray, W_ecdf: np.ndarray, W_oracle: np.ndarray) -> dict:
    """Return score differences between two copulas over two regions.

    Parameters
    ----------
    pdf_f : ndarray of shape (m, n)
        Density values of model ``f``.
    pdf_g : ndarray of shape (m, n)
        Density values of model ``g``.
    W_ecdf : ndarray of shape (m, n)
        Region mask from ECDF PITs.
    W_oracle : ndarray of shape (m, n)
        Region mask from oracle PITs.

    Returns
    -------
    dict[str, float]
        Mapping ``{'CLS_diff_ecdf', 'CLS_diff_oracle', 'CS_diff_ecdf', 'CS_diff_oracle'}`` to scalar differences.
    """
    CS_f_oracle, CLS_f_oracle = compute_scores_over_region(pdf_f, W_oracle)
    CS_g_oracle, CLS_g_oracle = compute_scores_over_region(pdf_g, W_oracle)
    CS_f_ecdf, CLS_f_ecdf = compute_scores_over_region(pdf_f, W_ecdf)
    CS_g_ecdf, CLS_g_ecdf = compute_scores_over_region(pdf_g, W_ecdf)
    return {
        "CLS_diff_ecdf": CLS_f_ecdf - CLS_g_ecdf,
        "CLS_diff_oracle": CLS_f_oracle - CLS_g_oracle,
        "CS_diff_ecdf": CS_f_ecdf - CS_g_ecdf,
        "CS_diff_oracle": CS_f_oracle - CS_g_oracle,
    }


def rejection_rate(differences: np.ndarray, alpha: float = 0.05) -> tuple[bool, float]:
    """Return rejection decision and p-value for a mean-zero t-test.

    Parameters
    ----------
    differences : ndarray of shape (n,)
        Score differences.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    tuple[bool, float]
        ``(reject, p_value)`` decision and corresponding p-value.
    """
    t_stat, p_val = ttest_1samp(differences, popmean=0)
    return p_val < alpha, p_val


def perform_size_test(differences: np.ndarray, alpha: float = 0.05) -> dict:
    """Perform a size test for equal predictive accuracy.

    Parameters
    ----------
    differences : ndarray of shape (n,)
        Score differences between models.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    dict[str, float | bool]
        ``{"reject": bool, "t_stat": float, "p_value": float}`` summary statistics.
    """
    t_stat, p_val = ttest_1samp(differences, popmean=0)
    return {"reject": p_val < alpha, "t_stat": t_stat, "p_value": p_val}


def perform_size_tests(score_dicts: dict, score_names: list[str], pair_to_keys: dict, alpha: float = 0.05) -> dict:
    """Compute size test results for multiple score differences.

    Parameters
    ----------
    score_dicts : dict
        Nested mapping ``DiffKey -> {score_name: array}``.
    score_names : list of str
        Score names to test.
    pair_to_keys : dict
        Mapping label -> (oracle_key, ecdf_key).
    alpha : float, default 0.05
        Significance level used for each test.

    Returns
    -------
    dict[str, dict[str, dict[str, float | bool]]]
        ``{label: {score_name: {...}}}`` results per pair label.
    """
    results: dict = {}
    for label, (oracle_key, ecdf_key) in pair_to_keys.items():
        if oracle_key not in score_dicts or ecdf_key not in score_dicts:
            continue
        res_pair = {}
        for score in score_names:
            diff_oracle = score_dicts[oracle_key][score]
            diff_ecdf = score_dicts[ecdf_key][score]
            res_pair[score] = {
                "oracle": perform_size_test(diff_oracle, alpha),
                "ecdf": perform_size_test(diff_ecdf, alpha),
            }
        results[label] = res_pair
    return results

def estimate_kl_divergence_copulasv2(
    u_samples: np.ndarray,
    pdf_p,
    pdf_q,
    eps: float = 1e-100,
) -> float:
    """
    Monte-Carlo estimate of  D_KL(P || Q) = E_P[log p/q].

    Parameters
    ----------
    u_samples : (n, d) ndarray
        IID draws from the *first* copula P.
    pdf_p, pdf_q : Callable[[ndarray], ndarray]
        Density functions for P and Q, evaluated at a batch of points.
    eps : float
        Lower bound used to clip the densities to avoid log(0).

    Returns
    -------
    float
        Non-negative KL divergence (up to MC noise).
    """
    # --- evaluate densities --------------------------------
    p_vals = pdf_p(u_samples)
    q_vals = pdf_q(u_samples)

    # --- MC estimate of E_P[log p/q] -----------------------------------------
    log_ratio = np.log(p_vals) - np.log(q_vals)
    return float(log_ratio.mean())

def estimate_localized_klv2(
    u_samples: np.ndarray,
    pdf_p,
    pdf_f,
    region_mask: np.ndarray,
    eps: float = 1e-100,
) -> float:
    """
    Localised KL  D_w(P||F)  for  w(x) = 1_R(x).

    The result depends *only* on the probability masses that P and F assign
    to the region R – exactly what Definition 3 requires.

    Parameters
    ----------
    u_samples : (n, d) ndarray
        Draws from the candidate / model copula P.
    pdf_p, pdf_f : Callable[[ndarray], ndarray]
        Copula densities of P and reference F.
    region_mask : (n,) boolean ndarray
        region_mask[i] is True  ⇔  u_samples[i] ∈ R.
    eps : float
        Numerical guard to keep log-arguments in (0,1).

    Returns
    -------
    float
        D_w(P‖F) ≥ 0.
    """
    # ---- probability of R under P ---------------------------------------
    p_R = float(region_mask.mean())

    # ---- probability of R under F (importance sampling) -----------------
    p_vals = pdf_p(u_samples)
    f_vals = pdf_f(u_samples)
    weights = f_vals / p_vals                                #  dF/dP
    f_R = float((weights * region_mask).mean())

    # ---- numerical guards ------------------------------------------------

    # ---- binary-KL formula ----------------------------------------------
    return (
        p_R * np.log(p_R / f_R)
        + (1.0 - p_R) * np.log((1.0 - p_R) / (1.0 - f_R))
    )


def estimate_local_klv2(
    u_samples: np.ndarray,
    pdf_p,
    pdf_f,
    region_mask: np.ndarray,
    eps: float = 1e-100,
) -> float:
    """
    KL *inside* the region – i.e.  KL( P(·|R)  ||  F(·|R) ).

    Unlike `estimate_localized_kl`, this keeps the full conditional densities
    on R.  It is useful when you want to match behaviour *within* the region
    after having matched the masses with `estimate_localized_kl`.

    Returns
    -------
    float
        KL(P(·|R)‖F(·|R)) ≥ 0, or raises ValueError if no sample falls in R.
    """
    # densities at the P-samples
    p_vals = pdf_p(u_samples)
    f_vals = pdf_f(u_samples)
    log_ratio = np.log(p_vals) - np.log(f_vals)

    w = region_mask.astype(float)
    p_R = w.mean()                                           # P(R)

    if p_R < eps:
        raise ValueError("Region mask selects no samples; cannot estimate KL on R.")

    # ---- 1.  E_P[ log p/f | R ] -----------------------------------------
    cond_term = (w * log_ratio).mean() / p_R

    # ---- 2.  log F(R) / P(R)  -------------------------------------------
    weights = f_vals / p_vals                                #  dF/dP
    f_R = (w * weights).mean()

    return cond_term + np.log(f_R / p_R)