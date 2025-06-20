"""Scoring rules and related helpers for copula evaluation."""

from __future__ import annotations

import numpy as np
from scipy.stats import ttest_1samp

from .copula_utils import (
    sGumbel_copula_pdf_from_PITs,
    bb1_copula_pdf_from_PITs,
    student_t_copula_pdf_from_PITs,
)


def LogS_sGumbel(u: np.ndarray, theta: float) -> float:
    """Return the logarithmic score for a survival Gumbel copula.

    Parameters
    ----------
    u : ndarray of shape (n, 2)
        PIT pairs at which to evaluate the copula density.
    theta : float
        Dependence parameter of the survival Gumbel copula.

    Returns
    -------
    float
        Sum of log densities ``\\sum log f(u_i)``.
    """
    mF = sGumbel_copula_pdf_from_PITs(u, theta)
    mF[mF == 0] = 1e-100
    return float(np.sum(np.log(mF)))


def CS_sGumbel(u: np.ndarray, theta: float, w: np.ndarray) -> float:
    """Return the censored logarithmic score for a survival Gumbel copula.

    Parameters
    ----------
    u : ndarray of shape (n, 2)
        PIT pairs.
    theta : float
        Dependence parameter of the copula.
    w : ndarray of shape (n,)
        Binary mask indicating which observations are in the evaluation region.

    Returns
    -------
    float
        Censored log score ``\\sum w_i log f(u_i) + (1-w_i) log \\bar f_w``.
    """
    mF = sGumbel_copula_pdf_from_PITs(u, theta)
    mF[mF == 0] = 1e-100
    log_mF = np.log(mF)
    Fw_bar = np.sum(mF * w) / np.sum(w)
    Fw_bar = max(Fw_bar, 1e-100)
    log_Fw_bar = np.log(Fw_bar)
    return float(np.sum(w * log_mF + (1 - w) * log_Fw_bar))


def CLS_sGumbel(u: np.ndarray, theta: float, w: np.ndarray) -> float:
    """Return the conditional logarithmic score for a survival Gumbel copula.

    Parameters
    ----------
    u : ndarray of shape (n, 2)
        PIT pairs.
    theta : float
        Dependence parameter.
    w : ndarray of shape (n,)
        Binary mask for the evaluation region.

    Returns
    -------
    float
        Conditional log score ``\\sum w_i (log f(u_i) - log(1-F_w))``.
    """
    mF = sGumbel_copula_pdf_from_PITs(u, theta)
    mF[mF == 0] = 1e-100
    F_total = np.sum(mF)
    F_outside = np.sum(mF * (1 - w))
    F_outside = min(F_outside, F_total - 1e-100)
    log_1_minus_Fw = np.log(F_outside / F_total + 1e-100)
    return float(np.sum(w * (np.log(mF) - log_1_minus_Fw)))


def LogS_bb1(u: np.ndarray, theta: float, delta: float) -> float:
    """Return the logarithmic score for a BB1 copula.

    Parameters
    ----------
    u : ndarray of shape (n, 2)
        PIT pairs.
    theta : float
        First BB1 parameter.
    delta : float
        Second BB1 parameter.

    Returns
    -------
    float
        Sum of log densities.
    """
    mF = bb1_copula_pdf_from_PITs(u, theta, delta)
    mF[mF == 0] = 1e-100
    return float(np.sum(np.log(mF)))


def CS_bb1(u: np.ndarray, theta: float, delta: float, w: np.ndarray) -> float:
    """Return the censored logarithmic score for a BB1 copula.

    Parameters
    ----------
    u : ndarray of shape (n, 2)
        PIT pairs.
    theta : float
        First BB1 parameter.
    delta : float
        Second BB1 parameter.
    w : ndarray of shape (n,)
        Binary mask selecting the evaluation region.

    Returns
    -------
    float
        Censored log score as for :func:`CS_sGumbel`.
    """
    mF = bb1_copula_pdf_from_PITs(u, theta, delta)
    mF[mF == 0] = 1e-100
    log_mF = np.log(mF)
    Fw_bar = np.sum(mF * w) / np.sum(w)
    Fw_bar = max(Fw_bar, 1e-100)
    log_Fw_bar = np.log(Fw_bar)
    return float(np.sum(w * log_mF + (1 - w) * log_Fw_bar))


def CLS_bb1(u: np.ndarray, theta: float, delta: float, w: np.ndarray) -> float:
    """Return the conditional logarithmic score for a BB1 copula.

    Parameters
    ----------
    u : ndarray of shape (n, 2)
        PIT pairs.
    theta : float
        First BB1 parameter.
    delta : float
        Second BB1 parameter.
    w : ndarray of shape (n,)
        Binary mask for the evaluation region.

    Returns
    -------
    float
        Conditional log score, analogous to :func:`CLS_sGumbel`.
    """
    mF = bb1_copula_pdf_from_PITs(u, theta, delta)
    mF[mF == 0] = 1e-100
    F_total = np.sum(mF)
    F_outside = np.sum(mF * (1 - w))
    F_outside = min(F_outside, F_total - 1e-100)
    log_1_minus_Fw = np.log(F_outside / F_total + 1e-100)
    return float(np.sum(w * (np.log(mF) - log_1_minus_Fw)))


def LogS_student_t_copula(u: np.ndarray, rho: float, df: float | int) -> float:
    """Return the logarithmic score for a Student-t copula.

    Parameters
    ----------
    u : ndarray of shape (n, 2)
        PIT pairs.
    rho : float
        Correlation parameter ``\rho``.
    df : int or float
        Degrees of freedom.

    Returns
    -------
    float
        Sum of log densities at ``u``.
    """
    mF = student_t_copula_pdf_from_PITs(u, rho, df)
    mF[mF == 0] = 1e-100
    return float(np.sum(np.log(mF)))


def CS_student_t_copula(u: np.ndarray, rho: float, df: float | int, w: np.ndarray) -> float:
    """Return the censored logarithmic score for a Student-t copula.

    Parameters
    ----------
    u : ndarray of shape (n, 2)
        PIT pairs.
    rho : float
        Correlation parameter ``\rho``.
    df : int or float
        Degrees of freedom.
    w : ndarray of shape (n,)
        Binary mask for region weights.

    Returns
    -------
    float
        Censored log score.
    """
    mF = student_t_copula_pdf_from_PITs(u, rho, df)
    mF[mF == 0] = 1e-100
    log_mF = np.log(mF)
    Fw_bar = np.sum(mF * w) / np.sum(w)
    Fw_bar = max(Fw_bar, 1e-100)
    log_Fw_bar = np.log(Fw_bar)
    return float(np.sum(w * log_mF + (1 - w) * log_Fw_bar))


def CLS_student_t_copula(u: np.ndarray, rho: float, df: float | int, w: np.ndarray) -> float:
    """Return the conditional logarithmic score for a Student-t copula.

    Parameters
    ----------
    u : ndarray of shape (n, 2)
        PIT pairs.
    rho : float
        Correlation parameter ``\rho``.
    df : int or float
        Degrees of freedom.
    w : ndarray of shape (n,)
        Binary mask for region weights.

    Returns
    -------
    float
        Conditional log score.
    """
    mF = student_t_copula_pdf_from_PITs(u, rho, df)
    mF[mF == 0] = 1e-100
    F_total = np.sum(mF)
    F_outside = np.sum(mF * (1 - w))
    F_outside = min(F_outside, F_total - 1e-100)
    log_1_minus_Fw = np.log(F_outside / F_total + 1e-100)
    return float(np.sum(w * (np.log(mF) - log_1_minus_Fw)))


def estimate_kl_divergence_copulas(u_samples: np.ndarray, pdf_p, pdf_q) -> float:
    """Estimate the KL divergence ``D_KL(p||q)`` from samples.

    Parameters
    ----------
    u_samples : ndarray of shape (n, 2)
        Samples from distribution ``p``.
    pdf_p : Callable[[np.ndarray], np.ndarray]
        Density function of ``p`` evaluated on pairs.
    pdf_q : Callable[[np.ndarray], np.ndarray]
        Density function of ``q`` evaluated on pairs.

    Returns
    -------
    float
        Approximation of ``E_p[log p/q]``.
    """
    p_vals = pdf_p(u_samples)
    q_vals = pdf_q(u_samples)
    p_vals[p_vals == 0] = 1e-100
    q_vals[q_vals == 0] = 1e-100
    return float(np.mean(np.log(p_vals) - np.log(q_vals)))


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
    u_in_A = u_samples[region_mask.astype(bool)]
    p_vals = pdf_p(u_in_A)
    f_vals = pdf_f(u_in_A)
    f_vals[f_vals == 0] = 1e-100
    p_vals[p_vals == 0] = 1e-100
    kl_vals = np.log(p_vals / f_vals)
    return float(np.mean(kl_vals))


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
    p_vals = np.clip(p_vals, 1e-100, 1e100)
    f_vals = np.clip(f_vals, 1e-100, 1e100)
    w_vals = region_mask.astype(float)
    ratio = np.clip(p_vals / f_vals, 1e-12, 1e12)
    kl_terms = w_vals * p_vals * np.log(ratio)
    return float(np.mean(kl_terms))


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
