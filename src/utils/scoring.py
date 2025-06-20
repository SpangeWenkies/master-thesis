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
    """Logarithmic score for the survival Gumbel copula."""
    mF = sGumbel_copula_pdf_from_PITs(u, theta)
    mF[mF == 0] = 1e-100
    return float(np.sum(np.log(mF)))


def CS_sGumbel(u: np.ndarray, theta: float, w: np.ndarray) -> float:
    """Censored logarithmic score for the survival Gumbel copula."""
    mF = sGumbel_copula_pdf_from_PITs(u, theta)
    mF[mF == 0] = 1e-100
    log_mF = np.log(mF)
    Fw_bar = np.sum(mF * w) / np.sum(w)
    Fw_bar = max(Fw_bar, 1e-100)
    log_Fw_bar = np.log(Fw_bar)
    return float(np.sum(w * log_mF + (1 - w) * log_Fw_bar))


def CLS_sGumbel(u: np.ndarray, theta: float, w: np.ndarray) -> float:
    """Conditional logarithmic score for the survival Gumbel copula."""
    mF = sGumbel_copula_pdf_from_PITs(u, theta)
    mF[mF == 0] = 1e-100
    F_total = np.sum(mF)
    F_outside = np.sum(mF * (1 - w))
    F_outside = min(F_outside, F_total - 1e-100)
    log_1_minus_Fw = np.log(F_outside / F_total + 1e-100)
    return float(np.sum(w * (np.log(mF) - log_1_minus_Fw)))


def LogS_bb1(u: np.ndarray, theta: float, delta: float) -> float:
    """Logarithmic score for the BB1 copula."""
    mF = bb1_copula_pdf_from_PITs(u, theta, delta)
    mF[mF == 0] = 1e-100
    return float(np.sum(np.log(mF)))


def CS_bb1(u: np.ndarray, theta: float, delta: float, w: np.ndarray) -> float:
    """Censored logarithmic score for the BB1 copula."""
    mF = bb1_copula_pdf_from_PITs(u, theta, delta)
    mF[mF == 0] = 1e-100
    log_mF = np.log(mF)
    Fw_bar = np.sum(mF * w) / np.sum(w)
    Fw_bar = max(Fw_bar, 1e-100)
    log_Fw_bar = np.log(Fw_bar)
    return float(np.sum(w * log_mF + (1 - w) * log_Fw_bar))


def CLS_bb1(u: np.ndarray, theta: float, delta: float, w: np.ndarray) -> float:
    """Conditional logarithmic score for the BB1 copula."""
    mF = bb1_copula_pdf_from_PITs(u, theta, delta)
    mF[mF == 0] = 1e-100
    F_total = np.sum(mF)
    F_outside = np.sum(mF * (1 - w))
    F_outside = min(F_outside, F_total - 1e-100)
    log_1_minus_Fw = np.log(F_outside / F_total + 1e-100)
    return float(np.sum(w * (np.log(mF) - log_1_minus_Fw)))


def LogS_student_t_copula(u: np.ndarray, rho: float, df: float | int) -> float:
    """Logarithmic score for the Student-t copula."""
    mF = student_t_copula_pdf_from_PITs(u, rho, df)
    mF[mF == 0] = 1e-100
    return float(np.sum(np.log(mF)))


def CS_student_t_copula(u: np.ndarray, rho: float, df: float | int, w: np.ndarray) -> float:
    """Censored logarithmic score for the Student-t copula."""
    mF = student_t_copula_pdf_from_PITs(u, rho, df)
    mF[mF == 0] = 1e-100
    log_mF = np.log(mF)
    Fw_bar = np.sum(mF * w) / np.sum(w)
    Fw_bar = max(Fw_bar, 1e-100)
    log_Fw_bar = np.log(Fw_bar)
    return float(np.sum(w * log_mF + (1 - w) * log_Fw_bar))


def CLS_student_t_copula(u: np.ndarray, rho: float, df: float | int, w: np.ndarray) -> float:
    """Conditional logarithmic score for the Student-t copula."""
    mF = student_t_copula_pdf_from_PITs(u, rho, df)
    mF[mF == 0] = 1e-100
    F_total = np.sum(mF)
    F_outside = np.sum(mF * (1 - w))
    F_outside = min(F_outside, F_total - 1e-100)
    log_1_minus_Fw = np.log(F_outside / F_total + 1e-100)
    return float(np.sum(w * (np.log(mF) - log_1_minus_Fw)))


def estimate_kl_divergence_copulas(u_samples: np.ndarray, pdf_p, pdf_q) -> float:
    """Estimate the KL divergence ``D_KL(p||q)`` from samples of ``p``."""
    p_vals = pdf_p(u_samples)
    q_vals = pdf_q(u_samples)
    p_vals[p_vals == 0] = 1e-100
    q_vals[q_vals == 0] = 1e-100
    return float(np.mean(np.log(p_vals) - np.log(q_vals)))


def estimate_localized_kl(u_samples: np.ndarray, pdf_p, pdf_f, region_mask: np.ndarray) -> float:
    """Localized KL divergence in a region defined by ``region_mask``."""
    u_in_A = u_samples[region_mask.astype(bool)]
    p_vals = pdf_p(u_in_A)
    f_vals = pdf_f(u_in_A)
    f_vals[f_vals == 0] = 1e-100
    p_vals[p_vals == 0] = 1e-100
    kl_vals = np.log(p_vals / f_vals)
    return float(np.mean(kl_vals))


def estimate_local_kl(u_samples: np.ndarray, pdf_p, pdf_f, region_mask: np.ndarray) -> float:
    """Local KL divergence using weights given by ``region_mask``."""
    p_vals = pdf_p(u_samples)
    f_vals = pdf_f(u_samples)
    p_vals = np.clip(p_vals, 1e-100, 1e100)
    f_vals = np.clip(f_vals, 1e-100, 1e100)
    w_vals = region_mask.astype(float)
    ratio = np.clip(p_vals / f_vals, 1e-12, 1e12)
    kl_terms = w_vals * p_vals * np.log(ratio)
    return float(np.mean(kl_terms))


def evaluate_mass_in_region(density: np.ndarray, W: np.ndarray) -> float:
    """Fraction of total density mass contained in ``W``."""
    weighted_mass = np.sum(density * W)
    total_mass = np.sum(density)
    return float(weighted_mass / total_mass) if total_mass != 0 else float('nan')


def compute_scores_over_region(density: np.ndarray, W: np.ndarray, eps: float = 1e-12) -> tuple[float, float]:
    """Compute CS and CLS scores over a given region mask."""
    density = density + eps
    log_density = np.log(density)
    CS = float(np.sum(log_density * density * W))
    p_A = np.sum(density * W)
    p_cond = (density * W) / p_A
    CLS = float(np.sum(log_density * p_cond))
    return CS, CLS


def compute_score_differences(pdf_f: np.ndarray, pdf_g: np.ndarray, W_ecdf: np.ndarray, W_oracle: np.ndarray) -> dict:
    """Return score differences between two copulas over two regions."""
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
    """Return rejection decision and p-value for a mean-zero t-test."""
    t_stat, p_val = ttest_1samp(differences, popmean=0)
    return p_val < alpha, p_val


def perform_size_test(differences: np.ndarray, alpha: float = 0.05) -> dict:
    """Perform a size test for equal predictive accuracy."""
    t_stat, p_val = ttest_1samp(differences, popmean=0)
    return {"reject": p_val < alpha, "t_stat": t_stat, "p_value": p_val}


def perform_size_tests(score_dicts: dict, score_names: list[str], pair_to_keys: dict, alpha: float = 0.05) -> dict:
    """Compute size test results for multiple score differences."""
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
