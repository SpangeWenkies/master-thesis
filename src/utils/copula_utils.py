"""Utility functions related to copula simulation and transforms."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, t as student_t, multivariate_t
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from collections.abc import Callable


def sample_region_mask(u: np.ndarray, q_threshold: float, df: float | int) -> np.ndarray:
    """Return indicator mask for observations with ``y1 + y2`` below ``q_threshold``.

    Parameters
    ----------
    u : ndarray of shape (n, 2)
        PIT pairs.
    q_threshold : float
        Quantile threshold for the region.
    df : int | float
        Degrees of freedom of the Student-t marginals.
    """
    if u.ndim != 2 or u.shape[1] != 2:
        raise ValueError("u must be of shape (n, 2)")
    if not 0 < q_threshold < 1:
        raise ValueError("q_threshold must be in (0, 1)")
    if df <= 0:
        raise ValueError("df must be positive")

    y1 = student_t.ppf(u[:, 0], df)
    y2 = student_t.ppf(u[:, 1], df)
    q_true = np.quantile(y1 + y2, q_threshold)
    return ((y1 + y2) <= q_true).astype(int)


def grid_region_mask(n: int, u: np.ndarray, q_threshold: float, df: float | int) -> np.ndarray:
    """Return a grid based region mask for copula scores."""
    if n <= 0:
        raise ValueError("n must be positive")
    if u.ndim != 2 or u.shape[1] != 2:
        raise ValueError("u must be of shape (n, 2)")
    if not 0 < q_threshold < 1:
        raise ValueError("q_threshold must be in (0, 1)")
    if df <= 0:
        raise ValueError("df must be positive")

    u_seq = np.linspace(0.005, 0.995, n)
    U1, U2 = np.meshgrid(u_seq, u_seq)
    y1 = student_t.ppf(u[:, 0], df)
    y2 = student_t.ppf(u[:, 1], df)
    q_true = np.quantile(y1 + y2, q_threshold)
    Y1 = student_t.ppf(U1, df)
    Y2 = student_t.ppf(U2, df)
    return ((Y1 + Y2) <= q_true).astype(int)


def sim_sGumbel_PITs(n: int, theta: float) -> np.ndarray:
    """Simulate PITs from a survival Gumbel copula."""
    if not 1 <= theta <= 17:
        raise ValueError(f"Gumbel theta={theta:.4f} must be in [1, 17]")
    numpy2ri.activate()
    ro.r.assign('theta', theta)
    ro.r.assign('n', n)
    ro.r('''
        library(VineCopula)
        C <- BiCop(14, par = theta)
        u <- BiCopSim(n, C)
    ''')
    return np.array(ro.r('u'))


def sGumbel_copula_pdf_from_PITs(u: np.ndarray, theta: float) -> np.ndarray:
    """Evaluate survival Gumbel copula density at ``u``."""
    if not 1 <= theta <= 17:
        raise ValueError(f"Gumbel theta={theta:.4f} must be in [1, 17]")
    numpy2ri.activate()
    ro.globalenv['u_data'] = u
    ro.globalenv['theta'] = theta
    ro.r('''
        library(VineCopula)
        cop_model <- BiCop(family = 14, par = theta)
        cop_pdf <- BiCopPDF(u_data[,1], u_data[,2], cop_model)
    ''')
    return np.array(ro.r('cop_pdf'))


def student_t_copula_pdf_from_PITs(u: np.ndarray, rho: float, df: float | int) -> np.ndarray:
    """Density of a Student-t copula evaluated at ``u``."""
    if not -1 <= rho <= 1:
        raise ValueError("rho must be in [-1, 1]")
    if df <= 0:
        raise ValueError("df must be positive")
    if u.ndim != 2 or u.shape[1] != 2:
        raise ValueError("u must be of shape (n, 2)")
    x = student_t.ppf(u[:, 0], df)
    y = student_t.ppf(u[:, 1], df)
    cov = np.array([[1, rho], [rho, 1]])
    mv_pdf = multivariate_t.pdf(np.stack([x, y], axis=-1), df=df, shape=cov)
    denom = student_t.pdf(x, df) * student_t.pdf(y, df)
    return mv_pdf / denom


def bb1_copula_pdf_from_PITs(u: np.ndarray, theta: float, delta: float) -> np.ndarray:
    """Density of a BB1 copula evaluated at ``u``."""
    if theta <= 0:
        raise ValueError("theta must be positive")
    if delta <= 0:
        raise ValueError("delta must be positive")
    if u.ndim != 2 or u.shape[1] != 2:
        raise ValueError("u must be of shape (n, 2)")
    numpy2ri.activate()
    ro.globalenv['u_data'] = u
    ro.globalenv['theta'] = theta
    ro.globalenv['delta'] = delta
    ro.r('''
        library(VineCopula)
        cop_model <- BiCop(family = 7, par = theta, par2 = delta)
        cop_pdf <- BiCopPDF(u_data[,1], u_data[,2], cop_model)
    ''')
    return np.array(ro.r('cop_pdf'))


def ecdf_transform(Y: np.ndarray) -> np.ndarray:
    """Apply an ECDF transform column-wise."""
    n, d = Y.shape
    U_hat = np.zeros_like(Y)
    for j in range(d):
        ranks = rankdata(Y[:, j], method='ordinal')
        U_hat[:, j] = ranks / (n + 2)
    return U_hat



def compute_ecdf_inverse(sample: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Return an inverse ECDF function for ``sample``."""
    if sample.size == 0:
        raise ValueError("sample must be non-empty")
    return lambda u: np.quantile(sample, u)


def compute_true_t_inverse(df: float | int) -> Callable[[np.ndarray], np.ndarray]:
    """Return the inverse Student-t CDF for the given ``df``."""
    if df <= 0:
        raise ValueError("df must be positive")
    return lambda u: student_t.ppf(u, df)


def simulate_independent_t_copula(n: int, df: float) -> tuple[np.ndarray, np.ndarray]:
    """Simulate PITs from an independent t copula."""
    if n <= 0:
        raise ValueError("n must be positive")
    if df <= 0:
        raise ValueError("df must be positive")
    eps1 = student_t.rvs(df, size=n)
    eps2 = student_t.rvs(df, size=n)
    u1 = student_t.cdf(eps1, df)
    u2 = student_t.cdf(eps2, df)
    return u1, u2


def create_region(
    U1: np.ndarray,
    U2: np.ndarray,
    F1_inv: Callable[[np.ndarray], np.ndarray],
    F2_inv: Callable[[np.ndarray], np.ndarray],
    q_alpha: float,
) -> np.ndarray:
    """Create binary mask ``F1_inv(U1)+F2_inv(U2) <= q_alpha``."""
    if not 0 < q_alpha < 1:
        raise ValueError("q_alpha must be in (0, 1)")
    Y1 = F1_inv(U1)
    Y2 = F2_inv(U2)
    return ((Y1 + Y2) <= q_alpha).astype(int)


def copula_pdf_student_t(U1: np.ndarray, U2: np.ndarray, rho: float, df: float | int) -> np.ndarray:
    """Compute Student-t copula PDF on a grid."""
    if not -1 <= rho <= 1:
        raise ValueError("rho must be in [-1, 1]")
    if df <= 0:
        raise ValueError("df must be positive")
    x = student_t.ppf(U1, df)
    y = student_t.ppf(U2, df)
    cov = np.array([[1, rho], [rho, 1]])
    mv_pdf = multivariate_t.pdf(np.stack([x, y], axis=-1), df=df, shape=cov)
    denom = student_t.pdf(x, df) * student_t.pdf(y, df)
    return mv_pdf / denom


def inverse_ecdf(u: np.ndarray, original_resid: np.ndarray) -> np.ndarray:
    """Map PIT values ``u`` back to residuals using the sample ECDF."""
    if u.ndim != 1:
        raise ValueError("u must be a 1D array")
    if original_resid.size == 0:
        raise ValueError("original_resid must be non-empty")
    sorted_resid = np.sort(original_resid)
    n = len(sorted_resid)
    indices = np.minimum((u * n).astype(int), n - 1)
    return sorted_resid[indices]
