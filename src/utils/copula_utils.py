"""Utility functions related to copula simulation and transforms."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, t as student_t, multivariate_t
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from collections.abc import Callable


def sample_region_mask(u: np.ndarray, q_threshold: float, df: float | int) -> np.ndarray:
    """Return an indicator for PIT pairs falling in a Student-t tail region.

    Parameters
    ----------
    u : ndarray of shape (n, 2)
        PIT pairs ``(u_1, u_2)``.
    q_threshold : float
        Quantile level used to construct the region.
    df : int | float
        Degrees of freedom of the Student-``t`` marginals.

    Returns
    -------
    ndarray of shape (n,)
        Binary mask ``1`` for observations with ``y1 + y2 <= q_true`` where
        ``y1`` and ``y2`` are Student-``t`` quantiles and ``q_true`` is the
        ``q_threshold`` quantile of ``y1 + y2``.
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
    """Return a binary grid for evaluating copula scores on a region.

    Parameters
    ----------
    n : int
        Number of grid points per dimension.
    u : ndarray of shape (m, 2)
        PIT pairs used to determine the quantile region.
    q_threshold : float
        Quantile level defining the region.
    df : int | float
        Degrees of freedom of the Student-``t`` marginals.

    Returns
    -------
    ndarray of shape (n, n)
        Indicator matrix with ones inside the region defined by
        ``F_1^{-1}(U1) + F_2^{-1}(U2) <= q_true``.
    """
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
    """Simulate PIT pairs from a survival Gumbel copula.

    Parameters
    ----------
    n : int
        Number of observations to generate.
    theta : float
        Dependence parameter of the copula.

    Returns
    -------
    ndarray of shape (n, 2)
        Simulated PIT pairs ``(U_1, U_2)``.
    """
    if not 1 <= theta <= 17:
        raise ValueError(f"sGumbel theta={theta:.4f} must be in [1, 17]")
    numpy2ri.activate()
    ro.r.assign('theta', theta)
    ro.r.assign('n', n)
    ro.r('''
        library(VineCopula)
        C <- BiCop(14, par = theta)
        u <- BiCopSim(n, C)
    ''')
    return np.array(ro.r('u'))

def sim_sClayton_PITs(n: int, theta: float) -> np.ndarray:
    """Simulate PIT pairs from a survival Clayton copula.

    Parameters
    ----------
    n : int
        Number of observations to generate.
    theta : float
        Dependence parameter of the copula.

    Returns
    -------
    ndarray of shape (n, 2)
        Simulated PIT pairs ``(U_1, U_2)``.
    """
    if not 0 < theta:
        raise ValueError(f"sClayton theta={theta:.4f} must be > 0")
    numpy2ri.activate()
    ro.r.assign('theta', theta)
    ro.r.assign('n', n)
    ro.r('''
        library(VineCopula)
        C <- BiCop(13, par = theta)
        u <- BiCopSim(n, C)
    ''')
    return np.array(ro.r('u'))

def sim_Clayton_PITs(n: int, theta: float) -> np.ndarray:
    """Simulate PIT pairs from a Clayton copula.

    Parameters
    ----------
    n : int
        Number of observations to generate.
    theta : float
        Dependence parameter of the copula.

    Returns
    -------
    ndarray of shape (n, 2)
        Simulated PIT pairs ``(U_1, U_2)``.
    """
    if not -1 < theta <= 15:
        raise ValueError(f"sClayton theta={theta:.4f} must be in (-1, 15]")
    numpy2ri.activate()
    ro.r.assign('theta', theta)
    ro.r.assign('n', n)
    ro.r('''
        library(VineCopula)
        C <- BiCop(3, par = theta)
        u <- BiCopSim(n, C)
    ''')
    return np.array(ro.r('u'))

def sim_sJoe_PITs(n: int, theta: float) -> np.ndarray:
    """Simulate PIT pairs from a survival Joe copula.

    Parameters
    ----------
    n : int
        Number of observations to generate.
    theta : float
        Dependence parameter of the copula.

    Returns
    -------
    ndarray of shape (n, 2)
        Simulated PIT pairs ``(U_1, U_2)``.
    """
    if not 1 < theta <= 30:
        raise ValueError(f"sClayton theta={theta:.4f} must be in (1, 30]")
    numpy2ri.activate()
    ro.r.assign('theta', theta)
    ro.r.assign('n', n)
    ro.r('''
        library(VineCopula)
        C <- BiCop(16, par = theta)
        u <- BiCopSim(n, C)
    ''')
    return np.array(ro.r('u'))

def sim_student_t_copula_PITs(n: int, rho: float, df: float | int) -> np.ndarray:
    """Simulate PIT pairs from a Student-``t`` copula.

    Parameters
    ----------
    n : int
        Number of observations to simulate.
    rho : float
        Correlation parameter of the copula.
    df : int | float
        Degrees of freedom of the marginals.

    Returns
    -------
    ndarray of shape (n, 2)
        Simulated PIT pairs ``(U_1, U_2)``.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if df <= 0:
        raise ValueError("df must be positive")
    cov = np.array([[1, rho], [rho, 1]])
    eps = multivariate_t.rvs(loc=[0, 0], shape=cov, df=df, size=n)
    return student_t.cdf(eps, df)


def sim_bb1_PITs(n: int, theta: float, delta: float) -> np.ndarray:
    """Simulate PIT pairs from a BB1 copula using R's VineCopula package."""
    if theta <= 0:
        raise ValueError("theta must be positive")
    if delta <= 0:
        raise ValueError("delta must be positive")
    numpy2ri.activate()
    ro.globalenv['theta'] = theta
    ro.globalenv['delta'] = delta
    ro.globalenv['n'] = n
    ro.r('''
        library(VineCopula)
        cop <- BiCop(family = 7, par = theta, par2 = delta)
        u <- BiCopSim(n, cop)
    ''')
    return np.array(ro.r('u'))


def sGumbel_copula_pdf_from_PITs(u: np.ndarray, theta: float) -> np.ndarray:
    """Evaluate the survival Gumbel copula density at PIT pairs.

    Parameters
    ----------
    u : ndarray of shape (n, 2)
        PIT pairs where the density is evaluated.
    theta : float
        Dependence parameter of the copula.

    Returns
    -------
    ndarray of shape (n,)
        Copula density evaluated at each row of ``u``.
    """
    if not 1 <= theta <= 17:
        raise ValueError(f"sGumbel theta={theta:.4f} must be in [1, 17]")
    numpy2ri.activate()
    ro.globalenv['u_data'] = u
    ro.globalenv['theta'] = theta
    ro.r('''
        library(VineCopula)
        cop_model <- BiCop(family = 14, par = theta)
        cop_pdf <- BiCopPDF(u_data[,1], u_data[,2], cop_model)
    ''')
    return np.array(ro.r('cop_pdf'))

def sClayton_copula_pdf_from_PITs(u: np.ndarray, theta: float) -> np.ndarray:
    """Evaluate the survival Clayton copula density at PIT pairs.

    Parameters
    ----------
    u : ndarray of shape (n, 2)
        PIT pairs where the density is evaluated.
    theta : float
        Dependence parameter of the copula.

    Returns
    -------
    ndarray of shape (n,)
        Copula density evaluated at each row of ``u``.
    """
    if not 0 < theta:
        raise ValueError(f"sClayton theta={theta:.4f} must be > 0")
    numpy2ri.activate()
    ro.globalenv['u_data'] = u
    ro.globalenv['theta'] = theta
    ro.r('''
        library(VineCopula)
        cop_model <- BiCop(family = 13, par = theta)
        cop_pdf <- BiCopPDF(u_data[,1], u_data[,2], cop_model)
    ''')
    return np.array(ro.r('cop_pdf'))

def Clayton_copula_pdf_from_PITs(u: np.ndarray, theta: float) -> np.ndarray:
    """Evaluate the Clayton copula density at PIT pairs.

    Parameters
    ----------
    u : ndarray of shape (n, 2)
        PIT pairs where the density is evaluated.
    theta : float
        Dependence parameter of the copula.

    Returns
    -------
    ndarray of shape (n,)
        Copula density evaluated at each row of ``u``.
    """
    if not -1 < theta <= 15:
        raise ValueError(f"Clayton theta={theta:.4f} must be in (-1, 15]")
    numpy2ri.activate()
    ro.globalenv['u_data'] = u
    ro.globalenv['theta'] = theta
    ro.r('''
        library(VineCopula)
        cop_model <- BiCop(family = 3, par = theta)
        cop_pdf <- BiCopPDF(u_data[,1], u_data[,2], cop_model)
    ''')
    return np.array(ro.r('cop_pdf'))

def sJoe_copula_pdf_from_PITs(u: np.ndarray, theta: float) -> np.ndarray:
    """Evaluate the survival Joe copula density at PIT pairs.

    Parameters
    ----------
    u : ndarray of shape (n, 2)
        PIT pairs where the density is evaluated.
    theta : float
        Dependence parameter of the copula.

    Returns
    -------
    ndarray of shape (n,)
        Copula density evaluated at each row of ``u``.
    """
    if not 1 < theta <= 30:
        raise ValueError(f"sJoe theta={theta:.4f} must be in (1, 30]")
    numpy2ri.activate()
    ro.globalenv['u_data'] = u
    ro.globalenv['theta'] = theta
    ro.r('''
        library(VineCopula)
        cop_model <- BiCop(family = 16, par = theta)
        cop_pdf <- BiCopPDF(u_data[,1], u_data[,2], cop_model)
    ''')
    return np.array(ro.r('cop_pdf'))


def student_t_copula_pdf_from_PITs(u: np.ndarray, rho: float, df: float | int) -> np.ndarray:
    """Density of a Student-t copula evaluated at PIT pairs.

    Parameters
    ----------
    u : ndarray of shape (n, 2)
        Locations where the density should be computed.
    rho : float
        Linear correlation parameter of the copula.
    df : int | float
        Degrees of freedom of the Student-``t`` marginals.

    Returns
    -------
    ndarray of shape (n,)
        Copula density evaluated at each row of ``u``.
    """
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
    """Density of a BB1 copula evaluated at PIT pairs.

    Parameters
    ----------
    u : ndarray of shape (n, 2)
        Locations where the density should be computed.
    theta : float
        First BB1 dependence parameter.
    delta : float
        Second BB1 dependence parameter.

    Returns
    -------
    ndarray of shape (n,)
        Copula density evaluated at each row of ``u``.
    """
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
    """Apply the empirical CDF transform to each column of ``Y``.

    Parameters
    ----------
    Y : ndarray of shape (n, d)
        Sample matrix to transform.

    Returns
    -------
    ndarray of shape (n, d)
        Column-wise empirical CDF values ``U_hat``.
    """
    n, d = Y.shape
    U_hat = np.zeros_like(Y)
    for j in range(d):
        ranks = rankdata(Y[:, j], method='ordinal')
        U_hat[:, j] = ranks / (n + 2)
    return U_hat



def compute_ecdf_inverse(sample: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Return a function that maps uniform values to quantiles of ``sample``.

    Parameters
    ----------
    sample : ndarray of shape (n,)
        One-dimensional sample used to construct the empirical quantile function.

    Returns
    -------
    Callable[[ndarray], ndarray]
        Function ``F^{-1}`` such that ``F^{-1}(u)`` has the same shape as ``u``.
    """
    if sample.size == 0:
        raise ValueError("sample must be non-empty")
    return lambda u: np.quantile(sample, u)


def compute_true_t_inverse(df: float | int) -> Callable[[np.ndarray], np.ndarray]:
    """Return the inverse Student-``t`` CDF for a given degrees of freedom.

    Parameters
    ----------
    df : int | float
        Degrees of freedom of the distribution.

    Returns
    -------
    Callable[[ndarray], ndarray]
        Function ``F^{-1}`` that maps uniform samples to Student-``t`` quantiles.
    """
    if df <= 0:
        raise ValueError("df must be positive")
    return lambda u: student_t.ppf(u, df)


def simulate_independent_t_copula(n: int, df: float) -> tuple[np.ndarray, np.ndarray]:
    """Simulate PITs from two independent Student-``t`` marginals.

    Parameters
    ----------
    n : int
        Number of observations to simulate.
    df : float
        Degrees of freedom of the marginal distributions.

    Returns
    -------
    tuple of ndarray
        Pair ``(u1, u2)`` each of shape ``(n,)`` containing the simulated PITs.
    """
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
    """Create binary mask ``F1_inv(U1)+F2_inv(U2) <= q_alpha``.

    Parameters
    ----------
    U1, U2 : ndarray
        Grid values where the mask should be evaluated. Both arrays must have
        the same shape.
    F1_inv, F2_inv : Callable[[ndarray], ndarray]
        Quantile functions applied to ``U1`` and ``U2`` respectively.
    q_alpha : float
        Quantile cutoff used to define the region.

    Returns
    -------
    ndarray
        Binary array with the same shape as ``U1`` and ``U2``.
    """
    if not 0 < q_alpha < 1:
        raise ValueError("q_alpha must be in (0, 1)")
    Y1 = F1_inv(U1)
    Y2 = F2_inv(U2)
    return ((Y1 + Y2) <= q_alpha).astype(int)


def copula_pdf_student_t(U1: np.ndarray, U2: np.ndarray, rho: float, df: float | int) -> np.ndarray:
    """Compute the Student-``t`` copula density on a grid.

    Parameters
    ----------
    U1, U2 : ndarray
        Grid values where the density should be evaluated. Both arrays must have
        the same shape.
    rho : float
        Linear correlation parameter.
    df : float | int
        Degrees of freedom of the Student-``t`` marginals.

    Returns
    -------
    ndarray
        Copula density evaluated at ``(U1, U2)`` with the same shape as ``U1``.
    """
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
    """Map PIT values ``u`` back to residuals using the sample ECDF.

    Parameters
    ----------
    u : ndarray of shape (m,)
        PIT values to transform.
    original_resid : ndarray of shape (n,)
        Residual sample used to build the empirical quantile function.

    Returns
    -------
    ndarray of shape (m,)
        Reconstructed residuals corresponding to ``u``.
    """
    if u.ndim != 1:
        raise ValueError("u must be a 1D array")
    if original_resid.size == 0:
        raise ValueError("original_resid must be non-empty")
    sorted_resid = np.sort(original_resid)
    n = len(sorted_resid)
    indices = np.minimum((u * n).astype(int), n - 1)
    return sorted_resid[indices]

def average_threshold(samples, q_threshold):
    """
    Average the (y1 + y2) q_threshold-quantile across many PIT samples.
    Returns a scalar q_val.
    """
    q_vals = [
        np.quantile(u[:, 0] + u[:, 1], q_threshold)   # tail boundary for this sample
        for u in samples
    ]
    return float(np.mean(q_vals))


def make_fixed_region_mask(reference_u, q_value):
    """
    Build a 0/1 mask from a single PIT sample using the *fixed* boundary q_value.
    """
    y1, y2 = reference_u[:, 0], reference_u[:, 1]
    return ((y1 + y2) <= q_value).astype(int)