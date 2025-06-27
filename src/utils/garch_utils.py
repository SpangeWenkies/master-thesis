"""Utility functions for GARCH simulation and estimation."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
import logging
from .optimize_utils import minimize_with_tqdm

logger = logging.getLogger(__name__)


def WhiteNoiseSim(iT: int, vDistrParams: np.ndarray, sDistrName: str) -> np.ndarray:
    """Generate white noise series.

    Parameters
    ----------
    iT : int
        Length of the series.
    vDistrParams : ndarray of shape (p,)
        Parameters for the chosen distribution. For ``'t'`` this is
        ``[df]``.
    sDistrName : {'normal', 't'}
        Name of the distribution.

    Returns
    -------
    ndarray of shape (iT,)
        Generated white noise.
    """
    if iT <= 0:
        raise ValueError("iT must be positive")
    lDistrName = ['normal', 't']
    if sDistrName not in lDistrName:
        raise ValueError("Distribution not supported.")
    if sDistrName == 'normal':
        return np.random.randn(iT)
    if vDistrParams.size == 0 or vDistrParams[0] <= 0:
        raise ValueError("vDistrParams[0] must be positive for t distribution")
    return np.random.standard_t(vDistrParams[0], size=iT)


def GARCHSim(
    iT: int,
    vGARCHParams: np.ndarray,
    iP: int,
    vDistrParams: np.ndarray,
    sDistrName: str,
) -> np.ndarray:
    """Simulate a univariate GARCH process.

    Parameters
    ----------
    iT : int
        Length of the simulated series.
    vGARCHParams : ndarray of shape (iP + q + 1,)
        Parameters ``[omega, alpha_1, ..., alpha_p, beta_1, ..., beta_q]``.
    iP : int
        Order ``p`` of the ARCH part.
    vDistrParams : ndarray of shape (p,)
        Parameters for the innovation distribution.
    sDistrName : {'normal', 't'}
        Distribution of the innovations.

    Returns
    -------
    ndarray of shape (iT,)
        Simulated GARCH series.
    """
    if iT <= 0:
        raise ValueError("iT must be positive")
    if vGARCHParams.size < iP + 1:
        raise ValueError("vGARCHParams has insufficient length")
    dOmega = vGARCHParams[0]
    vAlpha = vGARCHParams[1:iP + 1]
    vBeta = vGARCHParams[iP + 1:]
    iQ = len(vBeta)
    iR = max(iP, iQ)
    vEps = WhiteNoiseSim(iT + iR, vDistrParams, sDistrName)
    vSig2 = np.zeros(iT + iR)
    dSig20 = dOmega / (1 - np.sum(vAlpha) - np.sum(vBeta))
    vSig2[0:iR] = np.repeat(dSig20, iR)
    vY = np.zeros(iT + iR)
    dY0 = 0
    vY[0:iR] = np.repeat(dY0, iR)
    for t in range(iR, iT):
        vSig2[t] = dOmega + vAlpha @ vY[t - iP:t][::-1] ** 2 + vBeta @ vSig2[t - iQ:t][::-1]
        vY[t] = vEps[t] * np.sqrt(vSig2[t])
    return vY[iR:]


def AvgNLnLGARCH(vGARCHParams: np.ndarray, iP: int, vY: np.ndarray) -> float:
    """Negative average log-likelihood of a GARCH model.

    Parameters
    ----------
    vGARCHParams : ndarray of shape (iP + q + 1,)
        Parameters ``[omega, alpha_1, ..., alpha_p, beta_1, ..., beta_q]``.
    iP : int
        Order ``p`` of the ARCH component.
    vY : ndarray of shape (n,)
        Observed series.

    Returns
    -------
    float
        Negative mean log-likelihood.
    """
    dOmega = vGARCHParams[0]
    vAlpha = vGARCHParams[1:iP + 1]
    vBeta = vGARCHParams[iP + 1:]
    iT = len(vY)
    iQ = len(vBeta)
    iR = max(iP, iQ)
    vSig2 = np.zeros(iT + 1)
    dSig20 = dOmega / (1 - np.sum(vAlpha) - np.sum(vBeta))
    vSig2[0:iR] = np.repeat(dSig20, iR)
    for t in range(iR, iT + 1):
        vSig2[t] = dOmega + vAlpha @ vY[t - iP:t][::-1] ** 2 + vBeta @ vSig2[t - iQ:t][::-1]
    vLL = -0.5 * (np.log(2 * np.pi) + np.log(vSig2[0:-1]) + (vY ** 2) / vSig2[0:-1])
    dALL = np.mean(vLL, axis=0)
    return -dALL


def AvgNLnLGARCHTr(vGARCHParamsTr: np.ndarray, iP: int, vY: np.ndarray) -> float:
    """Wrapper for :func:`AvgNLnLGARCH` using log-transformed parameters."""
    vGARCHParams = np.exp(vGARCHParamsTr)
    return AvgNLnLGARCH(vGARCHParams, iP, vY)


def GARCHEstim(
    vGARCHParams0: np.ndarray,
    iP: int,
    vY: np.ndarray,
    verbose: bool = True,
) -> np.ndarray:
    """Estimate GARCH parameters via maximum likelihood.

    Parameters
    ----------
    vGARCHParams0 : ndarray of shape (iP + q + 1,)
        Initial parameter guess ``[omega, alpha_1, ..., alpha_p, beta_1, ...]``.
    iP : int
        Order ``p`` of the ARCH component.
    vY : ndarray of shape (n,)
        Observed series.
    verbose : bool, default True
        If ``True``, print optimisation information.

    Returns
    -------
    ndarray of shape (iP + q + 1,)
        Estimated parameter vector.
    """
    iT = len(vY)
    vGARCHParams0Tr = np.log(vGARCHParams0)
    res = minimize_with_tqdm(
        AvgNLnLGARCHTr,
        vGARCHParams0Tr,
        args=(iP, vY),
        method="BFGS",
        description="GARCH fit",
    )
    vGARCHParamsStar = np.exp(res.x)
    dLL = -iT * res.fun
    if verbose:
        logger.info(
            "\nBFGS results in %s\nParameter estimates (MLE): %s\nLog-likelihood = %s, f-eval= %s",
            res.message,
            vGARCHParamsStar,
            dLL,
            res.nfev,
        )
    return vGARCHParamsStar


def resid_to_unif_PIT_ECDF(residuals: np.ndarray) -> np.ndarray:
    """Transform residuals to PIT values using the ECDF.

    Parameters
    ----------
    residuals : ndarray of shape (n,)
        Residual series.

    Returns
    -------
    ndarray of shape (n,)
        Probability integral transform of ``residuals``.
    """
    ranks = rankdata(residuals, method="average")
    return ranks / (len(residuals) + 1)


def simulate_GARCH(
    n: int,
    omega: float,
    alpha: float,
    beta: float,
    dist: str,
    df: np.ndarray = np.array([5]),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate two independent GARCH processes.

    Parameters
    ----------
    n : int
        Series length.
    omega : float
        Long run variance level.
    alpha : float
        ARCH coefficient.
    beta : float
        GARCH coefficient.
    dist : {'normal', 't'}
        Distribution of the innovations.
    df : ndarray of shape (1,), default ``np.array([5])``
        Degrees of freedom when ``dist='t'``.

    Returns
    -------
    tuple of ndarray
        ``(resid1, var1, noise1, resid2, var2, noise2)`` each of shape ``(n,)``.
    """
    if dist == "normal":
        white_noise1 = np.random.normal(size=n)
        white_noise2 = np.random.normal(size=n)
    else:
        white_noise1 = np.random.standard_t(df, size=n)
        white_noise2 = np.random.standard_t(df, size=n)
    resid1 = np.zeros_like(white_noise1)
    variance1 = np.zeros_like(white_noise1)
    resid2 = np.zeros_like(white_noise2)
    variance2 = np.zeros_like(white_noise2)
    for t in range(1, n):
        variance1[t] = omega + alpha * resid1[t - 1] ** 2 + beta * variance1[t - 1]
        variance2[t] = omega + alpha * resid2[t - 1] ** 2 + beta * variance2[t - 1]
        resid1[t] = np.sqrt(variance1[t]) * white_noise1[t]
        resid2[t] = np.sqrt(variance2[t]) * white_noise2[t]
    return resid1, variance1, white_noise1, resid2, variance2, white_noise2