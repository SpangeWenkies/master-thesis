from __future__ import annotations

from typing import Callable, Tuple, Any
import numpy as np
import scipy.optimize as opt
from tqdm import tqdm
from .scoring import (
    estimate_kl_divergence_copulasv2,
    estimate_localized_klv2,
    estimate_local_klv2,
)
from scipy.optimize import minimize
from src.score_sim_config import (
    bb1_param_bounds,
    sClayton_param_bounds,
    kl_match_optim_method
)
import logging
from .copula_utils import (
    bb1_copula_pdf_from_PITs,
    sClayton_copula_pdf_from_PITs,
)


def minimize_with_tqdm(
    func: Callable[[np.ndarray], float],
    x0: np.ndarray | list,
    *,
    args: Tuple[Any, ...] = (),
    tol: float = 1e-6,
    description: str = "Optimizing",
    **kwargs: Any,
) -> opt.OptimizeResult:
    """Wrap ``scipy.optimize.minimize`` with a tqdm progress bar.

    Parameters
    ----------
    func : callable
        Objective function with signature ``func(x, *args)``.
    x0 : array_like
        Initial guess for the parameters.
    args : tuple, optional
        Extra positional arguments to ``func``.
    tol : float, default ``1e-6``
        Tolerance passed to ``scipy.optimize.minimize`` and used for the
        progress computation.
    description : str, default ``"Optimizing"``
        Text shown next to the progress bar.
    **kwargs : Any
        Additional keyword arguments forwarded to ``scipy.optimize.minimize``.

    Returns
    -------
    scipy.optimize.OptimizeResult
        Result object from ``scipy.optimize.minimize``.

    Notes
    -----
    The progress bar indicates how close the objective value is to ``tol``. The
    bar completes once the objective value drops below ``tol``.
    """

    pbar = tqdm(total=1.0, desc=description)

    def wrapped_func(x: np.ndarray, *f_args: Any) -> float:
        val = func(x, *f_args)
        progress = min(1.0, tol / (abs(val) + 1e-12))
        pbar.n = progress
        pbar.refresh()
        return val

    result = opt.minimize(wrapped_func, x0, args=args, tol=tol, **kwargs)
    pbar.n = 1.0
    pbar.close()
    return result

def tune_bb1_params(samples_list, masks_list, pdf_sg, pdf_f, verbose=False):
    """KL-match BB1 parameters to the survival Gumbel."""

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("rpy2").setLevel(logging.WARNING)

    target_kl = np.mean([
        estimate_kl_divergence_copulasv2(u, pdf_sg, pdf_f)
        for u in samples_list
    ])
    target_loc = np.mean([
        estimate_localized_klv2(u, pdf_sg, pdf_f, m)
        for u, m in zip(samples_list, masks_list)
    ])
    target_local = np.mean([
        estimate_local_klv2(u, pdf_sg, pdf_f, m)
        for u, m in zip(samples_list, masks_list)
    ])

    def obj(params):
        theta, delta = params
        if theta <= 0 or delta < 1:
            return np.inf
        pdf = lambda u: bb1_copula_pdf_from_PITs(u, theta, delta)
        kl_vals = [estimate_kl_divergence_copulasv2(u, pdf_sg, pdf) for u in samples_list]
        return (np.mean(kl_vals) - target_kl) ** 2

    def obj_loc(params):
        theta, delta = params
        if theta <= 0 or delta < 1:
            return np.inf
        pdf = lambda u: bb1_copula_pdf_from_PITs(u, theta, delta)
        kl_vals = [estimate_localized_klv2(u, pdf_sg, pdf, m) for u, m in zip(samples_list, masks_list)]
        return (np.mean(kl_vals) - target_loc) ** 2

    def obj_local(params):
        theta, delta = params
        if theta <= 0 or delta < 1:
            return np.inf
        pdf = lambda u: bb1_copula_pdf_from_PITs(u, theta, delta)
        kl_vals = [estimate_local_klv2(u, pdf_sg, pdf, m) for u, m in zip(samples_list, masks_list)]
        return (np.mean(kl_vals) - target_local) ** 2

    res_full = minimize(
        obj,
        x0=[2.0, 2.5],
        bounds=bb1_param_bounds,
        method=kl_match_optim_method,
    )
    res_loc = minimize(
        obj_loc,
        x0=[2.0, 2.5],
        bounds=bb1_param_bounds,
        method=kl_match_optim_method,
    )
    res_local = minimize(
        obj_local,
        x0=[2.0, 2.5],
        bounds=bb1_param_bounds,
        method=kl_match_optim_method,
    )

    pdf_full = lambda u: bb1_copula_pdf_from_PITs(u, res_full.x[0], res_full.x[1])
    pdf_loc = lambda u: bb1_copula_pdf_from_PITs(u, res_loc.x[0], res_loc.x[1])
    pdf_local = lambda u: bb1_copula_pdf_from_PITs(u, res_local.x[0], res_local.x[1])

    optim_kl = np.mean([estimate_kl_divergence_copulasv2(u, pdf_sg, pdf_full) for u in samples_list])
    optim_kl_loc = np.mean([estimate_localized_klv2(u, pdf_sg, pdf_loc, m) for u, m in zip(samples_list, masks_list)])
    optim_kl_local = np.mean([estimate_local_klv2(u, pdf_sg, pdf_local, m) for u, m in zip(samples_list, masks_list)])


    if verbose:
        logger.info(f"Tuned BB1 (full): theta = {res_full.x[0]:.4f}, delta = {res_full.x[1]:.4f}")
        logger.info(f"Target KL(sGumbel||f) full: {target_kl:.6f}")
        logger.info(f"Optimized full KL(sGumbel||bb1): {optim_kl:.6f}")
        logger.info(f"Tuned BB1 (localized): theta = {res_loc.x[0]:.4f}, delta = {res_loc.x[1]:.4f}")
        logger.info(f"Target KL(sGumbel||f) localized: {target_loc:.6f}")
        logger.info(f"Optimized localized KL(sGumbel||bb1): {optim_kl_loc:.6f}")
        logger.info(f"Tuned BB1 (local): theta = {res_local.x[0]:.4f}, delta = {res_local.x[1]:.4f}")
        logger.info(f"Target KL(sGumbel||f) local: {target_local:.6f}")
        logger.info(f"Optimized local KL(sGumbel||bb1): {optim_kl_local:.6f}")

    return res_full.x, res_loc.x, res_local.x


def tune_sClayton_params(samples_list, masks_list, pdf_sg, pdf_f, verbose=False):
    """KL-match sClayton parameter to the survival Gumbel."""

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("rpy2").setLevel(logging.WARNING)

    target_kl = np.mean([
        estimate_kl_divergence_copulasv2(u, pdf_sg, pdf_f)
        for u in samples_list
    ])
    target_loc = np.mean([
        estimate_localized_klv2(u, pdf_sg, pdf_f, m)
        for u, m in zip(samples_list, masks_list)
    ])
    target_local = np.mean([
        estimate_local_klv2(u, pdf_sg, pdf_f, m)
        for u, m in zip(samples_list, masks_list)
    ])

    def obj(params):
        theta = params[0]
        if theta <= 0:
            return np.inf
        pdf = lambda u: sClayton_copula_pdf_from_PITs(u, theta)
        kl_vals = [estimate_kl_divergence_copulasv2(u, pdf_sg, pdf) for u in samples_list]
        return (np.mean(kl_vals) - target_kl) ** 2

    def obj_loc(params):
        theta = params[0]
        if theta <= 0:
            return np.inf
        pdf = lambda u: sClayton_copula_pdf_from_PITs(u, theta)
        kl_vals = [estimate_localized_klv2(u, pdf_sg, pdf, m) for u, m in zip(samples_list, masks_list)]
        return (np.mean(kl_vals) - target_loc) ** 2

    def obj_local(params):
        theta = params[0]
        if theta <= 0:
            return np.inf
        pdf = lambda u: sClayton_copula_pdf_from_PITs(u, theta)
        kl_vals = [estimate_local_klv2(u, pdf_sg, pdf, m) for u, m in zip(samples_list, masks_list)]
        return (np.mean(kl_vals) - target_local) ** 2

    res_full = minimize(
        obj,
        x0=[1.0],
        bounds=sClayton_param_bounds,
        method=kl_match_optim_method,
    )
    res_loc = minimize(
        obj_loc,
        x0=[1.0],
        bounds=sClayton_param_bounds,
        method=kl_match_optim_method,
    )
    res_local = minimize(
        obj_local,
        x0=[1.0],
        bounds=sClayton_param_bounds,
        method=kl_match_optim_method,
    )

    pdf_full = lambda u: sClayton_copula_pdf_from_PITs(u, res_full.x[0])
    pdf_loc = lambda u: sClayton_copula_pdf_from_PITs(u, res_loc.x[0])
    pdf_local = lambda u: sClayton_copula_pdf_from_PITs(u, res_local.x[0])

    optim_kl = np.mean([estimate_kl_divergence_copulasv2(u, pdf_sg, pdf_full) for u in samples_list])
    optim_kl_loc = np.mean([estimate_localized_klv2(u, pdf_sg, pdf_loc, m) for u, m in zip(samples_list, masks_list)])
    optim_kl_local = np.mean([estimate_local_klv2(u, pdf_sg, pdf_local, m) for u, m in zip(samples_list, masks_list)])

    if verbose:
        logger.info(f"Tuned sClayton (full): theta = {res_full.x[0]:.4f}")
        logger.info(f"Target KL(sGumbel||f) full: {target_kl:.6f}")
        logger.info(f"Optimized full KL(sGumbel||sClayton): {optim_kl:.6f}")
        logger.info(f"Tuned sClayton (localized): theta = {res_loc.x[0]:.4f}")
        logger.info(f"Target KL(sGumbel||f) localized: {target_loc:.6f}")
        logger.info(f"Optimized localized KL(sGumbel||sClayton): {optim_kl_loc:.6f}")
        logger.info(f"Tuned sClayton (local): theta = {res_local.x[0]:.4f}")
        logger.info(f"Target KL(sGumbel||f) local: {target_local:.6f}")
        logger.info(f"Optimized local KL(sGumbel||sClayton): {optim_kl_local:.6f}")

    return res_full.x[0], res_loc.x[0], res_local.x[0]
