from __future__ import annotations

from typing import Callable, Tuple, Any
import numpy as np
import scipy.optimize as opt
from tqdm import tqdm
from .divergences import full_kl, local_kl, localised_kl
from scipy.optimize import minimize
from src.score_sim_config import (
    sJoe_param_bounds,
    kl_match_optim_method
)
import logging
from .copula_utils import (
    Clayton_copula_pdf_from_PITs,
    sJoe_copula_pdf_from_PITs,
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

def tune_sJoe_params(samples_list, masks_list, pdf_sg, pdf_Clayton, verbose=False):
    """KL-match sJoe parameters to the Clayton through their distance to the sGumbel."""

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("rpy2").setLevel(logging.WARNING)

    # ── 1 ▸  Targets   sGumbel ‖ Clayton  ─────────────────────────────
    target_full = np.mean([
        full_kl(u, pdf_sg, pdf_Clayton)
        for u in samples_list
    ])

    target_localised = np.mean([
        localised_kl(u, pdf_sg, pdf_Clayton, m)
        for u, m in zip(samples_list, masks_list)
    ])

    target_local = np.mean([
        local_kl(u, pdf_sg, pdf_Clayton, m)
        for u, m in zip(samples_list, masks_list)
    ])

    # ── 2 ▸  Objectives  sGumbel ‖ sJoe(θ)  ──────────────────────────
    def make_pdf(theta):  # helper
        return lambda u: sJoe_copula_pdf_from_PITs(u, theta)

    def obj_global(theta):
        if theta < 1:
            return np.inf
        pdf = lambda u: sJoe_copula_pdf_from_PITs(u, theta)
        val = np.mean([
            full_kl(u, pdf_sg, pdf)  # reuse same PIT batches
            for u in samples_list
        ])
        return (val - target_full) ** 2

    def obj_localised(theta):
        if theta < 1: return np.inf
        val = np.mean([localised_kl(u, pdf_sg, make_pdf(theta), m)
                       for u, m in zip(samples_list, masks_list)])
        return (val - target_localised) ** 2

    def obj_local(theta):
        if theta < 1: return np.inf
        val = np.mean([local_kl(u, pdf_sg, make_pdf(theta), m)
                       for u, m in zip(samples_list, masks_list)])
        return (val - target_local) ** 2

    # ── 3 ▸  Optimise  ───────────────────────────────────────────────
    res_full = minimize(obj_global, x0=[2.0],
                        bounds=sJoe_param_bounds,
                        method=kl_match_optim_method,
                        tol=1e-20)

    res_localised = minimize(obj_localised, x0=[2.0],
                        bounds=sJoe_param_bounds,
                        method=kl_match_optim_method,
                        tol=1e-20)

    res_local = minimize(obj_local, x0=[2.0],
                        bounds=sJoe_param_bounds,
                        method=kl_match_optim_method,
                        tol=1e-20)

    if verbose:
        pdf_full = lambda u: sJoe_copula_pdf_from_PITs(u, res_full.x[0])
        pdf_localised = lambda u: sJoe_copula_pdf_from_PITs(u, res_localised.x[0])
        pdf_local = lambda u: sJoe_copula_pdf_from_PITs(u, res_local.x[0])
        optim_kl = np.mean([full_kl(u, pdf_sg, pdf_full) for u in samples_list])
        optim_kl_loc = np.mean([localised_kl(u, pdf_sg, pdf_localised, m) for u, m in zip(samples_list, masks_list)])
        optim_kl_local = np.mean([local_kl(u, pdf_sg, pdf_local, m) for u, m in zip(samples_list, masks_list)])
        logger.info(f"Tuned sJoe (full): theta = {res_full.x[0]:.4f}")
        logger.info(f"Target KL(sGumbel||Clayton) full: {target_full:.6f}")
        logger.info(f"Optimized full KL(sGumbel||sJoe): {optim_kl:.6f}")
        logger.info(f"Tuned sJoe (localized): theta = {res_localised.x[0]:.4f}")
        logger.info(f"Target KL(sGumbel||Clayton) localized: {target_localised:.6f}")
        logger.info(f"Optimized localized KL(sGumbel||sJoe): {optim_kl_loc:.6f}")
        logger.info(f"Tuned sJoe (local): theta = {res_local.x[0]:.4f}")
        logger.info(f"Target KL(sGumbel||Clayton) local: {target_local:.6f}")
        logger.info(f"Optimized local KL(sGumbel||sJoe): {optim_kl_local:.6f}")

    return res_full.x, res_localised.x, res_local.x

def tune_sJoe_given_target(samples_list, masks_list, pdf_sg, target_full, target_locd, target_loc, verbose=False):
    """KL-match sJoe parameters to the targets given through sJoe distance to the sGumbel."""

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("rpy2").setLevel(logging.WARNING)

    # ── 2 ▸  Objectives  sGumbel ‖ sJoe(θ)  ──────────────────────────
    def make_pdf(theta):  # helper
        return lambda u: sJoe_copula_pdf_from_PITs(u, theta)

    def obj_global(theta):
        if theta < 1:
            return np.inf
        pdf = lambda u: sJoe_copula_pdf_from_PITs(u, theta)
        val = np.mean([
            full_kl(u, pdf_sg, pdf)  # reuse same PIT batches
            for u in samples_list
        ])
        return (val - target_full) ** 2

    def obj_localised(theta):
        if theta < 1: return np.inf
        val = np.mean([localised_kl(u, pdf_sg, make_pdf(theta), m)
                       for u, m in zip(samples_list, masks_list)])
        return (val - target_locd) ** 2

    def obj_local(theta):
        if theta < 1: return np.inf
        val = np.mean([local_kl(u, pdf_sg, make_pdf(theta), m)
                       for u, m in zip(samples_list, masks_list)])
        return (val - target_loc) ** 2

    # ── 3 ▸  Optimise  ───────────────────────────────────────────────
    res_full = minimize(obj_global, x0=[3.0],
                        bounds=[(1.5, 20)],
                        method=kl_match_optim_method,
                        tol=1e-20)

    res_localised = minimize(obj_localised, x0=[3.0],
                        bounds=[(1.5, 20)],
                        method=kl_match_optim_method,
                        tol=1e-20)

    res_local = minimize(obj_local, x0=[3.0],
                        bounds=[(1.5, 20)],
                        method=kl_match_optim_method,
                        tol=1e-20)


    if verbose:
        pdf_full = lambda u: sJoe_copula_pdf_from_PITs(u, res_full.x[0])
        pdf_localised = lambda u: sJoe_copula_pdf_from_PITs(u, res_localised.x[0])
        pdf_local = lambda u: sJoe_copula_pdf_from_PITs(u, res_local.x[0])
        optim_kl = np.mean([full_kl(u, pdf_sg, pdf_full) for u in samples_list])
        optim_kl_loc = np.mean([localised_kl(u, pdf_sg, pdf_localised, m) for u, m in zip(samples_list, masks_list)])
        optim_kl_local = np.mean([local_kl(u, pdf_sg, pdf_local, m) for u, m in zip(samples_list, masks_list)])
        logger.info(f"Tuned sJoe (full): theta = {res_full.x[0]:.4f}")
        logger.info(f"Target KL(sGumbel||Clayton) full: {target_full:.6f}")
        logger.info(f"Optimized full KL(sGumbel||sJoe): {optim_kl:.6f}")
        logger.info(f"Tuned sJoe (localized): theta = {res_localised.x[0]:.4f}")
        logger.info(f"Target KL(sGumbel||Clayton) localized: {target_locd:.6f}")
        logger.info(f"Optimized localized KL(sGumbel||sJoe): {optim_kl_loc:.6f}")
        logger.info(f"Tuned sJoe (local): theta = {res_local.x[0]:.4f}")
        logger.info(f"Target KL(sGumbel||Clayton) local: {target_loc:.6f}")
        logger.info(f"Optimized local KL(sGumbel||sJoe): {optim_kl_local:.6f}")

    return res_full.x, res_localised.x, res_local.x