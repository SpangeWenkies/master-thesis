from __future__ import annotations

from typing import Callable, Tuple, Any
import numpy as np
import scipy.optimize as opt
from tqdm import tqdm


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
