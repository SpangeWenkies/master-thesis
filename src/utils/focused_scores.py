# focused_scores.py  ---------------------------------------------------
import numpy as np
from typing import Callable, Literal
from scipy.stats import t as student_t

EPS = 1e-12
# ----------------------------------------------------------------------
def _safe_log(x: np.ndarray | float) -> np.ndarray | float:
    return np.log(np.maximum(x, EPS))
# ----------------------------------------------------------------------
# ------------------------------------------------------------------
#  Log-Score (full, unconditional)
# ------------------------------------------------------------------
def log_score(
    y: np.ndarray,
    pdf_f: Callable[[np.ndarray], np.ndarray],
    *,
    reduce: Literal["none", "mean", "sum"] = "none",
) -> np.ndarray | float:
    """
    Vectorised logarithmic score  S(y) = log f(y).

    Parameters
    ----------
    y        : ndarray, shape (n, d)
               Observations (PIT pairs or any d-variate data).
    pdf_f    : callable
               Vectorised model density, pdf_f(y) → shape-(n,) array.
    reduce   : "none" | "mean" | "sum"
               • "none"  → return the n-vector of log scores
               • "mean"  → return the average log score
               • "sum"   → return the total log likelihood

    Returns
    -------
    ndarray of shape (n,) or float
        Log-scores according to the chosen reduction rule.
    """
    log_vals = np.log(np.maximum(pdf_f(y), EPS))   # safe log

    if reduce == "none":
        return log_vals
    if reduce == "mean":
        return float(np.mean(log_vals))
    if reduce == "sum":
        return float(np.sum(log_vals))

    raise ValueError('reduce must be "none", "mean", or "sum"')
# 1. Student-t tail indicator  ------------------------------------------
def student_t_region_mask(
    u: np.ndarray,
    q_threshold: float,
    df: float | int,
) -> np.ndarray:
    """
    Indicator 1{ y1 + y2 ≤ q_true } where y_i = t_df^{-1}(u_i)
    and q_true is the q_threshold-quantile of y1 + y2 under that df.
    """
    if u.ndim != 2 or u.shape[1] != 2:
        raise ValueError("u must be of shape (n, 2)")
    if not (0. < q_threshold < 1.):
        raise ValueError("q_threshold must be in (0,1)")
    if df <= 0:
        raise ValueError("df must be positive")

    y1 = student_t.ppf(u[:, 0], df)
    y2 = student_t.ppf(u[:, 1], df)
    q_true = np.quantile(y1 + y2, q_threshold)
    return ((y1 + y2) <= q_true).astype(float)   # 0/1 mask
# ----------------------------------------------------------------------
# 2. Helper: turn those two scalars into a callable weight_fn ----------
def make_region_fn(q_threshold: float, df: float | int) -> Callable[[np.ndarray], np.ndarray]:
    """
    Returns  w(u)  = indicator of the Student-t tail region.
    """
    return lambda u: student_t_region_mask(u, q_threshold, df)
# ----------------------------------------------------------------------
# 3. Monte-Carlo estimate of  F̄_w = ∫ (1 − w) f  -----------------------
def estimate_fbar(
    pdf_f: Callable[[np.ndarray], np.ndarray],
    weights,    # type depends on ecdf bool, if true then it is a mask, if false it is a function to create mask from u
    proposal_sampler: Callable[[int], np.ndarray],
    proposal_pdf: Callable[[np.ndarray], np.ndarray] | None = None,
    n: int = 200_000,
    ecdf: bool = False,
) -> float:
    """
    Importance-sampling estimate of F̄_w.
    proposal_sampler(k) → (k,2) draws ~ proposal g.
    If you can sample from f directly, pass that sampler and the weights
    reduce to 1.
    """
    u = proposal_sampler(n)  # (n,2)
    f_u = pdf_f(u)

    if ecdf:
        w_u = weights
    else:
        w_u = weights(u)

    if proposal_pdf is None:  # default: proposal == model  ⇒ weight = 1
        weight = np.ones_like(f_u)
    else:  # importance weight  f / g
        g_u = proposal_pdf(u)
        weight = f_u / np.maximum(g_u, EPS)

    return float(np.sum((1. - w_u) * weight) / np.sum(weight))
# ----------------------------------------------------------------------
# 4. Two Student-t-ROI scoring rules -----------------------------------
def censored_log_score(
    y: np.ndarray,
    pdf_f: Callable[[np.ndarray], np.ndarray],
    *,
    q_threshold: float,
    df: float | int,
    fbar: float | None = None,
    proposal_sampler: Callable[[int], np.ndarray] | None = None,
    n_mc: int = 200_000,
) -> np.ndarray:
    """
    CSL score  S^{flat}_w  with w = Student-t tail indicator.
    """
    w_fn = make_region_fn(q_threshold, df)

    if fbar is None:
        if proposal_sampler is None:
            raise ValueError("Need either fbar or proposal_sampler.")
        fbar = estimate_fbar(pdf_f, w_fn, proposal_sampler, n=n_mc)

    w  = w_fn(y)
    logf = _safe_log(pdf_f(y))
    return w * logf + (1. - w) * _safe_log(fbar)


def conditional_log_score(
    y: np.ndarray,
    pdf_f: Callable[[np.ndarray], np.ndarray],
    *,
    q_threshold: float,
    df: float | int,
    fbar: float | None = None,
    proposal_sampler: Callable[[int], np.ndarray] | None = None,
    n_mc: int = 200_000,
) -> np.ndarray:
    """
    Conditional score  S^{sharp}_w  with w = Student-t tail indicator:
        w(y) * log[ f(y) / (1 - F̄_w) ] .
    """
    w_fn = make_region_fn(q_threshold, df)

    if fbar is None:
        if proposal_sampler is None:
            raise ValueError("Need either fbar or proposal_sampler.")
        fbar = estimate_fbar(pdf_f, w_fn, proposal_sampler, n=n_mc)

    w  = w_fn(y)
    logf = _safe_log(pdf_f(y))
    return w * (logf - _safe_log(1. - fbar))
# ----------------------------------------------------------------------
# 5. Quick demo ---------------------------------------------------------
if __name__ == "__main__":
    from scipy.stats import qmc

    # toy Clayton pdf ---------------------------------------------------
    theta = 2.0

    def clayton_pdf(u):
        u1, u2 = u[:, 0], u[:, 1]
        t = u1 ** (-theta) + u2 ** (-theta) - 1
        return (theta + 1) * (u1 * u2) ** (-(theta + 1)) * t ** (-2.0 - 1.0 / theta)

    # Sobol sampler ≈ Uniform(0,1)^2 -----------------------------------
    sobol = qmc.Sobol(d=2, scramble=True)
    sampler = lambda k: sobol.random(k)

    # fake observations
    obs = sobol.random(10_000)

    # parameters of the ROI
    q_thr, df_t = 0.05, 5

    cs = censored_log_score(
        obs, clayton_pdf, q_threshold=q_thr, df=df_t, proposal_sampler=sampler
    )
    hs = conditional_log_score(
        obs, clayton_pdf, q_threshold=q_thr, df=df_t, proposal_sampler=sampler
    )

    print(f"Mean CSL: {cs.mean():.4f}")
    print(f"Mean CLS: {hs.mean():.4f}")
