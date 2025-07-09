#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Option C – dynamic ROI **and** per-rep re-tuning of sJoe parameters
(Clayton vs survival-Joe, global LogS, CSL, CLS).
"""

# ─────────────────── Imports ──────────────────────────────────────────
from utils.focused_scores import (
    log_score,
    censored_log_score,
    conditional_log_score,
    student_t_region_mask,
    estimate_fbar,
)
from utils.copula_utils import (
    sGumbel_copula_pdf_from_PITs,
    Clayton_copula_pdf_from_PITs,
    sJoe_copula_pdf_from_PITs,
    sim_sGumbel_PITs,
    sim_Clayton_PITs,
    sim_sJoe_PITs,
)
from utils.optimize_utils import tune_sJoe_params

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as student_t
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing, os

# ─────────────────── 1. Configuration ────────────────────────────────
theta_sGumbel   = 2.0
theta_Clayton   = 3.0
df_tail         = 5
q_threshold     = 0.05
sample_size     = 10_000
n_rep           = 200
n_proc          = min(32, os.cpu_count() or 1)

# constant pdf/sampler for Clayton (needed in every worker)
def pdf_clayton(u):     return Clayton_copula_pdf_from_PITs(u, theta_Clayton)
def sampler_clayton(n): return sim_Clayton_PITs(n, theta_Clayton)
sampler_clayton.pdf = pdf_clayton

# constant pdf for truth
def pdf_sG(u): return sGumbel_copula_pdf_from_PITs(u, theta_sGumbel)

# ─────────────────── 2. Worker: retune + rescore ─────────────────────
def one_rep_retune(rep_idx: int) -> tuple[float, float, float]:
    """
    ➊ simulate truth PITs
    ➋ build ROI from that sample
    ➌ re-tune θ_sJoe*, using the *same* ROI mask
    ➍ recompute F̄_w for all three models
    ➎ return mean (LogS, CSL, CLS) differences
    """
    # ➊ truth sample -----------------------------------------------
    U = sim_sGumbel_PITs(sample_size, theta_sGumbel)

    # ➋ ROI mask for *this* sample ---------------------------------
    y1 = student_t.ppf(U[:, 0], df_tail)
    y2 = student_t.ppf(U[:, 1], df_tail)
    q_true_rep = np.quantile(y1 + y2, q_threshold)
    w_vec = ((y1 + y2) <= q_true_rep).astype(float)

    def w_fn(X):                    # vectorised indicator
        t1 = student_t.ppf(X[:, 0], df_tail)
        t2 = student_t.ppf(X[:, 1], df_tail)
        return ((t1 + t2) <= q_true_rep).astype(float)

    # ➌ re-tune the sJoe parameters on-the-fly ----------------------
    theta_sJoe, theta_sJoe_locd, theta_sJoe_loc = tune_sJoe_params(
        [U], [w_vec], pdf_sG, pdf_clayton, verbose=False
    )

    def pdf_sJoe(u):      return sJoe_copula_pdf_from_PITs(u, theta_sJoe)
    def pdf_sJoe_locd(u): return sJoe_copula_pdf_from_PITs(u, theta_sJoe_locd)
    def pdf_sJoe_loc(u):  return sJoe_copula_pdf_from_PITs(u, theta_sJoe_loc)

    def sampler_sJoe_locd(n):
        return sim_sJoe_PITs(n, theta_sJoe_locd)
    sampler_sJoe_locd.pdf = pdf_sJoe_locd

    def sampler_sJoe_loc(n):
        return sim_sJoe_PITs(n, theta_sJoe_loc)
    sampler_sJoe_loc.pdf = pdf_sJoe_loc

    # ➍ fresh F̄_w with current ROI ---------------------------------
    fbar_C  = estimate_fbar(pdf_clayton,   w_fn, sampler_clayton,   n=60_000)
    fbar_Jd = estimate_fbar(pdf_sJoe_locd, w_fn, sampler_sJoe_locd, n=60_000)
    fbar_Jl = estimate_fbar(pdf_sJoe_loc,  w_fn, sampler_sJoe_loc,  n=60_000)

    # ➎ scores ------------------------------------------------------
    log_C  = np.log(pdf_clayton(U))
    log_J  = np.log(pdf_sJoe(U))
    log_Jd = np.log(pdf_sJoe_locd(U))
    log_Jl = np.log(pdf_sJoe_loc(U))

    mean_log = (log_C - log_J).mean()

    cs_Jd = (w_vec * log_Jd + (1 - w_vec) * np.log(fbar_Jd)).mean()
    cs_C  = (w_vec * log_C  + (1 - w_vec) * np.log(fbar_C )).mean()
    mean_CS = cs_Jd - cs_C

    cls_Jl = (w_vec * (log_Jl - np.log(1 - fbar_Jl))).mean()
    cls_C  = (w_vec * (log_C  - np.log(1 - fbar_C ))).mean()
    mean_CLS = cls_Jl - cls_C

    return mean_log, mean_CS, mean_CLS


# ─────────────────── 3. Main driver (with progress bar) ───────────────
def main() -> None:
    mean_log  = np.empty(n_rep)
    mean_CS   = np.empty(n_rep)
    mean_CLS  = np.empty(n_rep)

    with ProcessPoolExecutor(max_workers=n_proc) as ex:
        futures = [ex.submit(one_rep_retune, i) for i in range(n_rep)]
        for i, fut in enumerate(
            tqdm(as_completed(futures), total=n_rep, ncols=80,
                 desc="Simulations")
        ):
            mean_log[i], mean_CS[i], mean_CLS[i] = fut.result()

    # histograms ----------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].hist(mean_log, bins=40, alpha=0.85)
    ax[0].set_title("Mean LogS diff (Clayton – tuned sJoe)")
    ax[1].hist(mean_CS, bins=40, alpha=0.85, color="tab:orange")
    ax[1].set_title("Mean CSL diff (locd – Clayton)")
    ax[2].hist(mean_CLS, bins=40, alpha=0.85, color="tab:green")
    ax[2].set_title("Mean CLS diff (loc – Clayton)")
    for a in ax: a.set_xlabel("difference")
    ax[0].set_ylabel("frequency")
    plt.tight_layout()
    plt.show()

    print(f"Grand mean LogS diff : {mean_log.mean(): .6f}")
    print(f"Grand mean CSL diff : {mean_CS.mean(): .6f}")
    print(f"Grand mean CLS diff : {mean_CLS.mean(): .6f}")


# ─────────────────── 4. Safe multiprocessing guard ────────────────────
if __name__ == "__main__":
    multiprocessing.freeze_support()   # for Windows/PyInstaller
    main()
