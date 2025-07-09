#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rolling-window study (manual score formulas):
    • window length  R_window
    • P_steps one-step-ahead forecasts per replication
    • B_rep Monte-Carlo replications, run in parallel

Score differences returned:
    ΔLogS      = log f_Clayton – log f_sJoe
    ΔCSL, ΔCLS with ROI = Student-t tail rectangle (q_threshold, df_tail)
"""

# ───────────────── Imports ────────────────────────────────────────────
from utils.copula_utils import (
    sGumbel_copula_pdf_from_PITs,
    Clayton_copula_pdf_from_PITs,
    sJoe_copula_pdf_from_PITs,
    sim_sGumbel_PITs,
    sim_Clayton_PITs,
    sim_sJoe_PITs,
)
from utils.optimize_utils import tune_sJoe_params
from utils.focused_scores import estimate_fbar        # only for f̄_w

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as student_t, gaussian_kde, norm
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing, os

# ───────────────── 1. Configuration ───────────────────────────────────
theta_sGumbel = 2.0
theta_Clayton = 3.0
df_tail       = 5
q_threshold   = 0.05

R_window      = 500        # estimation window length
P_steps       = 100        # one-step forecasts per replication
B_rep         = 100        # Monte-Carlo replications

n_proc        = min(32, os.cpu_count() or 1)
n_mc_fbar     = 40_000       # MC points for f̄_w(f) each window
EPS           = 1e-12        # safe-log constant

alpha_grid    = np.linspace(0.01, 0.2, 20)

# ───────────────── 2. Constant model pieces ───────────────────────────
def pdf_clayton(u):     return Clayton_copula_pdf_from_PITs(u, theta_Clayton)
def sampler_clayton(n): return sim_Clayton_PITs(n, theta_Clayton)
sampler_clayton.pdf = pdf_clayton

def pdf_truth(u): return sGumbel_copula_pdf_from_PITs(u, theta_sGumbel)

# ───────────────── 3. Worker: one replication ─────────────────────────
def run_one_rep(rep_idx: int) -> tuple[float, float, float]:
    # 3.1 simulate R+P PITs
    U_all = sim_sGumbel_PITs(R_window + P_steps, theta_sGumbel)

    # accumulators for P differences
    sum_log, sum_cs, sum_cls = 0.0, 0.0, 0.0

    MC_C = sampler_clayton(n_mc_fbar)   # reusable MC grid for f̄_clayton

    # 2. containers to store the *P* one-step diffs
    log_diffs = np.empty(P_steps)
    cs_diffs = np.empty(P_steps)
    cls_diffs = np.empty(P_steps)

    for t in range(P_steps):            # rolling windows
        U_win  = U_all[t : t+R_window]             # window (R×2)
        U_next = U_all[t+R_window : t+R_window+1]  # next PIT (1×2)

        # ---------- ROI for this window --------------------------------
        y1 = student_t.ppf(U_win[:, 0], df_tail)
        y2 = student_t.ppf(U_win[:, 1], df_tail)
        q_star = np.quantile(y1 + y2, q_threshold)

        mask_win = ((y1 + y2) <= q_star).astype(float)

        def w_fn(x):
            t1 = student_t.ppf(x[:, 0], df_tail)
            t2 = student_t.ppf(x[:, 1], df_tail)
            return ((t1 + t2) <= q_star).astype(float)

        # ---------- re-tune θ_sJoe* on this window ----------------------
        theta_J, theta_J_localized, theta_J_local = tune_sJoe_params(
            [U_win], [mask_win], pdf_truth, pdf_clayton, verbose=False
        )

        pdf_J  = lambda u: sJoe_copula_pdf_from_PITs(u, theta_J)
        pdf_J_localized = lambda u: sJoe_copula_pdf_from_PITs(u, theta_J_localized)
        pdf_J_local = lambda u: sJoe_copula_pdf_from_PITs(u, theta_J_local)

        def sampler_J_localized(n): return sim_sJoe_PITs(n, theta_J_localized)
        sampler_J_localized.pdf = pdf_J_localized
        def sampler_J_local(n): return sim_sJoe_PITs(n, theta_J_local)
        sampler_J_local.pdf = pdf_J_local

        # ---------- fresh f̄_w(f) for the same ROI ----------------------
        fbar_C  = estimate_fbar(pdf_clayton, w_fn, sampler_clayton, n=n_mc_fbar)
        fbar_J_localized = estimate_fbar(pdf_J_localized,      w_fn, sampler_J_localized,      n=n_mc_fbar)
        fbar_J_local = estimate_fbar(pdf_J_local,      w_fn, sampler_J_local,      n=n_mc_fbar)

        # ---------- manual scores for the one-step PIT ------------------
        w_next = w_fn(U_next)[0]                      # 0 or 1

        log_C = np.log(np.maximum(pdf_clayton(U_next)[0], EPS))
        log_J = np.log(np.maximum(pdf_J(U_next)[0],       EPS))
        log_J_localized= np.log(np.maximum(pdf_J_localized(U_next)[0],      EPS))
        log_J_local= np.log(np.maximum(pdf_J_local(U_next)[0],      EPS))

        # Log-score difference
        sum_log += log_C - log_J

        # CS difference
        cs_J_localized = w_next * log_J_localized + (1 - w_next) * np.log(np.maximum(fbar_J_localized, EPS))
        cs_C  = w_next * log_C  + (1 - w_next) * np.log(np.maximum(fbar_C,  EPS))
        sum_cs +=  cs_C - cs_J_localized

        # CLS difference
        cls_J_local = w_next * (log_J_local - np.log(np.maximum(1 - fbar_J_local, EPS)))
        cls_C  = w_next * (log_C  - np.log(np.maximum(1 - fbar_C,  EPS)))
        sum_cls +=  cls_C - cls_J_local

        # ----- score differences for this step --------------------
        log_diffs[t] = log_C - log_J
        cs_diffs[t] = cs_C - cs_J_localized  # keep your sign
        cls_diffs[t] = cls_C - cls_J_local

    mean_full = sum_log / P_steps
    mean_localized = sum_cs / P_steps
    mean_local = sum_cls / P_steps

    # DM statistic with safety
    var_full = log_diffs.var(ddof=1)
    var_locd = cs_diffs.var(ddof=1)
    var_local = cls_diffs.var(ddof=1)

    dm_full = mean_full / np.sqrt(var_full / P_steps)
    dm_localized = (0.0 if var_locd < 1e-20 else mean_localized / np.sqrt(var_locd / P_steps))
    dm_local = (0.0 if var_local < 1e-20 else mean_local / np.sqrt(var_local / P_steps))


    return mean_full, mean_localized, mean_local, dm_full, dm_localized, dm_local


# ───────────────── 4. Main driver (parallel) ──────────────────────────
def main() -> None:
    mean_LogS  = np.empty(B_rep)
    mean_CS  = np.empty(B_rep)
    mean_CLS  = np.empty(B_rep)
    dm_full = np.empty(B_rep)
    dm_localized = np.empty(B_rep)
    dm_local = np.empty(B_rep)


    with ProcessPoolExecutor(max_workers=n_proc) as ex:
        futures = [ex.submit(run_one_rep, i) for i in range(B_rep)]
        for i, fut in enumerate(
            tqdm(as_completed(futures), total=B_rep, ncols=80,
                 desc="Rolling-window sims")
        ):
            mean_LogS[i], mean_CS[i], mean_CLS[i], dm_full[i], dm_localized[i], dm_local[i] = fut.result()

    # KDE + histograms --------------------------------------------------
    def show(ax, data, title, color):
        ax.hist(data, bins=40, density=True, alpha=0.6, color=color)
        grid = np.linspace(data.min(), data.max(), 500)
        ax.plot(grid, gaussian_kde(data)(grid), lw=2, color="crimson")
        ax.set_title(title); ax.set_xlabel("difference")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    show(ax[0], mean_LogS, "Mean LogS diff (C – sJoe KL matched)",  "tab:blue")
    show(ax[1], mean_CS, "Mean CS diff (C - sJoe localized KL matched)",   "tab:orange")
    show(ax[2], mean_CLS, "Mean CLS diff (C – sJoe local KL matched)",    "tab:green")
    ax[0].set_ylabel("density")
    plt.tight_layout()
    plt.show()

    print(f"Grand mean LogS diff : {mean_LogS.mean(): .6f}")
    print(f"Grand mean CS diff : {mean_CS.mean(): .6f}")
    print(f"Grand mean CLS diff : {mean_CLS.mean(): .6f}")

    # CDF --------------------------------------------------------------
    def show_cdf(ax, data, title, color):
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax.step(sorted_data, cdf, where="post", label="CDF", color=color)
        ax.set_title(title)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    show_cdf(ax[0], mean_LogS, "Mean LogS diff (C - sJoe KL matched)", "tab:blue")
    show_cdf(ax[1], mean_CS, "Mean CS diff (C - sJoe localized KL matched)", "tab:orange")
    show_cdf(ax[2], mean_CLS, "Mean CLS diff (C - sJoe local KL matched)", "tab:green")
    plt.tight_layout()
    plt.show()

    # Rejection rate + Size discrepancy ---------------------------------------------------
    right_LogS = np.array([(dm_full > norm.ppf(1 - a)).mean() for a in alpha_grid]) - alpha_grid
    left_LogS = np.array([(dm_full < norm.ppf(a)).mean() for a in alpha_grid]) - alpha_grid
    two_sided_LogS = np.array([(np.abs(dm_full) > norm.ppf(1 - a / 2)).mean() for a in alpha_grid]) - alpha_grid

    right_CS = np.array([(dm_localized > norm.ppf(1 - a)).mean() for a in alpha_grid]) - alpha_grid
    left_CS = np.array([(dm_localized < norm.ppf(a)).mean() for a in alpha_grid]) - alpha_grid
    two_sided_CS = np.array([(np.abs(dm_localized) > norm.ppf(1 - a / 2)).mean() for a in alpha_grid]) - alpha_grid

    right_CLS = np.array([(dm_local > norm.ppf(1 - a)).mean() for a in alpha_grid]) - alpha_grid
    left_CLS = np.array([(dm_local < norm.ppf(a)).mean() for a in alpha_grid]) - alpha_grid
    two_sided_CLS = np.array([(np.abs(dm_local) > norm.ppf(1 - a / 2)).mean() for a in alpha_grid]) - alpha_grid

    def show_discrepancy(ax, nominal_sizes, discrepancies, title, color):
        ax.plot(nominal_sizes, discrepancies, color=color)
        ax.plot(nominal_sizes, 1.96 * np.sqrt(nominal_sizes * (1 - nominal_sizes) / len(discrepancies)), color="gray",
                 linestyle="--", linewidth=1)
        ax.plot(nominal_sizes, -1.96 * np.sqrt(nominal_sizes * (1 - nominal_sizes) / len(discrepancies)), color="gray",
                 linestyle="--", linewidth=1)
        ax.set_title(title)

    fig1, ax1 = plt.subplots(1, 3, figsize=(16, 4))
    show_discrepancy(ax1[0], alpha_grid, right_LogS, "Size discrepancy LogS right-tailed", "tab:blue")
    show_discrepancy(ax1[1], alpha_grid, left_LogS, "Size discrepancy LogS left-tailed", "tab:blue")
    show_discrepancy(ax1[2], alpha_grid, two_sided_LogS, "Size discrepancy LogS two-tailed", "tab:blue")
    plt.tight_layout()
    plt.show()

    fig2, ax2 = plt.subplots(1, 3, figsize=(16, 4))
    show_discrepancy(ax2[0], alpha_grid, right_CS, "Size discrepancy LogS right-tailed", "tab:orange")
    show_discrepancy(ax2[1], alpha_grid, left_CS, "Size discrepancy LogS left-tailed", "tab:orange")
    show_discrepancy(ax2[2], alpha_grid, two_sided_CS, "Size discrepancy LogS two-tailed", "tab:orange")
    plt.tight_layout()
    plt.show()

    fig3, ax3 = plt.subplots(1, 3, figsize=(16, 4))
    show_discrepancy(ax3[0], alpha_grid, right_CLS, "Size discrepancy LogS right-tailed", "tab:green")
    show_discrepancy(ax3[1], alpha_grid, left_CLS, "Size discrepancy LogS left-tailed", "tab:green")
    show_discrepancy(ax3[2], alpha_grid, two_sided_CLS, "Size discrepancy LogS two-tailed", "tab:green")
    plt.tight_layout()
    plt.show()

# ───────────────── 5. Multiprocessing guard ───────────────────────────
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
