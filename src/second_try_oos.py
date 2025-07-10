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
def run_one_rep(rep_idx: int):
    # 3.1 simulate R+P PITs
    U_all = sim_sGumbel_PITs(R_window + P_steps, theta_sGumbel)

    # accumulators for P differences
    sum_log, sum_cs, sum_cls = 0.0, 0.0, 0.0
    sum_log_ecdf, sum_cs_ecdf, sum_cls_ecdf = 0.0, 0.0, 0.0

    MC_C = sampler_clayton(n_mc_fbar)   # reusable MC grid for f̄_clayton

    # 2. containers to store the *P* one-step diffs
    log_diffs = np.empty(P_steps)
    cs_diffs = np.empty(P_steps)
    cls_diffs = np.empty(P_steps)

    log_diffs_ecdf = np.empty(P_steps)
    cs_diffs_ecdf = np.empty(P_steps)
    cls_diffs_ecdf = np.empty(P_steps)

    for k, t in enumerate(range(R_window, R_window+P_steps)):            # rolling windows
        U_win  = U_all[k : k+R_window]             # window (R×2)
        U_next = U_all[k+R_window : k+R_window+1]  # next PIT (1×2)


        # ---------- ROI for this window --------------------------------
        y1 = student_t.ppf(U_win[:, 0], df_tail)
        y2 = student_t.ppf(U_win[:, 1], df_tail)
        q_star = np.quantile(y1 + y2, q_threshold)

        mask_win = ((y1 + y2) <= q_star).astype(float)

        def w_fn(x):
            t1 = student_t.ppf(x[:, 0], df_tail)
            t2 = student_t.ppf(x[:, 1], df_tail)
            return ((t1 + t2) <= q_star).astype(float)

        def w_fn_ecdf(y):
            return ((y[:, 0] + y[:, 1]) <= q_star).astype(float)

        # ---------- ECDF PITs for this window ---------------------------
        Y_win = np.column_stack([                   # in this case these are standard innovations
            student_t.ppf(U_win[:, 0], df_tail),
            student_t.ppf(U_win[:, 1], df_tail)
        ])
        Y_next = np.column_stack([                   # in this case these are standard innovations
            student_t.ppf(U_next[:,0], df_tail),
            student_t.ppf(U_next[:,1], df_tail)
        ])

        def ecdf_next(Y_win, Y_next):
            R, d = Y_win.shape
            combined_1 = np.concatenate([Y_win, Y_next])[:,0]
            combined_2 = np.concatenate([Y_win, Y_next])[:,1]
            rank_1 = np.sum(combined_1 <= Y_next[:,0])
            rank_2 = np.sum(combined_2 <= Y_next[:,1])
            U1_ecdf_next = rank_1 / (R + 2)
            U2_ecdf_next = rank_2 / (R + 2)

            return np.column_stack([U1_ecdf_next, U2_ecdf_next])

        def ecdf_window(Y_win):
            R, d = Y_win.shape
            U_ecdf_win = np.empty_like(Y_win)

            for j in range(d):
                ranks = np.argsort(np.argsort(Y_win[:,j]))
                U_ecdf_win[:,j] = (ranks + 1) / (R + 1)

            return U_ecdf_win

        U_next_ecdf = ecdf_next(Y_win, Y_next)
        U_win_ecdf = ecdf_window(Y_win)

        # ---------- re-tune θ_sJoe* on this window ----------------------
        theta_J, theta_J_localized, theta_J_local = tune_sJoe_params(
            [U_win], [mask_win], pdf_truth, pdf_clayton, verbose=False
        )

        theta_J_ecdf, theta_J_localized_ecdf, theta_J_local_ecdf = tune_sJoe_params(
            [U_win_ecdf], [mask_win], pdf_truth, pdf_clayton, verbose=False
        )

        pdf_J  = lambda u: sJoe_copula_pdf_from_PITs(u, theta_J)
        pdf_J_localized = lambda u: sJoe_copula_pdf_from_PITs(u, theta_J_localized)
        pdf_J_local = lambda u: sJoe_copula_pdf_from_PITs(u, theta_J_local)

        pdf_J_ecdf = lambda u: sJoe_copula_pdf_from_PITs(u, theta_J_ecdf)
        pdf_J_localized_ecdf = lambda u: sJoe_copula_pdf_from_PITs(u, theta_J_localized_ecdf)
        pdf_J_local_ecdf = lambda u: sJoe_copula_pdf_from_PITs(u, theta_J_local_ecdf)

        def sampler_J_localized(n): return sim_sJoe_PITs(n, theta_J_localized)
        sampler_J_localized.pdf = pdf_J_localized
        def sampler_J_local(n): return sim_sJoe_PITs(n, theta_J_local)
        sampler_J_local.pdf = pdf_J_local

        def sampler_J_localized_ecdf(n): return sim_sJoe_PITs(n, theta_J_localized_ecdf)
        sampler_J_localized_ecdf.pdf = pdf_J_localized_ecdf
        def sampler_J_local_ecdf(n): return sim_sJoe_PITs(n, theta_J_local_ecdf)
        sampler_J_local_ecdf.pdf = pdf_J_local_ecdf

        # ---------- fresh f̄_w(f) for the same ROI ----------------------
        fbar_C  = estimate_fbar(pdf_clayton, w_fn, sampler_clayton, n=n_mc_fbar)
        fbar_C_ecdf = estimate_fbar(pdf_clayton, w_fn_ecdf(Y_win), sampler_clayton, n=n_mc_fbar, ecdf=True)

        fbar_J_localized = estimate_fbar(pdf_J_localized,      w_fn, sampler_J_localized,      n=n_mc_fbar)
        fbar_J_local = estimate_fbar(pdf_J_local,      w_fn, sampler_J_local,      n=n_mc_fbar)

        fbar_J_localized_ecdf = estimate_fbar(pdf_J_localized_ecdf, w_fn_ecdf(Y_win), sampler_J_localized_ecdf, n=n_mc_fbar, ecdf=True)
        fbar_J_local_ecdf = estimate_fbar(pdf_J_local_ecdf, w_fn_ecdf(Y_win), sampler_J_local_ecdf, n=n_mc_fbar, ecdf=True)

        # ---------- manual scores for the one-step PIT ------------------
        w_next = w_fn(U_next)[0]                      # 0 or 1
        w_next_ecdf = w_fn_ecdf(Y_next)[0]

        log_C = np.log(np.maximum(pdf_clayton(U_next)[0], EPS))
        log_J = np.log(np.maximum(pdf_J(U_next)[0],       EPS))
        log_J_localized= np.log(np.maximum(pdf_J_localized(U_next)[0],      EPS))
        log_J_local= np.log(np.maximum(pdf_J_local(U_next)[0],      EPS))

        log_C_ecdf = np.log(np.maximum(pdf_clayton(U_next_ecdf)[0], EPS))
        log_J_ecdf = np.log(np.maximum(pdf_J_ecdf(U_next_ecdf)[0], EPS))
        log_J_localized_ecdf = np.log(np.maximum(pdf_J_localized_ecdf(U_next_ecdf)[0], EPS))
        log_J_local_ecdf = np.log(np.maximum(pdf_J_local_ecdf(U_next_ecdf)[0], EPS))

        # Log-score difference
        sum_log += log_C - log_J
        sum_log_ecdf += log_C_ecdf - log_J_ecdf

        # CS difference
        cs_J_localized = w_next * log_J_localized + (1 - w_next) * np.log(np.maximum(fbar_J_localized, EPS))
        cs_C  = w_next * log_C  + (1 - w_next) * np.log(np.maximum(fbar_C,  EPS))

        cs_J_localized_ecdf = w_next_ecdf * log_J_localized_ecdf + (1 - w_next_ecdf) * np.log(np.maximum(fbar_J_localized_ecdf, EPS))
        cs_C_ecdf = w_next_ecdf * log_C_ecdf + (1 - w_next_ecdf) * np.log(np.maximum(fbar_C_ecdf, EPS))

        sum_cs +=  cs_C - cs_J_localized
        sum_cs_ecdf += cs_C_ecdf - cs_J_localized_ecdf

        # CLS difference
        cls_J_local = w_next * (log_J_local - np.log(np.maximum(1 - fbar_J_local, EPS)))
        cls_C  = w_next * (log_C  - np.log(np.maximum(1 - fbar_C,  EPS)))

        cls_J_local_ecdf = w_next_ecdf * (log_J_local_ecdf - np.log(np.maximum(1 - fbar_J_local_ecdf, EPS)))
        cls_C_ecdf = w_next_ecdf * (log_C_ecdf - np.log(np.maximum(1 - fbar_C_ecdf, EPS)))

        sum_cls +=  cls_C - cls_J_local
        sum_cls_ecdf += cls_C_ecdf - cls_J_local_ecdf

        # ----- score differences for this step --------------------
        log_diffs[t] = log_C - log_J
        cs_diffs[t] = cs_C - cs_J_localized
        cls_diffs[t] = cls_C - cls_J_local

        log_diffs_ecdf[t] = log_C_ecdf - log_J_ecdf
        cs_diffs_ecdf[t] = cs_C_ecdf - cs_J_localized_ecdf
        cls_diffs_ecdf[t] = cls_C_ecdf - cls_J_local_ecdf

    mean_full = sum_log / P_steps
    mean_localized = sum_cs / P_steps
    mean_local = sum_cls / P_steps

    mean_full_ecdf = sum_log_ecdf / P_steps
    mean_localized_ecdf = sum_cs_ecdf / P_steps
    mean_local_ecdf = sum_cls_ecdf / P_steps

    # DM statistic with safety
    var_full = log_diffs.var(ddof=1)
    var_locd = cs_diffs.var(ddof=1)
    var_local = cls_diffs.var(ddof=1)

    var_full_ecdf = log_diffs_ecdf.var(ddof=1)
    var_locd_ecdf = cs_diffs_ecdf.var(ddof=1)
    var_local_ecdf = cls_diffs_ecdf.var(ddof=1)

    dm_full = mean_full / np.sqrt(var_full / P_steps)
    dm_localized = (0.0 if var_locd < 1e-20 else mean_localized / np.sqrt(var_locd / P_steps))
    dm_local = (0.0 if var_local < 1e-20 else mean_local / np.sqrt(var_local / P_steps))

    dm_full_ecdf = mean_full_ecdf / np.sqrt(var_full_ecdf / P_steps)
    dm_localized_ecdf = (0.0 if var_locd_ecdf < 1e-20 else mean_localized_ecdf / np.sqrt(var_locd_ecdf / P_steps))
    dm_local_ecdf = (0.0 if var_local_ecdf < 1e-20 else mean_local_ecdf / np.sqrt(var_local_ecdf / P_steps))


    return (mean_full, mean_localized, mean_local, dm_full, dm_localized, dm_local,
            mean_full_ecdf, mean_localized_ecdf, mean_local_ecdf, dm_full_ecdf, dm_localized_ecdf, dm_local_ecdf)


# ───────────────── 4. Main driver (parallel) ──────────────────────────
def main() -> None:
    mean_LogS  = np.empty(B_rep)
    mean_CS  = np.empty(B_rep)
    mean_CLS  = np.empty(B_rep)

    mean_LogS_ecdf = np.empty(B_rep)
    mean_CS_ecdf = np.empty(B_rep)
    mean_CLS_ecdf = np.empty(B_rep)

    dm_full = np.empty(B_rep)
    dm_localized = np.empty(B_rep)
    dm_local = np.empty(B_rep)

    dm_full_ecdf = np.empty(B_rep)
    dm_localized_ecdf = np.empty(B_rep)
    dm_local_ecdf = np.empty(B_rep)


    with (ProcessPoolExecutor(max_workers=n_proc) as ex):
        futures = [ex.submit(run_one_rep, i) for i in range(B_rep)]
        for i, fut in enumerate(
            tqdm(as_completed(futures), total=B_rep, ncols=80,
                 desc="Rolling-window sims")
        ):
            (mean_LogS[i], mean_CS[i], mean_CLS[i], dm_full[i], dm_localized[i], dm_local[i],
            mean_LogS_ecdf[i], mean_CS_ecdf[i], mean_CLS_ecdf[i], dm_full_ecdf[i], dm_localized_ecdf[i], dm_local_ecdf[i]) = fut.result()

    # KDE + histograms --------------------------------------------------
    def show_kde_hist(ax, data1, data2, label1, label2, color1, color2):
        ax.hist(data1, bins=40, density=True, alpha=0.4, color=color1, label=label1)
        ax.hist(data2, bins=40, density=True, alpha=0.4, color=color2, label=label2)
        grid = np.linspace(min(data1.min(), data2.min()), max(data1.max(), data2.max()), 500)
        ax.plot(grid, gaussian_kde(data1)(grid), lw=2, color=color1)
        ax.plot(grid, gaussian_kde(data2)(grid), lw=2, color=color2)
        ax.set_xlabel("difference")
        ax.legend()

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    show_kde_hist(ax[0], mean_LogS, mean_LogS_ecdf, "Oracle", "ECDF", "tab:blue", "navy")
    show_kde_hist(ax[1], mean_CS, mean_CS_ecdf, "Oracle", "ECDF", "tab:orange", "darkorange")
    show_kde_hist(ax[2], mean_CLS, mean_CLS_ecdf, "Oracle", "ECDF", "tab:green", "seagreen")

    ax[0].set_title("Mean LogS diff (Clayton – sJoe KL matched)")
    ax[1].set_title("Mean CS diff (Clayton – sJoe localized KL matched)")
    ax[2].set_title("Mean CLS diff (Clayton – sJoe local KL matched)")

    ax[0].set_ylabel("density")
    fig.suptitle("Oracle vs ECDF", fontsize=16)
    plt.tight_layout()
    plt.show()

    print(f"Oracle grand mean LogS diff : {mean_LogS.mean(): .6f}")
    print(f"Oracle grand mean CS diff : {mean_CS.mean(): .6f}")
    print(f"Oracle grand mean CLS diff : {mean_CLS.mean(): .6f}")

    print(f"ECDF grand mean LogS diff : {mean_LogS_ecdf.mean(): .6f}")
    print(f"ECDF grand mean CS diff : {mean_CS_ecdf.mean(): .6f}")
    print(f"ECDF grand mean CLS diff : {mean_CLS_ecdf.mean(): .6f}")

    # CDF --------------------------------------------------------------
    def show_cdf(ax, data1, data2, label1, label2, color1, color2):
        sorted_data1 = np.sort(data1)
        sorted_data2 = np.sort(data2)
        cdf1 = np.arange(1, len(sorted_data1) + 1) / len(sorted_data1)
        cdf2 = np.arange(1, len(sorted_data2) + 1) / len(sorted_data2)
        ax.step(sorted_data1, cdf1, where="post", label=label1, color=color1)
        ax.step(sorted_data2, cdf2, where="post", label=label2, color=color2)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    show_cdf(ax[0], mean_LogS, mean_LogS_ecdf, "Oracle", "ECDF", "tab:blue", "navy")
    show_cdf(ax[1], mean_CS, mean_CS_ecdf, "Oracle", "ECDF","tab:orange", "darkorange")
    show_cdf(ax[2], mean_CLS, mean_CLS_ecdf, "Oracle", "ECDF","tab:green", "seagreen")

    ax[0].set_title("Mean LogS diff (Clayton – sJoe KL matched)")
    ax[1].set_title("Mean CS diff (Clayton – sJoe localized KL matched)")
    ax[2].set_title("Mean CLS diff (Clayton – sJoe local KL matched)")

    ax[0].set_ylabel("CDF")
    fig.suptitle("Oracle vs ECDF", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Rejection rate + Size discrepancy ---------------------------------------------------
    right_LogS = np.array([(dm_full > norm.ppf(1 - a)).mean() for a in alpha_grid]) - alpha_grid
    left_LogS = np.array([(dm_full < norm.ppf(a)).mean() for a in alpha_grid]) - alpha_grid
    two_sided_LogS = np.array([(np.abs(dm_full) > norm.ppf(1 - a / 2)).mean() for a in alpha_grid]) - alpha_grid

    right_LogS_ecdf = np.array([(dm_full_ecdf > norm.ppf(1 - a)).mean() for a in alpha_grid]) - alpha_grid
    left_LogS_ecdf = np.array([(dm_full_ecdf < norm.ppf(a)).mean() for a in alpha_grid]) - alpha_grid
    two_sided_LogS_ecdf = np.array([(np.abs(dm_full_ecdf) > norm.ppf(1 - a / 2)).mean() for a in alpha_grid]) - alpha_grid

    right_CS = np.array([(dm_localized > norm.ppf(1 - a)).mean() for a in alpha_grid]) - alpha_grid
    left_CS = np.array([(dm_localized < norm.ppf(a)).mean() for a in alpha_grid]) - alpha_grid
    two_sided_CS = np.array([(np.abs(dm_localized) > norm.ppf(1 - a / 2)).mean() for a in alpha_grid]) - alpha_grid

    right_CS_ecdf = np.array([(dm_localized_ecdf > norm.ppf(1 - a)).mean() for a in alpha_grid]) - alpha_grid
    left_CS_ecdf = np.array([(dm_localized_ecdf < norm.ppf(a)).mean() for a in alpha_grid]) - alpha_grid
    two_sided_CS_ecdf = np.array([(np.abs(dm_localized_ecdf) > norm.ppf(1 - a / 2)).mean() for a in alpha_grid]) - alpha_grid

    right_CLS = np.array([(dm_local > norm.ppf(1 - a)).mean() for a in alpha_grid]) - alpha_grid
    left_CLS = np.array([(dm_local < norm.ppf(a)).mean() for a in alpha_grid]) - alpha_grid
    two_sided_CLS = np.array([(np.abs(dm_local) > norm.ppf(1 - a / 2)).mean() for a in alpha_grid]) - alpha_grid

    right_CLS_ecdf = np.array([(dm_local_ecdf > norm.ppf(1 - a)).mean() for a in alpha_grid]) - alpha_grid
    left_CLS_ecdf = np.array([(dm_local_ecdf < norm.ppf(a)).mean() for a in alpha_grid]) - alpha_grid
    two_sided_CLS_ecdf = np.array([(np.abs(dm_local_ecdf) > norm.ppf(1 - a / 2)).mean() for a in alpha_grid]) - alpha_grid

    def show_discrepancy(ax, nominal_sizes, discrepancies1, discrepancies2, label1, label2, color1, color2):
        ax.plot(nominal_sizes, discrepancies1, color=color1)
        ax.plot(nominal_sizes, discrepancies2, color=color2)
        ax.plot(nominal_sizes, 1.96 * np.sqrt(nominal_sizes * (1 - nominal_sizes) / len(discrepancies1)), color="gray",
                 linestyle="--", linewidth=1)
        ax.plot(nominal_sizes, -1.96 * np.sqrt(nominal_sizes * (1 - nominal_sizes) / len(discrepancies1)), color="gray",
                 linestyle="--", linewidth=1)
        plt.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Nominal size")

    fig1, ax1 = plt.subplots(1, 3, figsize=(16, 4))
    show_discrepancy(ax1[0], alpha_grid, right_LogS, right_LogS_ecdf, "Oracle", "ECDF", "tab:blue", "navy")
    show_discrepancy(ax1[1], alpha_grid, left_LogS, left_LogS_ecdf, "Oracle", "ECDF", "tab:blue", "navy")
    show_discrepancy(ax1[2], alpha_grid, two_sided_LogS, two_sided_LogS_ecdf, "Oracle", "ECDF", "tab:blue", "navy")
    ax1[0].set_title("Size discrepancy (Clayton – sJoe KL matched) right-tailed")
    ax1[1].set_title("Size discrepancy (Clayton – sJoe KL matched) left-tailed")
    ax1[2].set_title("Size discrepancy (Clayton – sJoe KL matched) two-tailed")
    ax1[0].set_ylabel("Size discrepancy")
    fig1.suptitle("Size discrepancies: Oracle vs ECDF", fontsize=16)
    plt.tight_layout()
    plt.show()

    fig2, ax2 = plt.subplots(1, 3, figsize=(16, 4))
    show_discrepancy(ax2[0], alpha_grid, right_CS, right_CS_ecdf, "Oracle", "ECDF", "tab:orange", "darkorange")
    show_discrepancy(ax2[1], alpha_grid, left_CS, left_CS_ecdf,"Oracle", "ECDF", "tab:orange", "darkorange")
    show_discrepancy(ax2[2], alpha_grid, two_sided_CS, two_sided_CS_ecdf,"Oracle", "ECDF", "tab:orange", "darkorange")
    ax2[0].set_title("Size discrepancy (Clayton – sJoe localized KL matched) right-tailed")
    ax2[1].set_title("Size discrepancy (Clayton – sJoe localized KL matched) left-tailed")
    ax2[2].set_title("Size discrepancy (Clayton – sJoe localized KL matched) two-tailed")
    ax2[0].set_ylabel("Size discrepancy")
    fig2.suptitle("Size discrepancies: Oracle vs ECDF", fontsize=16)
    plt.tight_layout()
    plt.show()

    fig3, ax3 = plt.subplots(1, 3, figsize=(16, 4))
    show_discrepancy(ax3[0], alpha_grid, right_CLS, right_CLS_ecdf, "Oracle", "ECDF", "tab:green", "seagreen")
    show_discrepancy(ax3[1], alpha_grid, left_CLS, left_CLS_ecdf,"Oracle", "ECDF", "tab:green", "seagreen")
    show_discrepancy(ax3[2], alpha_grid, two_sided_CLS, two_sided_CLS_ecdf,"Oracle", "ECDF", "tab:green", "seagreen")
    ax3[0].set_title("Size discrepancy (Clayton – sJoe local KL matched) right-tailed")
    ax3[1].set_title("Size discrepancy (Clayton – sJoe local KL matched) left-tailed")
    ax3[2].set_title("Size discrepancy (Clayton – sJoe local KL matched) two-tailed")
    ax3[0].set_ylabel("Size discrepancy")
    fig3.suptitle("Size discrepancies: Oracle vs ECDF", fontsize=16)
    plt.tight_layout()
    plt.show()

# ───────────────── 5. Multiprocessing guard ───────────────────────────
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
