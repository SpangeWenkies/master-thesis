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
from utils.divergences import full_kl, localised_kl, local_kl
from utils.optimize_utils import tune_sJoe_given_target
from utils.focused_scores import estimate_fbar        # only for f̄_w

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as student_t, gaussian_kde, norm
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing, os

# ───────────────── 1. Configuration ───────────────────────────────────
theta_sGumbel = 1.5
theta_Clayton = 8
df_tail       = 5
q_threshold   = 0.05

R_window      = 100        # estimation window length
P_steps       = 50        # one-step forecasts per replication
B_rep         = 20        # Monte-Carlo replications

n_points = 5

n_proc        = min(32, os.cpu_count() or 1)
n_mc_fbar     = 500       # MC points for f̄_w(f) each window
EPS           = 1e-12        # safe-log constant

nominal_size    = 0.05

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
    sum_log, sum_cs, sum_cls = np.zeros(n_points), np.zeros(n_points), np.zeros(n_points)
    sum_log_ecdf, sum_cs_ecdf, sum_cls_ecdf = np.zeros(n_points), np.zeros(n_points), np.zeros(n_points)

    MC_C = sampler_clayton(n_mc_fbar)   # reusable MC grid for f̄_clayton

    # 2. containers to store the *P* one-step diffs
    log_diffs = np.empty((n_points, P_steps))
    cs_diffs = np.empty((n_points, P_steps))
    cls_diffs = np.empty((n_points, P_steps))

    log_diffs_ecdf = np.empty((n_points, P_steps))
    cs_diffs_ecdf = np.empty((n_points, P_steps))
    cls_diffs_ecdf = np.empty((n_points, P_steps))

    for k, t in enumerate(range(R_window, R_window+P_steps)):            # rolling windows
        U_win  = U_all[k : k+R_window]             # window (R×2)
        U_next = U_all[k+R_window : k+R_window+1]  # next PIT (1×2)

        # ---------- ROI & observed marginals for this window --------------------------------
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

        # ---------- calculate the target KL -----------------------------
        clayton_kl = full_kl(U_win, pdf_truth, pdf_clayton)
        clayton_kl_localized = localised_kl(U_win, pdf_truth, pdf_clayton, mask_win)
        clayton_kl_local = local_kl(U_win, pdf_truth, pdf_clayton, mask_win)

        clayton_kl_ecdf = full_kl(U_win_ecdf, pdf_truth, pdf_clayton)
        clayton_kl_localized_ecdf = localised_kl(U_win_ecdf, pdf_truth, pdf_clayton, mask_win)
        clayton_kl_local_ecdf = local_kl(U_win_ecdf, pdf_truth, pdf_clayton, mask_win)

        # ---------- re-tune θ_sJoe* on this window such that it matches eval points on zero to target KL------------
        eval_clayton_kl = [clayton_kl * (i + 1) / n_points for i in range(n_points)]
        eval_clayton_kl_localized = [clayton_kl_localized * (i + 1) / n_points for i in range(n_points)]
        eval_clayton_kl_local = [clayton_kl_local * (i + 1) / n_points for i in range(n_points)]

        eval_clayton_kl_ecdf = [clayton_kl_ecdf * (i + 1) / n_points for i in range(n_points)]
        eval_clayton_kl_localized_ecdf = [clayton_kl_localized_ecdf * (i + 1) / n_points for i in range(n_points)]
        eval_clayton_kl_local_ecdf = [clayton_kl_local_ecdf * (i + 1)/ n_points for i in range(n_points)]

        theta_J = np.empty(n_points)
        theta_J_localized = np.empty(n_points)
        theta_J_local = np.empty(n_points)

        theta_J_ecdf = np.empty(n_points)
        theta_J_localized_ecdf = np.empty(n_points)
        theta_J_local_ecdf = np.empty(n_points)

        pdf_J = np.empty(n_points, dtype=object)
        pdf_J_localized = np.empty(n_points, dtype=object)
        pdf_J_local = np.empty(n_points, dtype=object)

        pdf_J_ecdf = np.empty(n_points, dtype=object)
        pdf_J_localized_ecdf = np.empty(n_points, dtype=object)
        pdf_J_local_ecdf = np.empty(n_points, dtype=object)

        sampler_J_localized = np.empty(n_points, dtype=object)
        sampler_J_local = np.empty(n_points, dtype=object)

        sampler_J_localized_ecdf = np.empty(n_points, dtype=object)
        sampler_J_local_ecdf = np.empty(n_points, dtype=object)

        fbar_C = np.empty(n_points)
        fbar_C_ecdf = np.empty(n_points)
        fbar_J_localized = np.empty(n_points)
        fbar_J_local = np.empty(n_points)
        fbar_J_localized_ecdf = np.empty(n_points)
        fbar_J_local_ecdf = np.empty(n_points)

        log_C = np.empty(n_points)
        log_C_ecdf = np.empty(n_points)

        log_J = np.empty(n_points)
        log_J_localized = np.empty(n_points)
        log_J_local = np.empty(n_points)

        log_J_ecdf = np.empty(n_points)
        log_J_localized_ecdf = np.empty(n_points)
        log_J_local_ecdf = np.empty(n_points)

        cs_C = np.empty(n_points)
        cs_C_ecdf = np.empty(n_points)

        cs_J_localized = np.empty(n_points)
        cs_J_localized_ecdf = np.empty(n_points)

        cls_C = np.empty(n_points)
        cls_C_ecdf = np.empty(n_points)

        cls_J_local = np.empty(n_points)
        cls_J_local_ecdf = np.empty(n_points)

        optim_kl = np.empty(n_points)
        optim_kl_loc = np.empty(n_points)
        optim_kl_local = np.empty(n_points)

        optim_kl_ecdf = np.empty(n_points)
        optim_kl_loc_ecdf = np.empty(n_points)
        optim_kl_local_ecdf = np.empty(n_points)

        kl_full_acc = np.zeros(n_points)
        kl_localized_acc = np.zeros(n_points)
        kl_local_acc = np.zeros(n_points)

        kl_full_ecdf_acc = np.zeros(n_points)
        kl_localized_ecdf_acc = np.zeros(n_points)
        kl_local_ecdf_acc = np.zeros(n_points)

        def make_sampler(theta, pdf):
            def sampler(n, theta=theta):  # capture current theta
                return sim_sJoe_PITs(n, theta)
            sampler.pdf = pdf
            return sampler

        for i in range(n_points):
            theta_J[i], theta_J_localized[i], theta_J_local[i], optim_kl[i], optim_kl_loc[i], optim_kl_local[i] = tune_sJoe_given_target(
                [U_win], [mask_win], pdf_truth,
                eval_clayton_kl[i], eval_clayton_kl_localized[i], eval_clayton_kl_local[i], verbose=True
            )
            theta_J_ecdf[i], theta_J_localized_ecdf[i], theta_J_local_ecdf[i], optim_kl_ecdf[i], optim_kl_loc_ecdf[i], optim_kl_local_ecdf[i] = tune_sJoe_given_target(
                [U_win_ecdf], [mask_win], pdf_truth,
                eval_clayton_kl_ecdf[i], eval_clayton_kl_localized_ecdf[i], eval_clayton_kl_local_ecdf[i], verbose=False
            )

            kl_full_acc[i] += optim_kl[i]
            kl_localized_acc[i] += optim_kl_loc[i]
            kl_local_acc[i] += optim_kl_local[i]

            kl_full_ecdf_acc[i] += optim_kl_ecdf[i]
            kl_localized_ecdf_acc[i] += optim_kl_loc_ecdf[i]
            kl_local_ecdf_acc[i] += optim_kl_local_ecdf[i]

            pdf_J[i]  = (lambda theta: lambda u: sJoe_copula_pdf_from_PITs(u, theta))(theta_J[i])
            pdf_J_localized[i] = (lambda theta: lambda u: sJoe_copula_pdf_from_PITs(u, theta))(theta_J_localized[i])
            pdf_J_local[i] = (lambda theta: lambda u: sJoe_copula_pdf_from_PITs(u, theta))(theta_J_local[i])

            pdf_J_ecdf[i] = (lambda theta: lambda u: sJoe_copula_pdf_from_PITs(u, theta))(theta_J_ecdf[i])
            pdf_J_localized_ecdf[i] = (lambda theta: lambda u: sJoe_copula_pdf_from_PITs(u, theta))(theta_J_localized_ecdf[i])
            pdf_J_local_ecdf[i] = (lambda theta: lambda u: sJoe_copula_pdf_from_PITs(u, theta))(theta_J_local_ecdf[i])

            sampler_J_localized[i] = make_sampler(theta_J_localized[i], pdf_J_localized[i])
            sampler_J_local[i] = make_sampler(theta_J_local[i], pdf_J_local[i])

            sampler_J_localized_ecdf[i] = make_sampler(theta_J_localized_ecdf[i], pdf_J_localized_ecdf[i])
            sampler_J_local_ecdf[i] = make_sampler(theta_J_local_ecdf[i], pdf_J_local_ecdf[i])

            # ---------- fresh f̄_w(f) for the same ROI ----------------------
            fbar_C[i]  = estimate_fbar(pdf_clayton, sampler_clayton, w_fn, n=n_mc_fbar)
            fbar_C_ecdf[i] = estimate_fbar(pdf_clayton, sampler_clayton, n=n_mc_fbar, ecdf=True, df_tail=df_tail, q_threshold=q_threshold)

            fbar_J_localized[i] = estimate_fbar(pdf_J_localized[i], sampler_J_localized[i], w_fn, n=n_mc_fbar)
            fbar_J_local[i] = estimate_fbar(pdf_J_local[i], sampler_J_local[i], w_fn, n=n_mc_fbar)

            fbar_J_localized_ecdf[i] = estimate_fbar(pdf_J_localized_ecdf[i], sampler_J_localized_ecdf[i], n=n_mc_fbar, ecdf=True, df_tail=df_tail, q_threshold=q_threshold)
            fbar_J_local_ecdf[i] = estimate_fbar(pdf_J_local_ecdf[i], sampler_J_local_ecdf[i], n=n_mc_fbar, ecdf=True, df_tail=df_tail, q_threshold=q_threshold)

            # ---------- manual scores for the one-step PIT ------------------
            w_next = w_fn(U_next)[0]                      # 0 or 1
            w_next_ecdf = w_fn_ecdf(Y_next)[0]

            log_C[i] = np.log(np.maximum(pdf_clayton(U_next)[0], EPS))
            log_J[i] = np.log(np.maximum(pdf_J[i](U_next)[0], EPS))
            log_J_localized[i]= np.log(np.maximum(pdf_J_localized[i](U_next)[0], EPS))
            log_J_local[i]= np.log(np.maximum(pdf_J_local[i](U_next)[0], EPS))

            log_C_ecdf[i] = np.log(np.maximum(pdf_clayton(U_next_ecdf)[0], EPS))
            log_J_ecdf[i] = np.log(np.maximum(pdf_J_ecdf[i](U_next_ecdf)[0], EPS))
            log_J_localized_ecdf[i] = np.log(np.maximum(pdf_J_localized_ecdf[i](U_next_ecdf)[0], EPS))
            log_J_local_ecdf[i] = np.log(np.maximum(pdf_J_local_ecdf[i](U_next_ecdf)[0], EPS))

            # Log-score difference
            sum_log[i] += log_C[i] - log_J[i]
            sum_log_ecdf[i] += log_C_ecdf[i] - log_J_ecdf[i]

            # CS difference
            cs_J_localized[i] = w_next * log_J_localized[i] + (1 - w_next) * np.log(np.maximum(fbar_J_localized[i], EPS))
            cs_C[i]  = w_next * log_C[i]  + (1 - w_next) * np.log(np.maximum(fbar_C[i],  EPS))

            cs_J_localized_ecdf[i] = w_next_ecdf * log_J_localized_ecdf[i] + (1 - w_next_ecdf) * np.log(np.maximum(fbar_J_localized_ecdf[i], EPS))
            cs_C_ecdf[i] = w_next_ecdf * log_C_ecdf[i] + (1 - w_next_ecdf) * np.log(np.maximum(fbar_C_ecdf[i], EPS))

            sum_cs[i] +=  cs_C[i] - cs_J_localized[i]
            sum_cs_ecdf[i] += cs_C_ecdf[i] - cs_J_localized_ecdf[i]

            # CLS difference
            cls_J_local[i] = w_next * (log_J_local[i] - np.log(np.maximum(1 - fbar_J_local[i], EPS)))
            cls_C[i]  = w_next * (log_C[i]  - np.log(np.maximum(1 - fbar_C[i],  EPS)))

            cls_J_local_ecdf[i] = w_next_ecdf * (log_J_local_ecdf[i] - np.log(np.maximum(1 - fbar_J_local_ecdf[i], EPS)))
            cls_C_ecdf[i] = w_next_ecdf * (log_C_ecdf[i] - np.log(np.maximum(1 - fbar_C_ecdf[i], EPS)))

            sum_cls[i] +=  cls_C[i] - cls_J_local[i]
            sum_cls_ecdf[i] += cls_C_ecdf[i] - cls_J_local_ecdf[i]

            # ----- score differences for this step --------------------
            log_diffs[i][k] = log_C[i] - log_J[i]
            cs_diffs[i][k] = cs_C[i] - cs_J_localized[i]
            cls_diffs[i][k] = cls_C[i] - cls_J_local[i]

            log_diffs_ecdf[i][k] = log_C_ecdf[i] - log_J_ecdf[i]
            cs_diffs_ecdf[i][k] = cs_C_ecdf[i] - cs_J_localized_ecdf[i]
            cls_diffs_ecdf[i][k] = cls_C_ecdf[i] - cls_J_local_ecdf[i]

        print(k)

    mean_kl_full = np.empty(n_points)
    mean_kl_localized = np.empty(n_points)
    mean_kl_local = np.empty(n_points)
    mean_kl_full_ecdf = np.empty(n_points)
    mean_kl_localized_ecdf = np.empty(n_points)
    mean_kl_local_ecdf = np.empty(n_points)

    mean_full = np.empty(n_points)
    mean_localized = np.empty(n_points)
    mean_local = np.empty(n_points)

    mean_full_ecdf = np.empty(n_points)
    mean_localized_ecdf = np.empty(n_points)
    mean_local_ecdf = np.empty(n_points)

    var_full = np.empty(n_points)
    var_locd = np.empty(n_points)
    var_local = np.empty(n_points)

    var_full_ecdf = np.empty(n_points)
    var_locd_ecdf = np.empty(n_points)
    var_local_ecdf = np.empty(n_points)

    dm_full = np.empty(n_points)
    dm_localized = np.empty(n_points)
    dm_local = np.empty(n_points)

    dm_full_ecdf = np.empty(n_points)
    dm_localized_ecdf = np.empty(n_points)
    dm_local_ecdf = np.empty(n_points)

    for i in range(n_points):
        mean_kl_full[i] = kl_full_acc[i] / P_steps
        mean_kl_localized[i] = kl_localized_acc[i] / P_steps
        mean_kl_local[i] = kl_local_acc[i] / P_steps

        mean_kl_full_ecdf[i] = kl_full_ecdf_acc[i] / P_steps
        mean_kl_localized_ecdf[i] = kl_localized_ecdf_acc[i] / P_steps
        mean_kl_local_ecdf[i] = kl_local_ecdf_acc[i] / P_steps

        mean_full[i] = sum_log[i] / P_steps
        mean_localized[i] = sum_cs[i] / P_steps
        mean_local[i] = sum_cls[i] / P_steps

        mean_full_ecdf[i] = sum_log_ecdf[i] / P_steps
        mean_localized_ecdf[i] = sum_cs_ecdf[i] / P_steps
        mean_local_ecdf[i] = sum_cls_ecdf[i] / P_steps

        # DM statistic with safety
        var_full[i] = log_diffs[i].var(ddof=1)
        var_locd[i] = cs_diffs[i].var(ddof=1)
        var_local[i] = cls_diffs[i].var(ddof=1)

        var_full_ecdf[i] = log_diffs_ecdf[i].var(ddof=1)
        var_locd_ecdf[i] = cs_diffs_ecdf[i].var(ddof=1)
        var_local_ecdf[i] = cls_diffs_ecdf[i].var(ddof=1)

        dm_full[i] = mean_full[i] / np.sqrt(var_full[i] / P_steps)
        dm_localized[i] = (0.0 if var_locd[i] < 1e-20 else mean_localized[i] / np.sqrt(var_locd[i] / P_steps))
        dm_local[i] = (0.0 if var_local[i] < 1e-20 else mean_local[i] / np.sqrt(var_local[i] / P_steps))

        dm_full_ecdf[i] = mean_full_ecdf[i] / np.sqrt(var_full_ecdf[i] / P_steps)
        dm_localized_ecdf[i] = (0.0 if var_locd_ecdf[i] < 1e-20 else mean_localized_ecdf[i] / np.sqrt(var_locd_ecdf[i] / P_steps))
        dm_local_ecdf[i] = (0.0 if var_local_ecdf[i] < 1e-20 else mean_local_ecdf[i] / np.sqrt(var_local_ecdf[i] / P_steps))


    return (mean_full, mean_localized, mean_local, dm_full, dm_localized, dm_local,
            mean_full_ecdf, mean_localized_ecdf, mean_local_ecdf, dm_full_ecdf, dm_localized_ecdf, dm_local_ecdf,
            mean_kl_full, mean_kl_localized, mean_kl_local, mean_kl_full_ecdf, mean_kl_localized_ecdf, mean_kl_local_ecdf)


# ───────────────── 4. Main driver (parallel) ──────────────────────────
def main() -> None:
    mean_LogS  = np.empty((n_points,B_rep))
    mean_CS  = np.empty((n_points,B_rep))
    mean_CLS  = np.empty((n_points,B_rep))

    mean_LogS_ecdf = np.empty((n_points,B_rep))
    mean_CS_ecdf = np.empty((n_points,B_rep))
    mean_CLS_ecdf = np.empty((n_points,B_rep))

    dm_full = np.empty((n_points,B_rep))
    dm_localized = np.empty((n_points,B_rep))
    dm_local = np.empty((n_points,B_rep))

    dm_full_ecdf = np.empty((n_points,B_rep))
    dm_localized_ecdf = np.empty((n_points,B_rep))
    dm_local_ecdf = np.empty((n_points,B_rep))

    mean_kl_full = np.empty((n_points,B_rep))
    mean_kl_localized = np.empty((n_points,B_rep))
    mean_kl_local = np.empty((n_points,B_rep))

    mean_kl_full_ecdf = np.empty((n_points,B_rep))
    mean_kl_localized_ecdf = np.empty((n_points,B_rep))
    mean_kl_local_ecdf = np.empty((n_points,B_rep))


    with (ProcessPoolExecutor(max_workers=n_proc) as ex):
        futures = [ex.submit(run_one_rep, i) for i in range(B_rep)]
        for i, fut in enumerate(
            tqdm(as_completed(futures), total=B_rep, ncols=80,
                 desc="Rolling-window sims")
        ):
            (mean_LogS[:,i], mean_CS[:,i], mean_CLS[:,i], dm_full[:,i], dm_localized[:,i], dm_local[:,i],
             mean_LogS_ecdf[:,i], mean_CS_ecdf[:,i], mean_CLS_ecdf[:,i], dm_full_ecdf[:,i], dm_localized_ecdf[:,i], dm_local_ecdf[:,i],
             mean_kl_full[:,i], mean_kl_localized[:,i], mean_kl_local[:,i],
             mean_kl_full_ecdf[:,i], mean_kl_localized_ecdf[:,i], mean_kl_local_ecdf[:,i]) = fut.result()

    plt.plot(mean_kl_full[0])
    plt.show()

    plt.plot(mean_kl_full[1])
    plt.show()

    plt.plot(mean_kl_full[3])
    plt.show()

    plt.plot(mean_kl_localized[2])
    plt.show()

    plt.plot(range(n_points), np.mean(mean_kl_full, axis=1), label="Oracle")
    plt.plot(range(n_points), np.mean(mean_kl_full_ecdf, axis=1), label="ECDF")
    plt.title("Average KL per eval point")
    plt.legend()
    plt.show()

    kl_full_for_plot = np.mean(mean_kl_full, axis=1)
    kl_localized_for_plot = np.mean(mean_kl_localized, axis=1)
    kl_local_for_plot = np.mean(mean_kl_local, axis=1)

    kl_full_for_plot_ecdf = np.mean(mean_kl_full_ecdf, axis=1)
    kl_localized_for_plot_ecdf = np.mean(mean_kl_localized_ecdf, axis=1)
    kl_local_for_plot_ecdf = np.mean(mean_kl_local_ecdf, axis=1)

    # Rejection rate + Size discrepancy ---------------------------------------------------
    left_LogS = np.array([(dm_full[i] < norm.ppf(nominal_size)).mean() for i in range(n_points)])
    left_LogS_ecdf = np.array([(dm_full_ecdf[i] < norm.ppf(nominal_size)).mean() for i in range(n_points)])

    left_CS = np.array([(dm_localized[i] < norm.ppf(nominal_size)).mean() for i in range(n_points)])
    left_CS_ecdf = np.array([(dm_localized_ecdf[i] < norm.ppf(nominal_size)).mean() for i in range(n_points)])

    left_CLS = np.array([(dm_local[i] < norm.ppf(nominal_size)).mean() for i in range(n_points)])
    left_CLS_ecdf = np.array([(dm_local_ecdf[i] < norm.ppf(nominal_size)).mean() for i in range(n_points)])

    def show_power(ax, kl_axis1, kl_axis2, power1, power2, label1, label2, color1, color2):
        ax.plot(kl_axis1, power1, color=color1)
        ax.plot(kl_axis2, power2, color=color2)
        plt.axhline(nominal_size, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("KL distance from true")

    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
    show_power(ax1, kl_full_for_plot, kl_full_for_plot_ecdf, left_LogS, left_LogS_ecdf, "Oracle", "ECDF", "tab:blue", "navy")
    ax1.set_title("Power (Clayton – sJoe KL matched) left-tailed")
    ax1.set_ylabel("Power (rejection rate)")
    fig1.suptitle("Power envelope LogS: Oracle vs ECDF", fontsize=16)
    plt.tight_layout()
    plt.show()

    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
    show_power(ax2, kl_localized_for_plot, kl_localized_for_plot_ecdf, left_CS, left_CS_ecdf,"Oracle", "ECDF", "red", "darkred")
    ax2.set_title("Power (Clayton – sJoe localized KL matched) left-tailed")
    ax2.set_ylabel("Power (rejection rate)")
    fig2.suptitle("Power envelope CS: Oracle vs ECDF", fontsize=16)
    plt.tight_layout()
    plt.show()

    fig3, ax3 = plt.subplots(1, 1, figsize=(6, 4))
    show_power(ax3, kl_local_for_plot, kl_local_for_plot_ecdf, left_CLS, left_CLS_ecdf,"Oracle", "ECDF", "seagreen", "darkgreen")
    ax3.set_title("Power (Clayton – sJoe local KL matched) left-tailed")
    ax3.set_ylabel("Power (rejection rate)")
    fig3.suptitle("Power envelope CLS: Oracle vs ECDF", fontsize=16)
    plt.tight_layout()
    plt.show()

# ───────────────── 5. Multiprocessing guard ───────────────────────────
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
