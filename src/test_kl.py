from scipy.ndimage import histogram

from utils.copula_utils import (
    sim_sGumbel_PITs,
    sample_region_mask,
    Clayton_copula_pdf_from_PITs,
    sGumbel_copula_pdf_from_PITs,
    sJoe_copula_pdf_from_PITs
)
from utils.optimize_utils import tune_sJoe_params
import numpy as np

from score_sim_config import *

from utils.scoring import LogS

from matplotlib import pyplot as plt

n = 1000
big_n = 1000

mean = np.zeros(n)
mean_localized = np.zeros(n)
mean_local = np.zeros(n)

for i in range(n):
    # --- 1. draw a BIG Monte-Carlo sample for tuning only ------------------
    u_big = sim_sGumbel_PITs(big_n, theta_sGumbel)
    mask_big = sample_region_mask(u_big, q_threshold, df)

    pdf_sGumbel = lambda u: sGumbel_copula_pdf_from_PITs(u, theta_sGumbel)
    pdf_clayton = lambda u: Clayton_copula_pdf_from_PITs(u, theta_Clayton)

    # feed ONLY u_big to tune_sJoe_params  --------------------------
    theta_sJoe, theta_sJoe_loc, theta_sJoe_local = tune_sJoe_params(
        [u_big],                # KL sample (size ≫ evaluation sample)
        [mask_big],             # region mask(s) for local KLs
        pdf_sGumbel,            # truth density
        pdf_clayton,            # Clayton(θ_C = 2) density
        verbose=True,
    )

    # --- 2. when you build the score differences, use FRESH data -----------
    u_eval = sim_sGumbel_PITs(n, theta_sGumbel)   # new RNG

    pdf_sJoe = sJoe_copula_pdf_from_PITs(u_eval, theta_sJoe)
    pdf_Clayton_eval = Clayton_copula_pdf_from_PITs(u_eval, theta_Clayton)

    LogS_sJoe = LogS(pdf_sJoe)
    LogS_Clayton = LogS(pdf_Clayton_eval)

    pdf_sJoe_localized = sJoe_copula_pdf_from_PITs(u_eval, theta_sJoe_loc)
    LogS_sJoe_localized = LogS(pdf_sJoe_localized)
    pdf_sJoe_local = sJoe_copula_pdf_from_PITs(u_eval, theta_sJoe_local)
    LogS_sJoe_local = LogS(pdf_sJoe_local)


    mean[i] = np.mean(LogS_sJoe - LogS_Clayton)
    mean_localized[i] = np.mean(LogS_sJoe_localized - LogS_Clayton)
    mean_local[i] = np.mean(LogS_sJoe_local - LogS_Clayton)

    print(i+1,"/",n)

plt.hist(mean_local)
plt.show()