
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

mean = np.zeros(100)

for i in range(100):
    # --- 1. draw a BIG Monte-Carlo sample for tuning only ------------------
    u_big = sim_sGumbel_PITs(1000, theta_sGumbel)
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
    u_eval = sim_sGumbel_PITs(1000, theta_sGumbel)   # new RNG

    pdf_sJoe = sJoe_copula_pdf_from_PITs(u_big, theta_sJoe)
    pdf_Clayton_eval = Clayton_copula_pdf_from_PITs(u_big, theta_Clayton)

    LogS_sJoe = LogS(pdf_sJoe)
    LogS_Clayton = LogS(pdf_Clayton_eval)

    mean[i] = np.mean(LogS_sJoe - LogS_Clayton)

    print(i,"/100")

plt.hist(mean)
plt.show()