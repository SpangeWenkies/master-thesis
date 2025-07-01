from src.utils.scoring import outside_prob_from_sample
from utils.copula_utils import (
    sim_sGumbel_PITs,
    sample_region_mask,
    Clayton_copula_pdf_from_PITs,
    sGumbel_copula_pdf_from_PITs,
    sJoe_copula_pdf_from_PITs
)
from utils.optimize_utils import tune_sJoe_params
import numpy as np

from score_sim_config import (
    theta_sGumbel,
    q_threshold,
    df,
    theta_Clayton
)

from utils.scoring import (
    LogS,
    CS,
    CLS,
    outside_prob_from_sample
)

from matplotlib import pyplot as plt

reps = 500
big_n = 1000
eval_n = 1000


mean = np.zeros(reps)
mean_localized = np.zeros(reps)
mean_local = np.zeros(reps)

for i in range(reps):
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

    MC_SIZE_FW = 50_000  # bigger = smoother importance weights

    u_ref_sg = sim_sGumbel_PITs(MC_SIZE_FW, theta_sGumbel)
    pdf_ref_sg = lambda v: sGumbel_copula_pdf_from_PITs(v, theta=theta_sGumbel)

    fw_bar_dict: dict[str, float] = {}  # filled just once per replication

    fw_bar_dict["sGumbel"] = outside_prob_from_sample(u_ref_sg, pdf_model=lambda v: sGumbel_copula_pdf_from_PITs(v,
                                                                                                                 theta=theta_sGumbel),
                                                      pdf_ref=pdf_ref_sg, q_level=q_threshold, df=df)
    fw_bar_dict["Clayton"] = outside_prob_from_sample(u_ref_sg, pdf_model=lambda v: Clayton_copula_pdf_from_PITs(v,
                                                                                                                 theta=theta_Clayton),
                                                      pdf_ref=pdf_ref_sg, q_level=q_threshold, df=df)
    fw_bar_dict["sJoe"] = outside_prob_from_sample(u_ref_sg,
                                                   pdf_model=lambda v: sJoe_copula_pdf_from_PITs(v, theta=theta_sJoe),
                                                   pdf_ref=pdf_ref_sg, q_level=q_threshold, df=df)
    fw_bar_dict["sJoe_localized"] = outside_prob_from_sample(u_ref_sg, pdf_model=lambda v: sJoe_copula_pdf_from_PITs(v,
                                                                                                                     theta=theta_sJoe_loc),
                                                             pdf_ref=pdf_ref_sg, q_level=q_threshold, df=df)
    fw_bar_dict["sJoe_local"] = outside_prob_from_sample(u_ref_sg, pdf_model=lambda v: sJoe_copula_pdf_from_PITs(v,
                                                                                                                 theta=theta_sJoe_local),
                                                         pdf_ref=pdf_ref_sg, q_level=q_threshold, df=df)

    # --- 2. when you build the score differences, use FRESH data -----------
    u_eval = sim_sGumbel_PITs(eval_n, theta_sGumbel)   # new RNG

    pdf_sJoe = sJoe_copula_pdf_from_PITs(u_eval, theta_sJoe)
    pdf_Clayton_eval = Clayton_copula_pdf_from_PITs(u_eval, theta_Clayton)
    LogS_sJoe = LogS(pdf_sJoe)
    LogS_Clayton = LogS(pdf_Clayton_eval)

    pdf_sJoe_localized = sJoe_copula_pdf_from_PITs(u_eval, theta_sJoe_loc)
    CS_sJoe_localized = CS(pdf_sJoe_localized, u_eval, q_threshold, df, Fw_bar=fw_bar_dict["sJoe_localized"]) # Gebruik van u_eval om weights mask te maken is misschien fout
    CS_Clayton = CS(pdf_Clayton_eval, u_eval, q_threshold, df, Fw_bar=fw_bar_dict["Clayton"])

    pdf_sJoe_local = sJoe_copula_pdf_from_PITs(u_eval, theta_sJoe_local)
    CLS_sJoe_local = CLS(pdf_sJoe_local, u_eval, q_threshold, df, Fw_bar=fw_bar_dict["sJoe_local"])
    CLS_Clayton = CLS(pdf_Clayton_eval, u_eval, q_threshold, df, Fw_bar=fw_bar_dict["Clayton"])

    mean[i] = np.mean(LogS_sJoe - LogS_Clayton)
    mean_localized[i] = np.mean(CS_sJoe_localized - CS_Clayton)
    mean_local[i] = np.mean(CLS_sJoe_local - CLS_Clayton)

    print(i+1,"/",reps)

plt.hist(mean, bins=30)
plt.show()
plt.hist(mean_localized, bins=30)
plt.show()
plt.hist(mean_local, bins=30)
plt.show()