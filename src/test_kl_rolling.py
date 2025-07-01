from utils.copula_utils import (
    sim_sGumbel_PITs,
    sJoe_copula_pdf_from_PITs,
    Clayton_copula_pdf_from_PITs,
    sGumbel_copula_pdf_from_PITs,
    sample_region_mask,
)

from utils.scoring import outside_prob_from_sample

import numpy as np

from score_sim_config import (theta_sGumbel, theta_Clayton, q_threshold, df)

from utils.optimize_utils import tune_sJoe_params

pdf_sGumbel = lambda u: sGumbel_copula_pdf_from_PITs(u, theta_sGumbel)
pdf_clayton = lambda u: Clayton_copula_pdf_from_PITs(u, theta_Clayton)

U_big = sim_sGumbel_PITs(300_000, theta_sGumbel)
mask_big = sample_region_mask(U_big, q_threshold, df)

# feed ONLY u_big to tune_sJoe_params  --------------------------
theta_sJoe, theta_sJoe_localized, theta_sJoe_local = tune_sJoe_params(
    [U_big],                # KL sample (size ≫ evaluation sample)
    [mask_big],             # region mask(s) for local KLs
    pdf_sGumbel,            # truth density
    pdf_clayton,            # Clayton(θ_C = 2) density
    verbose=True,
)


log_sJoe     = np.log(sJoe_copula_pdf_from_PITs(U_big, theta=theta_sJoe))
log_Clayton  = np.log(Clayton_copula_pdf_from_PITs(U_big, theta=theta_Clayton))
print("E[log f_sJoe] - E[log f_Clayton] =", (log_sJoe-log_Clayton).mean())

inside  = (mask_big * (log_sJoe - log_Clayton)).mean()
outside = ((1-mask_big) * (log_sJoe - log_Clayton)).mean()
print("inside =",inside, "outside =",outside)

fw_bar_dict: dict[str, float] = {}  # filled just once per replication

fw_bar_dict["sGumbel"] = outside_prob_from_sample(U_big, pdf_model = lambda v: sGumbel_copula_pdf_from_PITs(v, theta=theta_sGumbel),
                                                  pdf_ref = pdf_sGumbel, q_level = q_threshold, df = df)
fw_bar_dict["Clayton"] = outside_prob_from_sample(U_big, pdf_model = lambda v: Clayton_copula_pdf_from_PITs(v, theta=theta_Clayton),
                                                  pdf_ref = pdf_sGumbel, q_level = q_threshold, df = df)
fw_bar_dict["sJoe"] = outside_prob_from_sample(U_big, pdf_model = lambda v: sJoe_copula_pdf_from_PITs(v, theta=theta_sJoe),
                                               pdf_ref = pdf_sGumbel, q_level = q_threshold, df = df)
fw_bar_dict["sJoe_localized"] = outside_prob_from_sample(U_big, pdf_model = lambda v: sJoe_copula_pdf_from_PITs(v, theta=theta_sJoe_localized),
                                                         pdf_ref = pdf_sGumbel, q_level = q_threshold, df = df)
fw_bar_dict["sJoe_local"] = outside_prob_from_sample(U_big, pdf_model = lambda v: sJoe_copula_pdf_from_PITs(v, theta=theta_sJoe_local),
                                                     pdf_ref = pdf_sGumbel, q_level = q_threshold, df = df)
print("Fw_bar  sJoe     =", fw_bar_dict["sJoe"])
print("Fw_bar  Clayton =", fw_bar_dict["Clayton"])

p_out_sJoe    = (1-mask_big).mean()                           # true tail prob under P
p_out_Clayton = ((1-mask_big) * np.exp(log_Clayton-log_sJoe)).mean()
print("true  tail P_sJoe    =", p_out_sJoe)
print("true  tail P_Clayton =", p_out_Clayton)

def CS_pop(log_f, Fw_bar, w):   # vectorised
    return w*log_f + (1-w)*np.log(Fw_bar)
def CLS_pop(log_f, Fw_bar, w):
    return w*(log_f - np.log(1-Fw_bar))

log_sJoe_localized = np.log(sJoe_copula_pdf_from_PITs(U_big, theta=theta_sJoe_localized))
log_sJoe_local = np.log(sJoe_copula_pdf_from_PITs(U_big, theta=theta_sJoe_local))

cs_diff  = CS_pop(log_sJoe_localized,    fw_bar_dict["sJoe"],    mask_big) \
         - CS_pop(log_Clayton, fw_bar_dict["Clayton"], mask_big)
cls_diff = CLS_pop(log_sJoe_local,    fw_bar_dict["sJoe"],    mask_big) \
         - CLS_pop(log_Clayton, fw_bar_dict["Clayton"], mask_big)

print("E[CS diff ] =", cs_diff.mean())
print("E[CLS diff] =", cls_diff.mean())
