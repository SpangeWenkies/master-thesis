# Full-sample evaluation script. Run with `python src/score_total.py`

# Utilities for full-sample KL matching and score plotting
# Run this script with `python -m score_total` or `python src/score_total.py`

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from src.utils.copula_utils import average_threshold, make_fixed_region_mask, sJoe_copula_pdf_from_PITs
from itertools import combinations
import logging
from scipy.stats import t as student_t

from utils.optimize_utils import tune_sJoe_params

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("rpy2").setLevel(logging.WARNING)

from utils.copula_utils import (
    sim_sGumbel_PITs,
    sGumbel_copula_pdf_from_PITs,
    student_t_copula_pdf_from_PITs,
    Clayton_copula_pdf_from_PITs,
    sJoe_copula_pdf_from_PITs,
    ecdf_transform,
)
from utils.scoring import (
    LogS,
    CS,
    CLS,
)
from utils.score_helpers import div_by_stdev, make_score_dicts
from utils.plot_utils import (
    plot_score_differences,
    plot_score_differences_cdf,
    plot_aligned_kl_matched_scores,
    plot_aligned_kl_matched_scores_cdf,
)
from utils.structure_defs import DiffKey
from score_sim_config import (
    n,
    df,
    f_rho,
    g_rho,
    p_rho,
    theta_sGumbel,
    theta_Clayton,
    reps,
    q_threshold,
    pit_types,
    score_types,
    all_copula_models,
    copula_models_for_plots,
    pair_to_keys,
    pair_to_keys_size,  # Is used here for cdf plot of just f-g
    score_score_keys,
)

SCORE_FUNCS = {"LogS": LogS, "CS": CS, "CLS": CLS}

PDF_FUNCS = {
    "student_t": student_t_copula_pdf_from_PITs,
    "sGumbel": sGumbel_copula_pdf_from_PITs,
    "Clayton": Clayton_copula_pdf_from_PITs,
    "sJoe": sJoe_copula_pdf_from_PITs
}

PDF_PARAMS = {
    "student_t": ["rho", "df"],
    "sGumbel": ["theta"],
    "Clayton": ["theta"],
    "sJoe": ["theta"],
}

MODEL_FAMILY = {
    "f": "student_t",
    "g": "student_t",
    "p": "student_t",
    "sJoe": "sJoe",
    "sJoe_localized": "sJoe",
    "sJoe_local": "sJoe",
    "Clayton": "Clayton",
    "sGumbel": "sGumbel",
}

def simulate_one_rep_total(
    n,
    df,
    f_rho,
    g_rho,
    p_rho,
    theta_sg,
    theta_Clayton,
):
    """Simulate a single repetition without rolling windows."""
    from scipy.stats import multivariate_t, t as student_t

    def score(u, fam, sc, **kw):
        score_func = SCORE_FUNCS[sc]
        pdf_func = PDF_FUNCS[fam]
        pdf_kwargs = {k: kw[k] for k in PDF_PARAMS[fam] if k in kw}
        w = kw.get("w")
        fw_bar = kw.get("Fw_bar")
        mF = pdf_func(u, **pdf_kwargs)
        if sc == "LogS":
            return float(np.sum(score_func(mF)))
        elif sc == "CS":
            return float(np.sum(score_func(mF, w, fw_bar)))
        elif sc == "CLS":
            return float(np.sum(score_func(mF, w, fw_bar)))
        else:
            raise ValueError(f"Unknown score type: {sc}")

    # === 1. KL match sJoe copulas on a fresh sample ===
    kl_sample = sim_sGumbel_PITs(2_000_000, theta_sg)
    kl_mask = sample_region_mask(kl_sample, q_threshold, df)
    pdf_sg_big = lambda u: sGumbel_copula_pdf_from_PITs(u, theta_sg)
    pdf_clayton_big = lambda u: Clayton_copula_pdf_from_PITs(u, theta_Clayton)
    (
        theta_sJoe,
        theta_sJoe_loc,
        theta_sJoe_local,
    ) = tune_sJoe_params([kl_sample], [kl_mask], pdf_sg_big, pdf_clayton_big)

    # === 2. Generate evaluation sample ===
    samples_p = multivariate_t.rvs(loc=[0,0], shape=[[1,0],[0,1]], df=df, size=n)
    oracle_p = student_t.cdf(samples_p, df)
    ecdf_p = ecdf_transform(samples_p)

    oracle_sg = sim_sGumbel_PITs(n, theta_sg)
    samples_sg = np.column_stack([
        student_t.ppf(oracle_sg[:,0], df),
        student_t.ppf(oracle_sg[:,1], df),
    ])
    ecdf_sg = ecdf_transform(samples_sg)

    avg_q_sg = average_threshold([oracle_sg], q_threshold)
    avg_q_t = average_threshold([oracle_p], q_threshold)

    fixed_region_mask_sg = make_fixed_region_mask(oracle_sg, avg_q_sg)
    fixed_region_mask_t = make_fixed_region_mask(oracle_p, avg_q_t)

    reference_masks = {
        "oracle": fixed_region_mask_t,
        "ecdf": fixed_region_mask_t,  # use same mask!
        "sGumbel_oracle": fixed_region_mask_sg,
        "sGumbel_ecdf": fixed_region_mask_sg
    }

    model_info = {
        "f": {"oracle": (oracle_p, {"rho": f_rho, "df": df}),
               "ecdf": (ecdf_p, {"rho": f_rho, "df": df})},
        "g": {"oracle": (oracle_p, {"rho": g_rho, "df": df}),
               "ecdf": (ecdf_p, {"rho": g_rho, "df": df})},
        "p": {"oracle": (oracle_p, {"rho": p_rho, "df": df}),
               "ecdf": (ecdf_p, {"rho": p_rho, "df": df})},
        "sJoe": {"oracle": (oracle_sg, {"theta": theta_sJoe}),
                 "ecdf": (ecdf_sg, {"theta": theta_sJoe})},
        "sJoe_localized": {"oracle": (oracle_sg, {"theta": theta_sJoe_loc}),
                           "ecdf": (ecdf_sg, {"theta": theta_sJoe_loc})},
        "sJoe_local": {"oracle": (oracle_sg, {"theta": theta_sJoe_local}),
                       "ecdf": (ecdf_sg, {"theta": theta_sJoe_local})},
        "Clayton": {"oracle": (oracle_sg, {"theta": theta_Clayton}),
                               "ecdf": (ecdf_sg, {"theta": theta_Clayton})},
        "sGumbel": {"oracle": (oracle_sg, {"theta": theta_sg}),
                    "ecdf": (ecdf_sg, {"theta": theta_sg})},
    }

    score_vecs = {s: {m: {} for m in model_info} for s in score_types}
    score_sums = {s: {m: {} for m in model_info} for s in score_types}

    for model, pits in model_info.items():
        fam = MODEL_FAMILY[model]
        for pit, (u, params) in pits.items():
            w = reference_masks[pit]
            log_v = score(u, fam, "LogS", **params)
            cs_v = score(u, fam, "CS", w=w, **params)
            cls_v = score(u, fam, "CLS", w=w, **params)
            score_vecs["LogS"][model][pit] = log_v
            score_vecs["CS"][model][pit] = cs_v
            score_vecs["CLS"][model][pit] = cls_v
            score_sums["LogS"][model][pit] = float(log_v)
            score_sums["CS"][model][pit] = float(cs_v)
            score_sums["CLS"][model][pit] = float(cls_v)

    model_pairs = list(combinations(copula_models_for_plots, 2))
    diff_vecs = {sc: {pt: {} for pt in pit_types} for sc in score_types}
    for ma, mb in model_pairs:
        for pit in pit_types:
            for sc in score_types:
                va = score_vecs[sc][ma][pit]
                vb = score_vecs[sc][mb][pit]
                diff_vecs[sc][pit][DiffKey(pit, ma, mb)] = va - vb

    return {"sums": score_sums, "vecs": score_vecs, "diff_vecs": diff_vecs}

def main():
    results = []
    with ProcessPoolExecutor() as exe:
        futures = [
            exe.submit(
                simulate_one_rep_total,
                n,
                df,
                f_rho,
                g_rho,
                p_rho,
                theta_sGumbel,
                theta_Clayton,
            )
            for _ in range(reps)
        ]
        for fut in tqdm(as_completed(futures), total=reps, desc="Running simulations"):
            results.append(fut.result())

    vecs = {s: {m: {} for m in all_copula_models} for s in score_types}
    for s in score_types:
        for m in all_copula_models:
            for pit in pit_types:
                try:
                    vecs[s][m][pit] = np.array([
                        res["sums"][s][m][pit] for res in results
                    ])
                except KeyError:
                    pass

    model_pairs = list(combinations(copula_models_for_plots, 2))
    diffs = {s: {} for s in score_types}
    for pit in pit_types:
        for sc in score_types:
            for ma, mb in model_pairs:
                key = DiffKey(pit, ma, mb)
                try:
                    va = vecs[sc][ma][pit]
                    vb = vecs[sc][mb][pit]
                except KeyError:
                    continue
                diffs[sc][key] = va - vb

    for sc in score_types:
        for key in list(diffs[sc].keys()):
            diffs[sc][key] = div_by_stdev(str(key), diffs[sc][key])

    diff_keys = sorted({k for d in diffs.values() for k in d})
    score_dicts = make_score_dicts(diffs, diff_keys, score_types)

    plot_score_differences(score_dicts, score_types, pair_to_keys)
    plot_score_differences_cdf(score_dicts, score_types, pair_to_keys_size)
    plot_aligned_kl_matched_scores(score_dicts, score_score_keys)
    plot_aligned_kl_matched_scores_cdf(score_dicts, score_score_keys)

if __name__ == "__main__":
    main()
