# Full-sample evaluation script. Run with `python src/score_total.py`

# Utilities for full-sample KL matching and score plotting
# Run this script with `python -m score_total` or `python src/score_total.py`

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from scipy.optimize import minimize
from tqdm import tqdm

from src.utils.copula_utils import average_threshold, make_fixed_region_mask
from itertools import combinations
import logging
from scipy.stats import t as student_t

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("rpy2").setLevel(logging.WARNING)

from utils.copula_utils import (
    sim_sGumbel_PITs,
    sGumbel_copula_pdf_from_PITs,
    student_t_copula_pdf_from_PITs,
    bb1_copula_pdf_from_PITs,
    ecdf_transform,
)
from utils.scoring import (
    estimate_kl_divergence_copulas,
    estimate_localized_kl,
    estimate_local_kl,
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
    reps,
    q_threshold,
    kl_match_optim_method,
    bb1_param_bounds,
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
    "bb1": bb1_copula_pdf_from_PITs,
}

PDF_PARAMS = {
    "student_t": ["rho", "df"],
    "sGumbel": ["theta"],
    "bb1": ["theta", "delta"],
}

MODEL_FAMILY = {
    "f": "student_t",
    "g": "student_t",
    "p": "student_t",
    "bb1": "bb1",
    "bb1_localized": "bb1",
    "bb1_local": "bb1",
    "f_for_KL_matching": "student_t",
    "sGumbel": "sGumbel",
}

def tune_bb1_params(samples_list, masks_list, pdf_sg, pdf_f, verbose=False):
    """KL-match BB1 parameters to the survival Gumbel."""

    target_kl = np.mean([
        estimate_kl_divergence_copulas(u, pdf_sg, pdf_f)
        for u in samples_list
    ])
    target_loc = np.mean([
        estimate_localized_kl(u, pdf_sg, pdf_f, m)
        for u, m in zip(samples_list, masks_list)
    ])
    target_local = np.mean([
        estimate_local_kl(u, pdf_sg, pdf_f, m)
        for u, m in zip(samples_list, masks_list)
    ])

    def obj(params):
        theta, delta = params
        if theta <= 0 or delta < 1:
            return np.inf
        pdf = lambda u: bb1_copula_pdf_from_PITs(u, theta, delta)
        kl_vals = [estimate_kl_divergence_copulas(u, pdf_sg, pdf) for u in samples_list]
        return (np.mean(kl_vals) - target_kl) ** 2

    def obj_loc(params):
        theta, delta = params
        if theta <= 0 or delta < 1:
            return np.inf
        pdf = lambda u: bb1_copula_pdf_from_PITs(u, theta, delta)
        kl_vals = [estimate_localized_kl(u, pdf_sg, pdf, m) for u, m in zip(samples_list, masks_list)]
        return (np.mean(kl_vals) - target_loc) ** 2

    def obj_local(params):
        theta, delta = params
        if theta <= 0 or delta < 1:
            return np.inf
        pdf = lambda u: bb1_copula_pdf_from_PITs(u, theta, delta)
        kl_vals = [estimate_local_kl(u, pdf_sg, pdf, m) for u, m in zip(samples_list, masks_list)]
        return (np.mean(kl_vals) - target_local) ** 2

    res_full = minimize(
        obj,
        x0=[2.0, 2.5],
        bounds=bb1_param_bounds,
        method=kl_match_optim_method,
    )
    res_loc = minimize(
        obj_loc,
        x0=[2.0, 2.5],
        bounds=bb1_param_bounds,
        method=kl_match_optim_method,
    )
    res_local = minimize(
        obj_local,
        x0=[2.0, 2.5],
        bounds=bb1_param_bounds,
        method=kl_match_optim_method,
    )

    pdf_full = lambda u: bb1_copula_pdf_from_PITs(u, res_full.x[0], res_full.x[1])
    pdf_loc = lambda u: bb1_copula_pdf_from_PITs(u, res_loc.x[0], res_loc.x[1])
    pdf_local = lambda u: bb1_copula_pdf_from_PITs(u, res_local.x[0], res_local.x[1])

    optim_kl = np.mean([estimate_kl_divergence_copulas(u, pdf_sg, pdf_full) for u in samples_list])
    optim_kl_loc = np.mean([estimate_localized_kl(u, pdf_sg, pdf_loc, m) for u, m in zip(samples_list, masks_list)])
    optim_kl_local = np.mean([estimate_local_kl(u, pdf_sg, pdf_local, m) for u, m in zip(samples_list, masks_list)])


    if verbose:
        logger.info(f"Tuned BB1 (full): theta = {res_full.x[0]:.4f}, delta = {res_full.x[1]:.4f}")
        logger.info(f"Target KL(sGumbel||f) full: {target_kl:.6f}")
        logger.info(f"Optimized full KL(sGumbel||bb1): {optim_kl:.6f}")
        logger.info(f"Tuned BB1 (localized): theta = {res_loc.x[0]:.4f}, delta = {res_loc.x[1]:.4f}")
        logger.info(f"Target KL(sGumbel||f) localized: {target_loc:.6f}")
        logger.info(f"Optimized localized KL(sGumbel||bb1): {optim_kl_loc:.6f}")
        logger.info(f"Tuned BB1 (local): theta = {res_local.x[0]:.4f}, delta = {res_local.x[1]:.4f}")
        logger.info(f"Target KL(sGumbel||f) local: {target_local:.6f}")
        logger.info(f"Optimized local KL(sGumbel||bb1): {optim_kl_local:.6f}")

    return res_full.x, res_loc.x, res_local.x

def simulate_one_rep_total(n, df, f_rho, g_rho, p_rho,
                           theta_bb1, delta_bb1,
                           theta_bb1_loc, delta_bb1_loc,
                           theta_bb1_local, delta_bb1_local,
                           theta_sg,
                           fixed_region_mask_sg, fixed_region_mask_t):
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

    samples_p = multivariate_t.rvs(loc=[0,0], shape=[[1,0],[0,1]], df=df, size=n)
    oracle_p = student_t.cdf(samples_p, df)
    ecdf_p = ecdf_transform(samples_p)

    oracle_sg = sim_sGumbel_PITs(n, theta_sg)
    samples_sg = np.column_stack([
        student_t.ppf(oracle_sg[:,0], df),
        student_t.ppf(oracle_sg[:,1], df),
    ])
    ecdf_sg = ecdf_transform(samples_sg)

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
        "bb1": {"oracle": (oracle_sg, {"theta": theta_bb1, "delta": delta_bb1}),
                 "ecdf": (ecdf_sg, {"theta": theta_bb1, "delta": delta_bb1})},
        "bb1_localized": {"oracle": (oracle_sg, {"theta": theta_bb1_loc, "delta": delta_bb1_loc}),
                           "ecdf": (ecdf_sg, {"theta": theta_bb1_loc, "delta": delta_bb1_loc})},
        "bb1_local": {"oracle": (oracle_sg, {"theta": theta_bb1_local, "delta": delta_bb1_local}),
                       "ecdf": (ecdf_sg, {"theta": theta_bb1_local, "delta": delta_bb1_local})},
        "f_for_KL_matching": {"oracle": (oracle_sg, {"rho": f_rho, "df": df}),
                               "ecdf": (ecdf_sg, {"rho": f_rho, "df": df})},
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
    pdf_sg = lambda u: sGumbel_copula_pdf_from_PITs(u, theta_sGumbel)
    pdf_f = lambda u: student_t_copula_pdf_from_PITs(u, rho=f_rho, df=df)

    samples_sg = [sim_sGumbel_PITs(n, theta_sGumbel) for _ in range(reps)]
    samples_t = [student_t.cdf(
        np.random.multivariate_normal([0, 0], [[1, f_rho], [f_rho, 1]], size=n), df=df)
        for _ in range(reps)]

    avg_q_sg = average_threshold(samples_sg, q_threshold)
    avg_q_t = average_threshold(samples_t, q_threshold)

    fixed_region_mask_sg = make_fixed_region_mask(samples_sg[0], avg_q_sg)
    fixed_region_mask_t = make_fixed_region_mask(samples_t[0], avg_q_t)


    (theta_bb1, delta_bb1), (theta_loc, delta_loc), (theta_local, delta_local) = tune_bb1_params(samples_sg, [fixed_region_mask_sg] * reps, pdf_sg, pdf_f, verbose=True)

    results = []
    with ProcessPoolExecutor() as exe:
        futures = [exe.submit(
            simulate_one_rep_total, n, df, f_rho, g_rho, p_rho,
            theta_bb1, delta_bb1,
            theta_loc, delta_loc,
            theta_local, delta_local,
            theta_sGumbel,
            fixed_region_mask_t, fixed_region_mask_sg,
        ) for _ in range(reps)]
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
