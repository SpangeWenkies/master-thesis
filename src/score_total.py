# Full-sample evaluation script. Run with `python src/score_total.py`

# Utilities for full-sample KL matching and score plotting
# Run this script with `python -m score_total` or `python src/score_total.py`

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.optimize import minimize
from tqdm import tqdm

from utils.copula_utils import (
    sim_sGumbel_PITs,
    sGumbel_copula_pdf_from_PITs,
    student_t_copula_pdf_from_PITs,
    bb1_copula_pdf_from_PITs,
    ecdf_transform,
    sample_region_mask,
)
from utils.scoring import (
    estimate_kl_divergence_copulas,
    estimate_localized_kl,
    estimate_local_kl,
    LogS_student_t_copula,
    CS_student_t_copula,
    CLS_student_t_copula,
    LogS_sGumbel,
    CS_sGumbel,
    CLS_sGumbel,
    LogS_bb1,
    CS_bb1,
    CLS_bb1,
)
from utils.score_helpers import div_by_stdev, make_score_dicts
from utils.plot_utils import (
    plot_score_differences,
    plot_aligned_kl_matched_scores,
    plot_aligned_kl_matched_scores_cdf,
)
from utils.structure_defs import DiffKey
from score_sim_config import (
    R,
    P,
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
    score_score_keys,
)

SCORE_FUNCS = {
    "student_t": {"LogS": LogS_student_t_copula,
                  "CS":   CS_student_t_copula,
                  "CLS":  CLS_student_t_copula},
    "sGumbel":   {"LogS": LogS_sGumbel,
                  "CS":   CS_sGumbel,
                  "CLS":  CLS_sGumbel},
    "bb1":       {"LogS": LogS_bb1,
                  "CS":   CS_bb1,
                  "CLS":  CLS_bb1}
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

def tune_bb1_params(samples_list, masks_list, pdf_sg, pdf_f):
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

    res_full = minimize(obj, x0=[2.0, 2.5], bounds=bb1_param_bounds,
                        method=kl_match_optim_method)
    res_loc = minimize(obj_loc, x0=[2.0, 2.5], bounds=bb1_param_bounds,
                       method=kl_match_optim_method)
    res_local = minimize(obj_local, x0=[2.0, 2.5], bounds=bb1_param_bounds,
                         method=kl_match_optim_method)

    return res_full.x, res_loc.x, res_local.x

def simulate_one_rep_total(n, df, f_rho, g_rho, p_rho,
                           theta_bb1, delta_bb1,
                           theta_bb1_loc, delta_bb1_loc,
                           theta_bb1_local, delta_bb1_local,
                           theta_sg):
    """Simulate a single repetition without rolling windows."""
    from scipy.stats import multivariate_t, t as student_t

    def score(u, fam, sc, **kw):
        return SCORE_FUNCS[fam][sc](u, **kw)

    samples_p = multivariate_t.rvs(loc=[0,0], shape=[[1,0],[0,1]], df=df, size=n)
    oracle_p = student_t.cdf(samples_p, df)
    ecdf_p = ecdf_transform(samples_p)

    oracle_sg = sim_sGumbel_PITs(n, theta_sg)
    samples_sg = np.column_stack([
        student_t.ppf(oracle_sg[:,0], df),
        student_t.ppf(oracle_sg[:,1], df),
    ])
    ecdf_sg = ecdf_transform(samples_sg)

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
            w = sample_region_mask(u, q_threshold, df)
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
    samples = [sim_sGumbel_PITs(n, theta_sGumbel) for _ in range(reps)]
    masks = [sample_region_mask(u, q_threshold, df=df) for u in samples]

    (theta_bb1, delta_bb1), (theta_loc, delta_loc), (theta_local, delta_local) = tune_bb1_params(samples, masks, pdf_sg, pdf_f)

    results = []
    with ProcessPoolExecutor() as exe:
        futures = [exe.submit(
            simulate_one_rep_total, n, df, f_rho, g_rho, p_rho,
            theta_bb1, delta_bb1,
            theta_loc, delta_loc,
            theta_local, delta_local,
            theta_sGumbel
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
    plot_aligned_kl_matched_scores(score_dicts, score_score_keys)
    plot_aligned_kl_matched_scores_cdf(score_dicts, score_score_keys)

if __name__ == "__main__":
    main()
