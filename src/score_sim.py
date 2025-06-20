import numpy as np

from utils.util_funcs import *
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from scipy.optimize import minimize
from score_sim_config import *
from utils.plot_utils import *
from itertools import combinations
from utils.score_helpers import *

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

def simulate_one_rep(n, df, f_rho, g_rho, p_rho,
                     theta_bb1, delta_bb1,
                     theta_bb1_localized, delta_bb1_localized,
                     theta_bb1_local, delta_bb1_local,
                     theta_sGumbel):
    """
        Helper function for simulating one repetition in multi-threading

        Inputs:
            inputs for all functions used in one repetition

        Returns:
            different scores of the candidates and true DGP
        """

    def score_vectors(u, model_id, score_type, **kwargs):
        """
        u          : (n, 2) array of PITs or pseudo-obs
        model_id   : 'student_t' | 'sGumbel' | 'bb1'            (string)
        score_type : 'LogS' | 'CS' | 'CLS'                     (string)
        kwargs     : parameters that the specific scorer needs
        """
        scorer = SCORE_FUNCS[model_id][score_type]
        return scorer(u, **kwargs)  # <-- ALWAYS utils.py!

    # Define DGP1 (indep. student-t)
    samples_p = multivariate_t.rvs(loc=[0, 0], shape=[[1, 0], [0, 1]], df=df, size=n)
    total_oracle_u_p = student_t.cdf(samples_p, df)
    total_ecdf_u_p = ecdf_transform(samples_p)  #DIT MOET MISSCHIEN total_oracle_u_p GEBRUIKEN
    w_p = region_weight_function(n, total_oracle_u_p, q_threshold, df)

    # Define DGP2 (survival Gumbel)
    total_oracle_u_sGumbel = sim_sGumbel_PITs(n, theta_sGumbel)
    samples_sGumbel = np.column_stack([
        student_t.ppf(total_oracle_u_sGumbel[:, 0], df),
        student_t.ppf(total_oracle_u_sGumbel[:, 1], df)
    ])
    total_ecdf_u_sGumbel = ecdf_transform(samples_sGumbel)  #DIT MOET MISSCHIEN total_oracle_u_sGumbel GEBRUIKEN
    w_sGumbel = region_weight_function(n, total_oracle_u_sGumbel, 0.05, df)

    ecdf_u_p = np.empty((P, R, 2))
    oracle_u_p = np.empty((P, R, 2))
    ecdf_u_sGumbel = np.empty((P, R, 2))
    oracle_u_sGumbel = np.empty((P, R, 2))

    for k, t in enumerate(range(R, R+P)):
        ecdf_u_p[k] = ecdf_transform(samples_p[t-R:t])
        oracle_u_p[k] = total_oracle_u_p[t-R:t]
        ecdf_u_sGumbel[k] = ecdf_transform(samples_sGumbel[t-R:t])
        oracle_u_sGumbel[k] = total_oracle_u_sGumbel[t - R:t]


    # Store rolling-window PITs and model parameters
    model_info = {
        "f": {
            "oracle": (oracle_u_p, w_p, {"rho": f_rho, "df": df}),
            "ecdf": (ecdf_u_p, w_p, {"rho": f_rho, "df": df}),
        },
        "g": {
            "oracle": (oracle_u_p, w_p, {"rho": g_rho, "df": df}),
            "ecdf": (ecdf_u_p, w_p, {"rho": g_rho, "df": df}),
        },
        "p": {
            "oracle": (oracle_u_p, w_p, {"rho": p_rho, "df": df}),
            "ecdf": (ecdf_u_p, w_p, {"rho": p_rho, "df": df}),
        },
        "bb1": {
            "oracle": (oracle_u_sGumbel, w_sGumbel, {"theta": theta_bb1, "delta": delta_bb1}),
            "ecdf": (ecdf_u_sGumbel, w_sGumbel, {"theta": theta_bb1, "delta": delta_bb1}),
        },
        "bb1_localized": {
            "oracle": (oracle_u_sGumbel, w_sGumbel, {"theta": theta_bb1_localized, "delta": delta_bb1_localized}),
            "ecdf": (ecdf_u_sGumbel, w_sGumbel, {"theta": theta_bb1_localized, "delta": delta_bb1_localized}),
        },
        "bb1_local": {
            "oracle": (oracle_u_sGumbel, w_sGumbel, {"theta": theta_bb1_local, "delta": delta_bb1_local}),
            "ecdf": (ecdf_u_sGumbel, w_sGumbel, {"theta": theta_bb1_local, "delta": delta_bb1_local}),
        },
        "f_for_KL_matching": {
            "oracle": (oracle_u_sGumbel, w_sGumbel, {"rho": f_rho, "df": df}),
            "ecdf": (ecdf_u_sGumbel, w_sGumbel, {"rho": f_rho, "df": df}),
        },
        "sGumbel": {
            "oracle": (oracle_u_sGumbel, w_sGumbel, {"theta": theta_sGumbel}),
            "ecdf": (ecdf_u_sGumbel, w_sGumbel, {"theta": theta_sGumbel}),
        },
    }

    score_vecs = {score: {model: {} for model in model_info} for score in score_types}
    score_sums = {score: {model: {} for model in model_info} for score in score_types}

    # Evaluate scores for each rolling window
    for model, pits in model_info.items():
        for pit, (u_dat, w_dat, params) in pits.items():
            log_v = np.empty(P)
            cs_v = np.empty(P)
            cls_v = np.empty(P)
            for k in range(P):
                window_u = u_dat[k]
                log_v[k] = score_vectors(window_u, model, "LogS", **params)
                cs_v[k] = score_vectors(window_u, model, "CS", w=w_dat, **params)
                cls_v[k] = score_vectors(window_u, model, "CLS", w=w_dat, **params)
            for name, vec in zip(["LogS", "CS", "CLS"], [log_v, cs_v, cls_v]):
                score_vecs[name][model][pit] = vec
                score_sums[name][model][pit] = float(np.sum(vec))

    model_pairs = list(combinations(copula_models_for_plots, 2))
    diff_vecs = {score: {pit: {} for pit in pit_types} for score in score_types}
    for model_a, model_b in model_pairs:
        for pit in pit_types:
            for score in score_types:
                vec_a = score_vecs[score][model_a][pit]
                vec_b = score_vecs[score][model_b][pit]
                diff_vecs[score][pit][DiffKey(pit, model_a, model_b)] = vec_a - vec_b

    return {
        "sums": score_sums,
        "vecs": score_vecs,
        "diff_vecs": diff_vecs,
    }

if __name__ == '__main__':

    # === Oracle KL matching ===

    # Simulate oracle survival Gumbel PITs & region masks based on those PITs once to reuse
    oracle_samples_list = [sim_sGumbel_PITs(n, theta_sGumbel) for _ in range(reps)]
    oracle_masks_list = [region_weight_function_for_kl_match(u, q_threshold, df=df) for u in oracle_samples_list]

    # Define PDFs
    pdf_sGumbel = lambda u: sGumbel_copula_pdf_from_PITs(u, theta_sGumbel)
    pdf_f = lambda u: student_t_copula_pdf_from_PITs(u, rho=f_rho, df=df)

    # Full KL divergence target calculation
    kl_oracle_sGumbel_g_list = [
        estimate_kl_divergence_copulas(u, pdf_sGumbel, pdf_f)
        for u in oracle_samples_list
    ]
    target_kl_oracle = np.mean(kl_oracle_sGumbel_g_list)

    # Localized KL divergence target calculation
    localized_kl_list = [
        estimate_localized_kl(u, pdf_sGumbel, pdf_f, mask)
        for u, mask in zip(oracle_samples_list, oracle_masks_list)
    ]
    target_localized_kl_oracle = np.mean(localized_kl_list)

    # Local KL divergence target calculation
    localized_kl_list = [
        estimate_local_kl(u, pdf_sGumbel, pdf_f, mask)
        for u, mask in zip(oracle_samples_list, oracle_masks_list)
    ]
    target_local_kl_oracle = np.mean(localized_kl_list)


    # === Optimization ===

    # Optimization objectives
    def bb1_oracle_objective(params):
        theta, delta = params
        if theta <= 0 or delta < 1:
            return np.inf
        pdf_bb1 = lambda u: bb1_copula_pdf_from_PITs(u, theta, delta)
        kl_vals = [estimate_kl_divergence_copulas(u, pdf_sGumbel, pdf_bb1) for u in oracle_samples_list]
        return (np.mean(kl_vals) - target_kl_oracle) ** 2

    def bb1_localized_oracle_objective(params):
        theta, delta = params
        if theta <= 0 or delta < 1:
            return np.inf
        pdf_bb1 = lambda u: bb1_copula_pdf_from_PITs(u, theta, delta)
        kl_vals = [
            estimate_localized_kl(u, pdf_sGumbel, pdf_bb1, mask)
            for u, mask in zip(oracle_samples_list, oracle_masks_list)
        ]
        return (np.mean(kl_vals) - target_localized_kl_oracle) ** 2

    def bb1_local_oracle_objective(params):
        theta, delta = params
        if theta <= 0 or delta < 1:
            return np.inf
        pdf_bb1 = lambda u: bb1_copula_pdf_from_PITs(u, theta, delta)
        kl_vals = [
            estimate_local_kl(u, pdf_sGumbel, pdf_bb1, mask)
            for u, mask in zip(oracle_samples_list, oracle_masks_list)
        ]
        return (np.mean(kl_vals) - target_local_kl_oracle) ** 2

    print("on to minimization")
    # Run optimization
    res_oracle = minimize(
        bb1_oracle_objective,
        x0=[2.0, 2.5],
        bounds=bb1_param_bounds,
        method=kl_match_optim_method,
        options={'disp': True}
    )
    print("on to localized minimization")
    res_localized_oracle = minimize(
        bb1_localized_oracle_objective,
        x0=[2.0, 2.5],
        bounds=bb1_param_bounds,
        method=kl_match_optim_method,
        options={'disp': True}
    )
    print("on to local minimization")
    res_local_oracle = minimize(
        bb1_local_oracle_objective,
        x0=[2.0, 2.5],
        bounds=bb1_param_bounds,
        method=kl_match_optim_method,
        options={'disp': True}
    )

    # === Final reporting of KL matching ===

    theta_bb1_oracle, delta_bb1_oracle = res_oracle.x
    pdf_bb1_opt = lambda u: bb1_copula_pdf_from_PITs(u, theta_bb1_oracle, delta_bb1_oracle)

    theta_bb1_localized_oracle, delta_bb1_localized_oracle = res_localized_oracle.x
    pdf_bb1_opt_localized = lambda u: bb1_copula_pdf_from_PITs(u, theta_bb1_localized_oracle, delta_bb1_localized_oracle)

    theta_bb1_local_oracle, delta_bb1_local_oracle = res_local_oracle.x
    pdf_bb1_opt_local = lambda u: bb1_copula_pdf_from_PITs(u, theta_bb1_local_oracle, delta_bb1_local_oracle)

    kl_final = estimate_kl_divergence_copulas(np.vstack(oracle_samples_list), pdf_sGumbel, pdf_bb1_opt)
    kl_final_localized = estimate_localized_kl(
        np.vstack(oracle_samples_list), pdf_sGumbel, pdf_bb1_opt_localized, np.concatenate(oracle_masks_list)
    )
    kl_final_local = estimate_local_kl(
        np.vstack(oracle_samples_list), pdf_sGumbel, pdf_bb1_opt_local, np.concatenate(oracle_masks_list)
    )

    print(f"Tuned BB1 (oracle PITs): theta = {theta_bb1_oracle:.4f}, delta = {delta_bb1_oracle:.4f}")
    print(f"Target KL(sGumbel||f) oracle: {target_kl_oracle:.6f}")
    print(f"Optimized KL(sGumbel||bb1): {kl_final:.6f}")
    print(
        f"Tuned localized BB1 (oracle PITs): theta = {theta_bb1_localized_oracle:.4f}, delta = {delta_bb1_localized_oracle:.4f}")
    print(f"Target localized KL(sGumbel||f) oracle: {target_localized_kl_oracle:.6f}")
    print(f"Optimized localized KL(sGumbel||bb1): {kl_final_localized:.6f}")
    print(
        f"Tuned local BB1 (oracle PITs): theta = {theta_bb1_local_oracle:.4f}, delta = {delta_bb1_local_oracle:.4f}")
    print(f"Target local KL(sGumbel||f) oracle: {target_local_kl_oracle:.6f}")
    print(f"Optimized local KL(sGumbel||bb1): {kl_final_local:.6f}")

    results = []

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(
            simulate_one_rep, n, df, f_rho, g_rho, p_rho,
            theta_bb1_oracle, delta_bb1_oracle,
            theta_bb1_localized_oracle, delta_bb1_localized_oracle,
            theta_bb1_local_oracle, delta_bb1_local_oracle,
            theta_sGumbel
        ) for _ in range(reps)]

        for future in tqdm(as_completed(futures), total=reps, desc="Running simulations"):
            results.append(future.result())

    print("Sample result keys:")
    for k in results[0].keys():
        print(" ", k)

    # Store extracted vectors in a dictionary: e.g., vecs["LogS"]["f"]["oracle"]
    vecs = {score: {model: {} for model in all_copula_models} for score in score_types}
    for score in score_types:
        for model in all_copula_models:
            for pit in pit_types:
                try:
                    vecs[score][model][pit] = np.array([
                        res["sums"][score][model][pit] for res in results
                    ])
                except KeyError:
                    print(f"Key error for {score}: {model}, {pit}")

    # Get all pairwise model combinations (excluding self-pairs)
    model_pairs = list(combinations(copula_models_for_plots, 2))  # [('f', 'g'), ('f', 'p'), ..., ('bb1', 'f_for_KL_matching')]

    print(model_pairs)

    diffs = {score: {} for score in score_types}
    diff_mats = {score: {} for score in score_types}

    for pit in pit_types:
        for score in score_types:
            for model_a, model_b in model_pairs:
                key = DiffKey(pit, model_a, model_b)
                try:
                    vec_a = vecs[score][model_a][pit]
                    vec_b = vecs[score][model_b][pit]
                except KeyError:
                    continue  # Skip if data missing for a model/pit combination
                diffs[score][key] = vec_a - vec_b
                diff_mats[score][key] = np.vstack([
                    res["diff_vecs"][score][pit][key] for res in results
                ])

    for score, score_dict in diffs.items():
        for key in score_dict:
            print(f"{score}: {key}")

    diff_keys = sorted({k for d in diffs.values() for k in d})
    pair_names = {k: k.label for k in diff_keys}

    # === Size tests ===
    p_values = {
        score: {k: t_test_per_replication(mat) for k, mat in mats.items()} for score, mats in diff_mats.items()
    }
    size_curves = {
        score: {k: perform_size_tests(v) for k, v in pv.items()} for score, pv in p_values.items()
    }

    target_keys = [k for pair in pair_to_keys_size.values() for k in pair]
    pair_labels_subset = {
        key: f"{label} ({key.pit})"
        for label, (oracle_suf, ecdf_suf) in pair_to_keys_size.items()
        for suf in (oracle_suf, ecdf_suf)
    }

    for score in score_types:
        subset = {
            key: size_curves[score][key]
            for key in target_keys
            if key in size_curves[score]
        }
        plot_size_curves(subset, pair_labels_subset, plot_type="discrepancy", title=f"{score} Size Discrepancy")
        plot_size_curves(subset, pair_labels_subset, plot_type="rejection", title=f"{score} Rejection Rates")

    # For plotting divide CS and CLS by std dev
    for score in score_types:
        for key in list(diffs[score].keys()):
            diffs[score][key] = div_by_stdev(str(key), diffs[score][key])

    # Create score dictionaries for plotting
    score_dicts = make_score_dicts(diffs, diff_keys, score_types)

    print("Available keys:", [str(k) for k in score_dicts.keys()])

    # --- Plots for score differences ---
    plot_score_differences(score_dicts, score_types, pair_to_keys)

    plot_aligned_kl_matched_scores(score_dicts, score_score_keys)

    plot_aligned_kl_matched_scores_cdf(score_dicts, score_score_keys)
