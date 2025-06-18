from utils.utils import *
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from scipy.optimize import minimize
from score_sim_config import *
from utils.plot_utils import *
from itertools import combinations
from utils.score_helpers import *

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
    # Define DGP1 (indep. student-t)
    samples_p = multivariate_t.rvs(loc=[0, 0], shape=[[1, 0], [0, 1]], df=df, size=n)
    estim_u_p = ecdf_transform(samples_p)
    sim_u_p = student_t.cdf(samples_p, df)
    w_p = region_weight_function(n, sim_u_p, q_threshold, df)

    # Define DGP2 (survival Gumbel)
    sim_u_sGumbel = sim_sGumbel_PITs(n, theta_sGumbel)
    samples_sGumbel = np.column_stack([
        student_t.ppf(sim_u_sGumbel[:, 0], df),
        student_t.ppf(sim_u_sGumbel[:, 1], df)
    ])
    estim_u_sGumbel = ecdf_transform(samples_sGumbel)
    w_sGumbel = region_weight_function(n, sim_u_sGumbel, 0.05, df)

    def score_vectors(u, w, pdf_func):
        pdf = pdf_func(u)
        pdf[pdf == 0] = 1e-100
        log_pdf = np.log(pdf)
        if w.ndim == 1:
            Fw_bar = np.sum(pdf * w) / np.sum(w)
            Fw_bar = max(Fw_bar, 1e-100)
            CS_vec = w * log_pdf + (1 - w) * np.log(Fw_bar)
            F_total = np.sum(pdf)
            F_outside = np.sum(pdf * (1 - w))
            F_outside = min(F_outside, F_total - 1e-100)
            log_1_minus_Fw = np.log(F_outside / F_total + 1e-100)
            CLS_vec = w * (log_pdf - log_1_minus_Fw)
        else:
            row_w = w.sum(axis=1)
            row_not_w = w.shape[1] - row_w
            Fw_bar = np.sum(pdf * w) / np.sum(w)
            Fw_bar = max(Fw_bar, 1e-100)
            CS_vec = row_w * log_pdf + row_not_w * np.log(Fw_bar)
            F_total = np.sum(pdf)
            F_outside = np.sum(pdf * (1 - w))
            F_outside = min(F_outside, F_total - 1e-100)
            log_1_minus_Fw = np.log(F_outside / F_total + 1e-100)
            CLS_vec = row_w * (log_pdf - log_1_minus_Fw)
        return log_pdf, CS_vec, CLS_vec

    model_info = {
        "f": {
            "oracle": (sim_u_p, w_p, lambda u: student_t_copula_pdf_from_PITs(u, f_rho, df)),
            "ecdf": (estim_u_p, w_p, lambda u: student_t_copula_pdf_from_PITs(u, f_rho, df)),
        },
        "g": {
            "oracle": (sim_u_p, w_p, lambda u: student_t_copula_pdf_from_PITs(u, g_rho, df)),
            "ecdf": (estim_u_p, w_p, lambda u: student_t_copula_pdf_from_PITs(u, g_rho, df)),
        },
        "p": {
            "oracle": (sim_u_p, w_p, lambda u: student_t_copula_pdf_from_PITs(u, p_rho, df)),
            "ecdf": (estim_u_p, w_p, lambda u: student_t_copula_pdf_from_PITs(u, p_rho, df)),
        },
        "bb1": {
            "oracle": (sim_u_sGumbel, w_sGumbel, lambda u: bb1_copula_pdf_from_PITs(u, theta_bb1, delta_bb1)),
            "ecdf": (estim_u_sGumbel, w_sGumbel, lambda u: bb1_copula_pdf_from_PITs(u, theta_bb1, delta_bb1)),
        },
        "bb1_localized": {
            "oracle": (
            sim_u_sGumbel, w_sGumbel, lambda u: bb1_copula_pdf_from_PITs(u, theta_bb1_localized, delta_bb1_localized)),
            "ecdf": (estim_u_sGumbel, w_sGumbel,
                     lambda u: bb1_copula_pdf_from_PITs(u, theta_bb1_localized, delta_bb1_localized)),
        },
        "bb1_local": {
            "oracle": (
            sim_u_sGumbel, w_sGumbel, lambda u: bb1_copula_pdf_from_PITs(u, theta_bb1_local, delta_bb1_local)),
            "ecdf": (
            estim_u_sGumbel, w_sGumbel, lambda u: bb1_copula_pdf_from_PITs(u, theta_bb1_local, delta_bb1_local)),
        },
        "f_for_KL_matching": {
            "oracle": (sim_u_sGumbel, w_sGumbel, lambda u: student_t_copula_pdf_from_PITs(u, f_rho, df)),
            "ecdf": (estim_u_sGumbel, w_sGumbel, lambda u: student_t_copula_pdf_from_PITs(u, f_rho, df)),
        },
        "sGumbel": {
            "oracle": (sim_u_sGumbel, w_sGumbel, lambda u: sGumbel_copula_pdf_from_PITs(u, theta_sGumbel)),
            "ecdf": (estim_u_sGumbel, w_sGumbel, lambda u: sGumbel_copula_pdf_from_PITs(u, theta_sGumbel)),
        },
    }

    score_vecs = {score: {model: {} for model in model_info} for score in score_types}

    for model, pits in model_info.items():
        for pit, (u_dat, w_dat, pdf_func) in pits.items():
            log_v, cs_v, cls_v = score_vectors(u_dat, w_dat, pdf_func)
            score_vecs["LogS"][model][pit] = log_v
            score_vecs["CS"][model][pit] = cs_v
            score_vecs["CLS"][model][pit] = cls_v

    results = {}

    for score in score_types:
        for model in model_info:
            for pit in pit_types:
                vec = score_vecs[score][model][pit]
                results[f"{score}_{model}_{pit}"] = np.sum(vec)
                results[f"{score}_vec_{model}_{pit}"] = vec

    # Pairwise difference vectors
    model_pairs = list(combinations(copula_models_for_plots, 2))
    for model_a, model_b in model_pairs:
        for pit in pit_types:
            for score in score_types:
                vec_a = score_vecs[score][model_a][pit]
                vec_b = score_vecs[score][model_b][pit]
                results[f"{score}_diff_vec_{pit}_{model_a}_{model_b}"] = vec_a - vec_b

    return results

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
                key = f"{score}_{model}_{pit}"
                if key in results[0]:  # only include if key exists in result
                    print(f"Populating vecs[{score}][{model}][{pit}]")
                    vecs[score][model][pit] = np.array([res.get(key, np.nan) for res in results])
                else:
                    print(f"Missing key: {key}")

    # Get all pairwise model combinations (excluding self-pairs)
    model_pairs = list(combinations(copula_models_for_plots, 2))  # [('f', 'g'), ('f', 'p'), ..., ('bb1', 'f_for_KL_matching')]

    print(model_pairs)

    diffs = {}

    for pit in pit_types:
        for score in score_types:
            for model_a, model_b in model_pairs:
                try:
                    vec_a = vecs[score][model_a][pit]
                    vec_b = vecs[score][model_b][pit]
                except KeyError:
                    continue  # Skip if data missing for a model/pit combination

                suffix = f"{pit}_{model_a}_{model_b}"  # e.g., oracle_f_g
                diffs[f"{score}_diffs_{suffix}"] = vec_a - vec_b

    for key in diffs:
        print(f"{key}")

    tag_suffixes = list(diffs.keys())  # All keys are now "{score}_diffs_{pit}_{model_a}_{model_b}"

    # Extract suffix from each full key
    suffixes = list({extract_suffix_from_key(key, score_types) for key in diffs})

    # Optional: for plotting labels
    pair_names = make_pair_labels(suffixes)

    # For plotting divide CS and CLS by std dev
    diffs = {k: div_by_stdev(k, v) for k, v in diffs.items()}

    # Create score dictionaries for plotting
    score_dicts = make_score_dicts(diffs, suffixes, score_types)

    print("Available suffixes:", list(score_dicts.keys()))

    validate_plot_data(score_dicts, pair_names, score_types, pair_label=["bb1 - f_for_KL_matching",
                                                                         "bb1_localized - f_for_KL_matching",
                                                                         "bb1_local - f_for_KL_matching",
                                                                         "f - g", "f - p", "g - p"])

    # --- Plots for score differences ---
    plot_score_differences(score_dicts, score_types, pair_to_suffixes)

    plot_aligned_kl_matched_scores(score_dicts, score_score_suffixes)

    plot_aligned_kl_matched_scores_cdf(score_dicts, score_score_suffixes)

    # --- Size testing for equal predictive accuracy ---
    pair_to_suffixes_size = {"f - g": ("oracle_f_g", "ecdf_f_g")}
    size_results = perform_size_tests(score_dicts, score_types, pair_to_suffixes_size)
    plot_size_test_results(size_results, score_types)