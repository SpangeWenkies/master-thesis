from utils.utils import *
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from scipy.optimize import minimize
from score_sim_config import *
from utils.plot_utils import *
from itertools import combinations
from utils.score_helpers import *

def simulate_one_rep(n, df, f_rho, g_rho, p_rho, theta_oracle, delta_oracle, theta_sGumbel):
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

    return {
        "LogS_f_oracle": LogS_student_t_copula(sim_u_p, f_rho, df), #based on DGP1 (indep. student-t)
        "LogS_g_oracle": LogS_student_t_copula(sim_u_p, g_rho, df), #based on DGP1 (indep. student-t)
        "LogS_f_for_KL_matching_oracle": LogS_student_t_copula(sim_u_sGumbel, f_rho, df), #based on DGP2 (sGumbel)
        "LogS_p_oracle": LogS_student_t_copula(sim_u_p, p_rho, df), #based on DGP1 (indep. student-t)
        "LogS_bb1_oracle": LogS_bb1(sim_u_sGumbel, theta_oracle, delta_oracle), #based on DGP2 (sGumbel)
        "LogS_sGumbel_oracle": LogS_sGumbel(sim_u_sGumbel, theta_sGumbel), #based on DGP2 (sGumbel)
        "CS_f_oracle": CS_student_t_copula(sim_u_p, f_rho, df, w_p),
        "CS_g_oracle": CS_student_t_copula(sim_u_p, g_rho, df, w_p),
        "CS_f_for_KL_matching_oracle": CS_student_t_copula(sim_u_sGumbel, f_rho, df, w_sGumbel),
        "CS_p_oracle": CS_student_t_copula(sim_u_p, p_rho, df, w_p),
        "CS_bb1_oracle": CS_bb1(sim_u_sGumbel, theta_oracle, delta_oracle, w_sGumbel),
        "CS_sGumbel_oracle": CS_sGumbel(sim_u_sGumbel, theta_sGumbel, w_sGumbel),
        "CLS_f_oracle": CLS_student_t_copula(sim_u_p, f_rho, df, w_p),
        "CLS_g_oracle": CLS_student_t_copula(sim_u_p, g_rho, df, w_p),
        "CLS_f_for_KL_matching_oracle": CLS_student_t_copula(sim_u_sGumbel, f_rho, df, w_sGumbel),
        "CLS_p_oracle": CLS_student_t_copula(sim_u_p, p_rho, df, w_p),
        "CLS_bb1_oracle": CLS_bb1(sim_u_sGumbel, theta_oracle, delta_oracle, w_sGumbel),
        "CLS_sGumbel_oracle": CLS_sGumbel(sim_u_sGumbel, theta_sGumbel, w_sGumbel),
        "LogS_f_ecdf": LogS_student_t_copula(estim_u_p, f_rho, df),
        "LogS_g_ecdf": LogS_student_t_copula(estim_u_p, g_rho, df),
        "LogS_f_for_KL_matching_ecdf": LogS_student_t_copula(estim_u_sGumbel, f_rho, df),
        "LogS_p_ecdf": LogS_student_t_copula(estim_u_p, p_rho, df),
        "LogS_bb1_ecdf": LogS_bb1(estim_u_sGumbel, theta_oracle, delta_oracle),
        "LogS_sGumbel_ecdf": LogS_sGumbel(estim_u_sGumbel, theta_sGumbel),
        "CS_f_ecdf": CS_student_t_copula(estim_u_p, f_rho, df, w_p),
        "CS_g_ecdf": CS_student_t_copula(estim_u_p, g_rho, df, w_p),
        "CS_f_for_KL_matching_ecdf": CS_student_t_copula(estim_u_sGumbel, f_rho, df, w_sGumbel),
        "CS_p_ecdf": CS_student_t_copula(estim_u_p, p_rho, df, w_p),
        "CS_bb1_ecdf": CS_bb1(estim_u_sGumbel, theta_oracle, delta_oracle, w_sGumbel),
        "CS_sGumbel_ecdf": CS_sGumbel(estim_u_sGumbel, theta_sGumbel, w_sGumbel),
        "CLS_f_ecdf": CLS_student_t_copula(estim_u_p, f_rho, df, w_p),
        "CLS_g_ecdf": CLS_student_t_copula(estim_u_p, g_rho, df, w_p),
        "CLS_f_for_KL_matching_ecdf": CLS_student_t_copula(estim_u_sGumbel, f_rho, df, w_sGumbel),
        "CLS_p_ecdf": CLS_student_t_copula(estim_u_p, p_rho, df, w_p),
        "CLS_bb1_ecdf": CLS_bb1(estim_u_sGumbel, theta_oracle, delta_oracle, w_sGumbel),
        "CLS_sGumbel_ecdf": CLS_sGumbel(estim_u_sGumbel, theta_sGumbel, w_sGumbel),
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


    def bb1_local_oracle_objective(params):
        theta, delta = params
        if theta <= 0 or delta < 1:
            return np.inf
        pdf_bb1 = lambda u: bb1_copula_pdf_from_PITs(u, theta, delta)
        kl_vals = [
            estimate_localized_kl(u, pdf_sGumbel, pdf_bb1, mask)
            for u, mask in zip(oracle_samples_list, oracle_masks_list)
        ]
        return (np.mean(kl_vals) - target_local_kl_oracle) ** 2

    print("on to minimization")
    # Run optimization
    res_oracle = minimize(
        bb1_oracle_objective,
        x0=[2.0, 2.5],
        bounds=bb1_param_bounds,
        method=kl_match_optim_method
    )
    print("on to localized minimization")
    res_local_oracle = minimize(
        bb1_local_oracle_objective,
        x0=[2.0, 2.5],
        bounds=bb1_param_bounds,
        method=kl_match_optim_method
    )

    # === Final reporting of KL matching ===

    theta_bb1_oracle, delta_bb1_oracle = res_oracle.x
    pdf_bb1_opt = lambda u: bb1_copula_pdf_from_PITs(u, theta_bb1_oracle, delta_bb1_oracle)

    theta_bb1_local_oracle, delta_bb1_local_oracle = res_local_oracle.x
    pdf_bb1_opt_local = lambda u: bb1_copula_pdf_from_PITs(u, theta_bb1_local_oracle, delta_bb1_local_oracle)
    print("on to calculating the final KL")
    kl_final = estimate_kl_divergence_copulas(np.vstack(oracle_samples_list), pdf_sGumbel, pdf_bb1_opt)
    print("on to calculate the final local KL")
    kl_final_local = estimate_localized_kl(
        np.vstack(oracle_samples_list), pdf_sGumbel, pdf_bb1_opt_local, np.concatenate(oracle_masks_list)
    )

    print(f"Tuned BB1 (oracle PITs): theta = {theta_bb1_oracle:.4f}, delta = {delta_bb1_oracle:.4f}")
    print(f"Target KL(sGumbel||f) oracle: {target_kl_oracle:.6f}")
    print(f"Optimized KL(sGumbel||bb1): {kl_final:.6f}")
    print(
        f"Tuned localized BB1 (oracle PITs): theta = {theta_bb1_local_oracle:.4f}, delta = {delta_bb1_local_oracle:.4f}")
    print(f"Target localized KL(sGumbel||f) oracle: {target_local_kl_oracle:.6f}")
    print(f"Optimized localized KL(sGumbel||bb1): {kl_final_local:.6f}")

    results = []

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(simulate_one_rep, n, df, f_rho, g_rho, p_rho, theta_bb1_oracle,
                                   delta_bb1_oracle, theta_sGumbel) for _ in range(reps)]

        for future in tqdm(as_completed(futures), total=reps, desc="Running simulations"):
            results.append(future.result())

    # Store extracted vectors in a dictionary: e.g., vecs["LogS"]["f"]["oracle"]
    vecs = {score: {model: {} for model in all_copula_models} for score in score_types}
    for score in score_types:
        for model in all_copula_models:
            for pit in pit_types:
                key = f"{score}_{model}_{pit}"
                if key in results[0]:  # only include if key exists in result
                    vecs[score][model][pit] = np.array([res.get(key, np.nan) for res in results])

    # Get all pairwise model combinations (excluding self-pairs)
    model_pairs = list(combinations(copula_models_for_plots, 2))  # [('f', 'g'), ('f', 'p'), ..., ('bb1', 'f_for_KL_matching')]

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

    tag_suffixes = list(diffs.keys())  # All keys are now "{score}_diffs_{pit}_{model_a}_{model_b}"

    # Extract suffix from each full key
    suffixes = list({extract_suffix_from_key(key, score_types) for key in diffs})

    # Optional: for plotting labels
    pair_names = make_pair_labels(suffixes)

    # For plotting divide CS and CLS by std dev
    diffs = {k: div_by_stdev(k, v) for k, v in diffs.items()}

    # Create score dictionaries for plotting
    score_dicts = make_score_dicts(diffs, suffixes, score_types)


    # PLOTS
    # plot_score_differences(score_dicts, score_types, pair_names)  # All
    plot_score_differences(score_dicts, score_types, pair_names, pair_label=["bb1 - f_for_KL_matching", "f - g", "f - p", "g - p"])

