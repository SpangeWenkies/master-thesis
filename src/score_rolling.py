# Rolling-window simulation script. Run with `python src/score_rolling.py`

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from scipy.stats import multivariate_t, t as student_t, norm
from tqdm import tqdm
from matplotlib import pyplot as plt

from utils.copula_utils import (
    ecdf_transform,
    sample_region_mask,
    sim_student_t_copula_PITs,
    sim_sGumbel_PITs,
    sim_bb1_PITs,
    sGumbel_copula_pdf_from_PITs,
    student_t_copula_pdf_from_PITs,
    average_threshold,
    make_fixed_region_mask,
)
from utils.scoring import (
    LogS_student_t_copula,
    CS_student_t_copula,
    CLS_student_t_copula,
    LogS_sGumbel,
    CS_sGumbel,
    CLS_sGumbel,
    LogS_bb1,
    CS_bb1,
    CLS_bb1,
    outside_prob_from_sample
)
from utils.score_helpers import (
    t_test_per_replication,
    dm_statistic,
)
from utils.plot_utils import (
    plot_size_curves,
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
    pit_types,
    score_types,
    all_copula_models,
    copula_models_for_plots,
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

def simulate_one_rep(n, df, f_rho, g_rho, p_rho, theta_sGumbel):
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

    # Define DGP2 (survival Gumbel)
    total_oracle_u_sGumbel = sim_sGumbel_PITs(n, theta_sGumbel)
    samples_sGumbel = np.column_stack([
        student_t.ppf(total_oracle_u_sGumbel[:, 0], df),
        student_t.ppf(total_oracle_u_sGumbel[:, 1], df)
    ])

    ecdf_u_p = np.empty((P, R, 2))
    oracle_u_p = np.empty((P, R, 2))
    ecdf_u_sGumbel = np.empty((P, R, 2))
    oracle_u_sGumbel = np.empty((P, R, 2))
    mask_p = np.empty((P, R))
    mask_sg = np.empty((P, R))
    next_oracle_u_p = np.empty((P, 2))
    next_ecdf_u_p = np.empty((P, 2))
    next_oracle_u_sGumbel = np.empty((P, 2))
    next_ecdf_u_sGumbel = np.empty((P, 2))
    next_w_p = np.empty(P)
    next_w_sg = np.empty(P)

    for k, t in enumerate(range(R, R+P)):
        ecdf_u_p[k] = ecdf_transform(samples_p[t-R:t])
        oracle_u_p[k] = total_oracle_u_p[t-R:t]
        ecdf_u_sGumbel[k] = ecdf_transform(samples_sGumbel[t-R:t])
        oracle_u_sGumbel[k] = total_oracle_u_sGumbel[t - R:t]
        next_oracle_u_p[k] = total_oracle_u_p[t]
        next_ecdf_u_p[k] = ecdf_transform(np.vstack([samples_p[t - R:t], samples_p[t]]))[-1]
        next_oracle_u_sGumbel[k] = total_oracle_u_sGumbel[t]
        next_ecdf_u_sGumbel[k] = ecdf_transform(np.vstack([samples_sGumbel[t - R:t], samples_sGumbel[t]]))[-1]

    # === KL match BB1 copulas for this repetition ===
    pdf_sGumbel = lambda u: sGumbel_copula_pdf_from_PITs(u, theta_sGumbel)
    pdf_f = lambda u: student_t_copula_pdf_from_PITs(u, rho=f_rho, df=df)
    mask_full = sample_region_mask(total_oracle_u_sGumbel, q_threshold, df=df)
    from score_total import tune_bb1_params
    (
        (theta_bb1, delta_bb1),
        (theta_bb1_localized, delta_bb1_localized),
        (theta_bb1_local, delta_bb1_local),
    ) = tune_bb1_params([total_oracle_u_sGumbel], [mask_full], pdf_sGumbel, pdf_f)

    avg_q_p = np.empty(P)
    avg_q_sg = np.empty(P)
    for k in range(P):
        avg_q_p[k] = average_threshold([oracle_u_p[k]], q_threshold)
        avg_q_sg[k] = average_threshold([oracle_u_sGumbel[k]], q_threshold)
        mask_p[k] = make_fixed_region_mask(oracle_u_p[k], avg_q_p[k])
        mask_sg[k] = make_fixed_region_mask(oracle_u_sGumbel[k], avg_q_sg[k])
        next_w_p[k] = 1.0 if (next_oracle_u_p[k, 0] + next_oracle_u_p[k, 1]) <= avg_q_p[k] else 0.0
        next_w_sg[k] = 1.0 if (next_oracle_u_sGumbel[k, 0] + next_oracle_u_sGumbel[k, 1]) <= avg_q_sg[k] else 0.0

    MC_SIZE_FOR_FW_BAR = 10000
    sample_f = sim_student_t_copula_PITs(MC_SIZE_FOR_FW_BAR, f_rho, df)
    sample_g = sim_student_t_copula_PITs(MC_SIZE_FOR_FW_BAR, g_rho, df)
    sample_p = sim_student_t_copula_PITs(MC_SIZE_FOR_FW_BAR, p_rho, df)
    sample_sg = sim_sGumbel_PITs(MC_SIZE_FOR_FW_BAR, theta_sGumbel)
    sample_bb1 = sim_bb1_PITs(MC_SIZE_FOR_FW_BAR, theta_bb1, delta_bb1)
    sample_bb1_localized = sim_bb1_PITs(MC_SIZE_FOR_FW_BAR, theta_bb1_localized, delta_bb1_localized)
    sample_bb1_local = sim_bb1_PITs(MC_SIZE_FOR_FW_BAR, theta_bb1_local, delta_bb1_local)

    fw_bar_dict = {model: np.empty(P) for model in [
        "f", "g", "p", "bb1", "bb1_localized", "bb1_local", "f_for_KL_matching", "sGumbel"]}

    for k in range(P):
        q_p = avg_q_p[k]
        q_sg = avg_q_sg[k]
        fw_bar_dict["f"][k] = outside_prob_from_sample(sample_f, float(q_p))
        fw_bar_dict["g"][k] = outside_prob_from_sample(sample_g, float(q_p))
        fw_bar_dict["p"][k] = outside_prob_from_sample(sample_p, float(q_p))
        fw_bar_dict["bb1"][k] = outside_prob_from_sample(sample_bb1, float(q_sg))
        fw_bar_dict["bb1_localized"][k] = outside_prob_from_sample(sample_bb1_localized, float(q_sg))
        fw_bar_dict["bb1_local"][k] = outside_prob_from_sample(sample_bb1_local, float(q_sg))
        fw_bar_dict["f_for_KL_matching"][k] = outside_prob_from_sample(sample_f, float(q_sg))
        fw_bar_dict["sGumbel"][k] = outside_prob_from_sample(sample_sg, float(q_sg))

    # Store rolling-window PITs and model parameters
    model_info = {
        "f": {
            "oracle": (oracle_u_p, {"rho": f_rho, "df": df}),
            "ecdf": (ecdf_u_p, {"rho": f_rho, "df": df}),
        },
        "g": {
            "oracle": (oracle_u_p, {"rho": g_rho, "df": df}),
            "ecdf": (ecdf_u_p, {"rho": g_rho, "df": df}),
        },
        "p": {
            "oracle": (oracle_u_p, {"rho": p_rho, "df": df}),
            "ecdf": (ecdf_u_p, {"rho": p_rho, "df": df}),
        },
        "bb1": {
            "oracle": (oracle_u_sGumbel, {"theta": theta_bb1, "delta": delta_bb1}),
            "ecdf": (ecdf_u_sGumbel, {"theta": theta_bb1, "delta": delta_bb1}),
        },
        "bb1_localized": {
            "oracle": (oracle_u_sGumbel, {"theta": theta_bb1_localized, "delta": delta_bb1_localized}),
            "ecdf": (ecdf_u_sGumbel, {"theta": theta_bb1_localized, "delta": delta_bb1_localized}),
        },
        "bb1_local": {
            "oracle": (oracle_u_sGumbel, {"theta": theta_bb1_local, "delta": delta_bb1_local}),
            "ecdf": (ecdf_u_sGumbel, {"theta": theta_bb1_local, "delta": delta_bb1_local}),
        },
        "f_for_KL_matching": {
            "oracle": (oracle_u_sGumbel, {"rho": f_rho, "df": df}),
            "ecdf": (ecdf_u_sGumbel, {"rho": f_rho, "df": df}),
        },
        "sGumbel": {
            "oracle": (oracle_u_sGumbel, {"theta": theta_sGumbel}),
            "ecdf": (ecdf_u_sGumbel, {"theta": theta_sGumbel}),
        },
    }

    reference_masks = {
        "oracle": next_w_p,
        "ecdf": next_w_p,  # use oracle-based weight for ECDF PITs
        "sGumbel_oracle": next_w_sg,
        "sGumbel_ecdf": next_w_sg,
    }

    next_obs = {
        "oracle": next_oracle_u_p,
        "ecdf": next_ecdf_u_p,
        "sGumbel_oracle": next_oracle_u_sGumbel,
        "sGumbel_ecdf": next_ecdf_u_sGumbel,
    }

    score_vecs = {score: {model: {} for model in model_info} for score in score_types}
    score_sums = {score: {model: {} for model in model_info} for score in score_types}

    # Evaluate scores for each rolling window
    for model, pits in model_info.items():
        # Determine which scoring family is associated with this model
        family = MODEL_FAMILY[model]
        for pit, (u_dat, params) in pits.items():
            log_v = np.empty(P)
            cs_v = np.empty(P)
            cls_v = np.empty(P)
            for k in range(P):
                u_next = next_obs[pit][k]
                w_win = reference_masks[pit][k]
                fw_bar = fw_bar_dict[model][k]
                log_v[k] = score_vectors(u_next[np.newaxis, :], family, "LogS", **params)
                cs_v[k] = score_vectors(u_next[np.newaxis, :], family, "CS", w=np.array([w_win]), Fw_bar=fw_bar,
                                        **params)
                cls_v[k] = score_vectors(u_next[np.newaxis, :], family, "CLS", w=np.array([w_win]), Fw_bar=fw_bar,
                                         **params)
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

    dm_stats = {score: {pit: {} for pit in pit_types} for score in score_types}
    for score in score_types:
        for pit in pit_types:
            for key, diff_vec in diff_vecs[score][pit].items():
                dm_stats[score][pit][key] = dm_statistic(diff_vec)

    return {
        "sums": score_sums,
        "vecs": score_vecs,
        "diff_vecs": diff_vecs,
        "dm_stats": dm_stats,
    }

if __name__ == '__main__':


    results = []

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(
            simulate_one_rep, n, df, f_rho, g_rho, p_rho, theta_sGumbel
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
    dm_values = {score: {} for score in score_types}

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
                dm_values[score][key] = np.array([
                    res["dm_stats"][score][pit][key] for res in results
                ])

    for score, score_dict in diffs.items():
        for key in score_dict:
            print(f"{score}: {key}")

    # Compute rejection rates for a grid of alpha levels
    alpha_grid = np.linspace(0.01, 0.2, 20)
    dm_rejection_rates = {score: {} for score in score_types}

    for score, results_dict in dm_values.items():
        for key, stats_vec in results_dict.items():
            right = np.array([(stats_vec > norm.ppf(1 - a)).mean() for a in alpha_grid])
            left = np.array([(stats_vec < norm.ppf(a)).mean() for a in alpha_grid])
            two_sided = np.array([(np.abs(stats_vec) > norm.ppf(1 - a / 2)).mean() for a in alpha_grid])
            dm_rejection_rates[score][key] = {
                "right": right,
                "left": left,
                "two-sided": two_sided,
            }

    key_oracle_fg = DiffKey("oracle", "f", "g")
    key_ecdf_fg = DiffKey("ecdf", "f", "g")

    for score in score_types:
        if key_oracle_fg in dm_rejection_rates[score] and key_ecdf_fg in dm_rejection_rates[score]:
            rates_oracle = dm_rejection_rates[score][key_oracle_fg]["two-sided"]
            rates_ecdf = dm_rejection_rates[score][key_ecdf_fg]["two-sided"]
            plt.figure()
            plt.plot(alpha_grid, rates_oracle, label="oracle")
            plt.plot(alpha_grid, rates_ecdf, label="ecdf")
            plt.plot(alpha_grid, alpha_grid, "--", color="gray", label="alpha = rejection rate")
            plt.xlabel("alpha")
            plt.ylabel("rejection rate")
            plt.title(f"DM test size curves ({score}, f - g)")
            plt.legend()
            plt.grid(True)
            plt.show()


