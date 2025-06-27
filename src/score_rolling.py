# Rolling-window simulation script. Run with `python src/score_rolling.py`

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from scipy.stats import multivariate_t, t as student_t, norm
from tqdm import tqdm
from matplotlib import pyplot as plt
from utils.optimize_utils import tune_sJoe_params

from utils.copula_utils import (
    ecdf_transform,
    sample_region_mask,
    sim_student_t_copula_PITs,
    sim_sGumbel_PITs,
    sim_sJoe_PITs,
    sim_Clayton_PITs,
    sGumbel_copula_pdf_from_PITs,
    student_t_copula_pdf_from_PITs,
    Clayton_copula_pdf_from_PITs,
    sJoe_copula_pdf_from_PITs,
    average_threshold,
    make_fixed_region_mask,
)
from utils.scoring import (
    LogS,
    CS,
    CLS,
    outside_prob_from_sample,
)
from utils.score_helpers import (
    div_by_stdev,
    make_score_dicts,
    dm_statistic,
)
from utils.plot_utils import (
    plot_dm_size_discrepancy,
    plot_aligned_kl_matched_scores,
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
    theta_Clayton,
    reps,
    q_threshold,
    pit_types,
    score_types,
    all_copula_models,
    copula_models_for_plots,
    score_score_keys,
    tune_size
)

SCORE_FUNCS = {"LogS": LogS, "CS": CS, "CLS": CLS}

PDF_FUNCS = {
    "student_t": student_t_copula_pdf_from_PITs,
    "sGumbel": sGumbel_copula_pdf_from_PITs,
    "Clayton": Clayton_copula_pdf_from_PITs,
    "sJoe": sJoe_copula_pdf_from_PITs,
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

def simulate_one_rep(n, df, f_rho, g_rho, p_rho, theta_sGumbel):
    """
        Helper function for simulating one repetition in multi-threading

        Inputs:
            inputs for all functions used in one repetition

        Returns:
            different scores of the candidates and true DGP
        """

    def score_vectors(u, model_id, score_type, **kwargs):
        """Return score value for ``u`` using the generic scoring functions."""

        # Select scoring and density functions
        score_func = SCORE_FUNCS[score_type]
        pdf_func = PDF_FUNCS[model_id]

        # Separate parameters for the PDF and the scoring rule
        pdf_kwargs = {k: kwargs[k] for k in PDF_PARAMS[model_id] if k in kwargs}
        w = kwargs.get("w")
        fw_bar = kwargs.get("Fw_bar")

        mF = pdf_func(u, **pdf_kwargs)

        if score_type == "LogS":
            return float(np.sum(score_func(mF)))
        elif score_type == "CS":
            return float(np.sum(score_func(mF, w, fw_bar)))
        elif score_type == "CLS":
            return float(np.sum(score_func(mF, w, fw_bar)))
        else:
            raise ValueError(f"Unknown score_type: {score_type}")

    # === 1. KL match sJoe copulas on a fresh sample ===
    kl_sample = sim_sGumbel_PITs(tune_size, theta_sGumbel)
    kl_mask = sample_region_mask(kl_sample, q_threshold, df)
    pdf_sg_big = lambda u: sGumbel_copula_pdf_from_PITs(u, theta_sGumbel)
    pdf_clayton_big = lambda u: Clayton_copula_pdf_from_PITs(u, theta_Clayton)
    (
        theta_sJoe,
        theta_sJoe_localized,
        theta_sJoe_local,
    ) = tune_sJoe_params([kl_sample], [kl_mask], pdf_sg_big, pdf_clayton_big)

    # === 2. Generate evaluation sample ===
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
    next_oracle_u_p = np.empty((P, 2))
    next_ecdf_u_p = np.empty((P, 2))
    next_oracle_u_sGumbel = np.empty((P, 2))
    next_ecdf_u_sGumbel = np.empty((P, 2))
    next_w_p = np.empty(P)
    next_w_sg = np.empty(P)

    # --- fixed thresholds for the indicator weights -------------------------
    avg_q_p = average_threshold([total_oracle_u_p], q_threshold)
    avg_q_sg = average_threshold([total_oracle_u_sGumbel], q_threshold)

    for k, t in enumerate(range(R, R+P)):
        ecdf_u_p[k] = ecdf_transform(samples_p[t-R:t])
        oracle_u_p[k] = total_oracle_u_p[t-R:t]
        ecdf_u_sGumbel[k] = ecdf_transform(samples_sGumbel[t-R:t])
        oracle_u_sGumbel[k] = total_oracle_u_sGumbel[t - R:t]
        next_oracle_u_p[k] = total_oracle_u_p[t]
        next_ecdf_u_p[k] = ecdf_transform(np.vstack([samples_p[t - R:t], samples_p[t]]))[-1]
        next_oracle_u_sGumbel[k] = total_oracle_u_sGumbel[t]
        next_ecdf_u_sGumbel[k] = ecdf_transform(np.vstack([samples_sGumbel[t - R:t], samples_sGumbel[t]]))[-1]

    for k in range(P):
        # Indicator weights based on the sum of PITs compared to the fixed
        # threshold derived from the training sample.
        next_w_p[k] = (
            1.0
            if (next_oracle_u_p[k, 0] + next_oracle_u_p[k, 1]) <= avg_q_p
            else 0.0
        )
        next_w_sg[k] = (
            1.0
            if (next_oracle_u_sGumbel[k, 0] + next_oracle_u_sGumbel[k, 1]) <= avg_q_sg
            else 0.0
        )


    MC_SIZE_FOR_FW_BAR = 10000
    sample_f = sim_student_t_copula_PITs(MC_SIZE_FOR_FW_BAR, f_rho, df)
    sample_g = sim_student_t_copula_PITs(MC_SIZE_FOR_FW_BAR, g_rho, df)
    sample_p = sim_student_t_copula_PITs(MC_SIZE_FOR_FW_BAR, p_rho, df)
    sample_sg = sim_sGumbel_PITs(MC_SIZE_FOR_FW_BAR, theta_sGumbel)
    sample_sJoe = sim_sJoe_PITs(MC_SIZE_FOR_FW_BAR, theta_sJoe)
    sample_sJoe_localized = sim_sJoe_PITs(MC_SIZE_FOR_FW_BAR, theta_sJoe_localized)
    sample_sJoe_local = sim_sJoe_PITs(MC_SIZE_FOR_FW_BAR, theta_sJoe_local)
    sample_Clayton = sim_Clayton_PITs(MC_SIZE_FOR_FW_BAR, theta_Clayton)

    fw_bar_dict = {model: np.empty(P) for model in [
        "f", "g", "p", "sJoe", "sJoe_localized", "sJoe_local", "Clayton", "sGumbel"]}

    for k in range(P):
        # Right-tail probabilities corresponding to the same thresholds used for
        # the indicator weights above.
        fw_bar_dict["f"][k] = outside_prob_from_sample(sample_f, avg_q_p)
        fw_bar_dict["g"][k] = outside_prob_from_sample(sample_g, avg_q_p)
        fw_bar_dict["p"][k] = outside_prob_from_sample(sample_p, avg_q_p)
        fw_bar_dict["sJoe"][k] = outside_prob_from_sample(sample_sJoe, avg_q_sg)
        fw_bar_dict["sJoe_localized"][k] = outside_prob_from_sample(sample_sJoe_localized, avg_q_sg)
        fw_bar_dict["sJoe_local"][k] = outside_prob_from_sample(sample_sJoe_local, avg_q_sg)
        fw_bar_dict["Clayton"][k] = outside_prob_from_sample(sample_Clayton, avg_q_sg)
        fw_bar_dict["sGumbel"][k] = outside_prob_from_sample(sample_sg, avg_q_sg)

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
        "sJoe": {
            "oracle": (oracle_u_sGumbel, {"theta": theta_sJoe}),
            "ecdf": (ecdf_u_sGumbel, {"theta": theta_sJoe}),
        },
        "sJoe_localized": {
            "oracle": (oracle_u_sGumbel, {"theta": theta_sJoe_localized}),
            "ecdf": (ecdf_u_sGumbel, {"theta": theta_sJoe_localized}),
        },
        "sJoe_local": {
            "oracle": (oracle_u_sGumbel, {"theta": theta_sJoe_local}),
            "ecdf": (ecdf_u_sGumbel, {"theta": theta_sJoe_local}),
        },
        "Clayton": {
            "oracle": (oracle_u_sGumbel, {"theta": theta_Clayton}),
            "ecdf": (ecdf_u_sGumbel, {"theta": theta_Clayton}),
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

    # Print summary statistics for each score/model/pit combination
    # zero_rate_threshold = 0.1
    # for score, model_dict in score_vecs.items():
    #     for model, pit_dict in model_dict.items():
    #         for pit, vec in pit_dict.items():
    #             summary_mean = float(np.mean(vec))
    #             summary_std = float(np.std(vec))
    #             summary_min = float(np.min(vec))
    #             summary_max = float(np.max(vec))
    #             zero_fraction = float(np.sum(vec == 0) / len(vec))
    #             print(
    #                 f"Summary for {score}, {model}, {pit}: "
    #                 f"mean={summary_mean:.4f}, "
    #                 f"range=({summary_min:.4f}, {summary_max:.4f}), "
    #                 f"std={summary_std:.4f}"
    #             )
    #             if zero_fraction > zero_rate_threshold:
    #                 print(
    #                     f"  Warning: {zero_fraction:.0%} of entries are zero"
    #                 )
    # print("----------------------------------------------------------------------------")

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

    simulate_one_rep(n, df, f_rho, g_rho, p_rho, theta_sGumbel)
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
    model_pairs = list(combinations(copula_models_for_plots, 2))  # [('f', 'g'), ('f', 'p'), ..., ('sJoe', 'Clayton')]

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
        if (
                key_oracle_fg in dm_rejection_rates[score]
                and key_ecdf_fg in dm_rejection_rates[score]
        ):
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

            plot_dm_size_discrepancy(
                alpha_grid,
                rates_oracle,
                rates_ecdf,
                score,
            )

    for score, label, oracle_key, ecdf_key in score_score_keys:
        if (
                oracle_key in dm_rejection_rates.get(score, {})
                and ecdf_key in dm_rejection_rates.get(score, {})
        ):
            rates_oracle = dm_rejection_rates[score][oracle_key]["two-sided"]
            rates_ecdf = dm_rejection_rates[score][ecdf_key]["two-sided"]

            plt.figure()
            plt.plot(alpha_grid, rates_oracle, label="oracle")
            plt.plot(alpha_grid, rates_ecdf, label="ecdf")
            plt.plot(
                alpha_grid,
                alpha_grid,
                "--",
                color="gray",
                label="alpha = rejection rate",
            )
            plt.xlabel("alpha")
            plt.ylabel("rejection rate")
            plt.title(f"DM test size curves ({score}, {label})")
            plt.legend()
            plt.grid(True)
            plt.show()

            plot_dm_size_discrepancy(
                alpha_grid,
                rates_oracle,
                rates_ecdf,
                score,
                label,
            )

    # === Plot score differences for sJoe - Clayton to visually inspect results ===
    for sc in score_types:
        for key in list(diffs[sc].keys()):
            diffs[sc][key] = div_by_stdev(str(key), diffs[sc][key])

    diff_keys = sorted({k for d in diffs.values() for k in d})
    score_dicts = make_score_dicts(diffs, diff_keys, score_types)

    plot_aligned_kl_matched_scores(score_dicts, score_score_keys)
