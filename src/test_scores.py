from utils.utils import *
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

n = 100 # a year of daily observations where the market is open
df = 5
f_rho = -0.3
g_rho = 0.3
p_rho = 0
theta = 2
delta = 3
reps = 10000

def plot_histogram_kde(data_f, data_g, data_p, title, pit_type):
    """
        Plots histogram and KDE lines

        Inputs:
            data_f, data_g, data_p  :   The two candidate dist and true dist
            title :                     The name of the score for which the histogram will be plotted

        Returns:
            histograms and KDE lines overlays
        """
    kde_f = gaussian_kde(data_f)
    kde_g = gaussian_kde(data_g)
    kde_p = gaussian_kde(data_p)

    x_min = min(data_f.min(), data_g.min(), data_p.min())
    x_max = max(data_f.max(), data_g.max(), data_p.max())
    x_grid = np.linspace(x_min, x_max, 500)

    plt.figure(figsize=(10, 6))
    plt.hist(data_f, bins=30, alpha=0.3, density=True, label='Model f (rho=-0.3)', color='blue')
    plt.hist(data_g, bins=30, alpha=0.3, density=True, label='Model g (rho=0.3)', color='green')
    plt.hist(data_p, bins=30, alpha=0.3, density=True, label='Model p (rho=0)', color='red')

    plt.plot(x_grid, kde_f(x_grid), label='KDE f', linewidth=2, color='blue')
    plt.plot(x_grid, kde_g(x_grid), label='KDE g', linewidth=2, color='green')
    plt.plot(x_grid, kde_p(x_grid), label='KDE p', linewidth=2, color='red')

    plt.title(f"{title} Score Distribution over {len(data_f)} Repetitions for {pit_type} PITs ")
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_score_diff_histogram_kde(diff_oracle, diff_ecdf, title):
    """
    Plots histogram and KDE lines of score differences using oracle and ECDF-based PITs.

    Inputs:
        diff_oracle : np.ndarray – score differences (f - g) using oracle PITs
        diff_ecdf   : np.ndarray – score differences (f - g) using ECDF PITs
        title       : str – title of the score type (e.g., "LogS", "CS", "CLS")

    Output:
        Histogram and KDE plot
    """
    kde_oracle = gaussian_kde(diff_oracle)
    kde_ecdf = gaussian_kde(diff_ecdf)

    x_min = min(diff_oracle.min(), diff_ecdf.min())
    x_max = max(diff_oracle.max(), diff_ecdf.max())
    x_grid = np.linspace(x_min, x_max, 500)

    plt.figure(figsize=(10, 6))
    plt.hist(diff_oracle, bins=30, alpha=0.3, density=True, label='Oracle PITs', color='purple')
    plt.hist(diff_ecdf, bins=30, alpha=0.3, density=True, label='ECDF PITs', color='orange')

    plt.plot(x_grid, kde_oracle(x_grid), label='KDE Oracle', linewidth=2, color='purple')
    plt.plot(x_grid, kde_ecdf(x_grid), label='KDE ECDF', linewidth=2, color='orange')

    plt.title(f"{title} – Score Difference (Model f - g)")
    plt.xlabel("Score Difference")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_all_score_diffs_side_by_side(score_dict_oracle, score_dict_ecdf, score_names, title):
    """
    Creates subplots comparing oracle and ECDF score differences for different scoring rules.

    Parameters:
    - score_dict_oracle: dict of arrays for score differences using oracle PITs
    - score_dict_ecdf: dict of arrays for score differences using ECDF PITs
    - score_names: list of strings for each score type
    """
    num_scores = len(score_names)
    fig, axes = plt.subplots(1, num_scores, figsize=(6 * num_scores, 5), sharey=True)

    if num_scores == 1:
        axes = [axes]

    for ax, score in zip(axes, score_names):
        data_oracle = score_dict_oracle[score]
        data_ecdf = score_dict_ecdf[score]

        kde_oracle = gaussian_kde(data_oracle)
        kde_ecdf = gaussian_kde(data_ecdf)

        x_min = min(data_oracle.min(), data_ecdf.min())
        x_max = max(data_oracle.max(), data_ecdf.max())
        x_grid = np.linspace(x_min, x_max, 500)

        ax.hist(data_oracle, bins=30, alpha=0.3, density=True, label='Oracle', color='blue')
        ax.hist(data_ecdf, bins=30, alpha=0.3, density=True, label='ECDF', color='orange')

        ax.plot(x_grid, kde_oracle(x_grid), label='KDE Oracle', linewidth=2, color='blue')
        ax.plot(x_grid, kde_ecdf(x_grid), label='KDE ECDF', linewidth=2, color='orange')

        ax.set_title(f"{score} Score Difference for {title}")
        ax.set_xlabel("Score Difference")
        ax.set_ylabel("Density")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()

def region_weight_function(u, q_threshold, df):
    """
    Computes binary weight mask in PIT space based on Y1 + Y2 ≤ q.

    Inputs:
        u : (n, 2) array of PITs
        q_threshold : float, quantile threshold (e.g. 5% sum cutoff)
        df : degrees of freedom of Student-t marginals

    Returns:
        (n,) binary array: 1 if y1 + y2 ≤ q_threshold, else 0
    """

    u_seq = np.linspace(0.005, 0.995, n)
    U1, U2 = np.meshgrid(u_seq, u_seq)

    y1 = student_t.ppf(u[:, 0], df)
    y2 = student_t.ppf(u[:, 1], df)

    q_true = np.quantile(y1 + y2, q_threshold)

    Y1 = student_t.ppf(U1, df)
    Y2 = student_t.ppf(U2, df)

    return ((Y1 + Y2) <= q_true).astype(int)

def simulate_one_rep(n, df, f_rho, g_rho, p_rho, theta, delta):
    """
        Helper function for simulating one repetition in multi-threading

        Inputs:
            inputs for all functions used in one repetition

        Returns:
            different scores of the candidates and true DGP
        """
    samples = multivariate_t.rvs(loc=[0, 0], shape=[[1, 0], [0, 1]], df=df, size=n)
    estim_u = ecdf_transform(samples)
    sim_u = student_t.cdf(samples, df=df)
    w = region_weight_function(sim_u, 0.05, df)

    return {
        "LogS_f_oracle": LogS_student_t_copula(sim_u, f_rho, df),
        "LogS_g_oracle": LogS_student_t_copula(sim_u, g_rho, df),
        "LogS_p_oracle": LogS_student_t_copula(sim_u, p_rho, df),
        "LogS_bb7_oracle": LogS_bb7(sim_u, theta, delta),
        "CS_f_oracle": CS_student_t_copula(sim_u, f_rho, df, w),
        "CS_g_oracle": CS_student_t_copula(sim_u, g_rho, df, w),
        "CS_p_oracle": CS_student_t_copula(sim_u, p_rho, df, w),
        "CS_bb7_oracle": CS_bb7(sim_u, theta, delta, w),
        "CLS_f_oracle": CLS_student_t_copula(sim_u, f_rho, df, w),
        "CLS_g_oracle": CLS_student_t_copula(sim_u, g_rho, df, w),
        "CLS_p_oracle": CLS_student_t_copula(sim_u, p_rho, df, w),
        "CLS_bb7_oracle": CLS_bb7(sim_u, theta, delta, w),
        "LogS_f_ecdf": LogS_student_t_copula(estim_u, f_rho, df),
        "LogS_g_ecdf": LogS_student_t_copula(estim_u, g_rho, df),
        "LogS_p_ecdf": LogS_student_t_copula(estim_u, p_rho, df),
        "LogS_bb7_ecdf": LogS_bb7(estim_u, theta, delta),
        "CS_f_ecdf": CS_student_t_copula(estim_u, f_rho, df, w),
        "CS_g_ecdf": CS_student_t_copula(estim_u, g_rho, df, w),
        "CS_p_ecdf": CS_student_t_copula(estim_u, p_rho, df, w),
        "CS_bb7_ecdf": CS_bb7(estim_u, theta, delta, w),
        "CLS_f_ecdf": CLS_student_t_copula(estim_u, f_rho, df, w),
        "CLS_g_ecdf": CLS_student_t_copula(estim_u, g_rho, df, w),
        "CLS_p_ecdf": CLS_student_t_copula(estim_u, p_rho, df, w),
        "CLS_bb7_ecdf": CLS_bb7(estim_u, theta, delta, w),
    }
if __name__ == '__main__':
    results = []

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(simulate_one_rep, n, df, f_rho, g_rho, p_rho, theta, delta) for _ in range(reps)]

        for future in tqdm(as_completed(futures), total=reps, desc="Running simulations"):
            results.append(future.result())

    vecLogS_student_t_copula_f_oracle = np.array([res["LogS_f_oracle"] for res in results])
    vecLogS_student_t_copula_g_oracle = np.array([res["LogS_g_oracle"] for res in results])
    vecLogS_student_t_copula_p_oracle = np.array([res["LogS_p_oracle"] for res in results])
    vecLogS_bb7_oracle = np.array([res["LogS_bb7_oracle"] for res in results])

    vecCS_student_t_copula_f_oracle = np.array([res["CS_f_oracle"] for res in results])
    vecCS_student_t_copula_g_oracle = np.array([res["CS_g_oracle"] for res in results])
    vecCS_student_t_copula_p_oracle = np.array([res["CS_p_oracle"] for res in results])
    vecCS_bb7_oracle = np.array([res["CS_bb7_oracle"] for res in results])

    vecCLS_student_t_copula_f_oracle = np.array([res["CLS_f_oracle"] for res in results])
    vecCLS_student_t_copula_g_oracle = np.array([res["CLS_g_oracle"] for res in results])
    vecCLS_student_t_copula_p_oracle = np.array([res["CLS_p_oracle"] for res in results])
    vecCLS_bb7_oracle = np.array([res["CLS_bb7_oracle"] for res in results])

    vecLogS_student_t_copula_f_ecdf = np.array([res["LogS_f_ecdf"] for res in results])
    vecLogS_student_t_copula_g_ecdf = np.array([res["LogS_g_ecdf"] for res in results])
    vecLogS_student_t_copula_p_ecdf = np.array([res["LogS_p_ecdf"] for res in results])
    vecLogS_bb7_ecdf = np.array([res["LogS_bb7_ecdf"] for res in results])

    vecCS_student_t_copula_f_ecdf = np.array([res["CS_f_ecdf"] for res in results])
    vecCS_student_t_copula_g_ecdf = np.array([res["CS_g_ecdf"] for res in results])
    vecCS_student_t_copula_p_ecdf = np.array([res["CS_p_ecdf"] for res in results])
    vecCS_bb7_ecdf = np.array([res["CS_bb7_ecdf"] for res in results])

    vecCLS_student_t_copula_f_ecdf = np.array([res["CLS_f_ecdf"] for res in results])
    vecCLS_student_t_copula_g_ecdf = np.array([res["CLS_g_ecdf"] for res in results])
    vecCLS_student_t_copula_p_ecdf = np.array([res["CLS_p_ecdf"] for res in results])
    vecCLS_bb7_ecdf = np.array([res["CLS_bb7_ecdf"] for res in results])

    LogS_diffs_oracle = vecLogS_student_t_copula_f_oracle - vecLogS_student_t_copula_g_oracle
    CS_diffs_oracle = vecCS_student_t_copula_f_oracle - vecCS_student_t_copula_g_oracle
    CLS_diffs_oracle = vecCLS_student_t_copula_f_oracle - vecCLS_student_t_copula_g_oracle

    LogS_diffs_ecdf = vecLogS_student_t_copula_f_ecdf - vecLogS_student_t_copula_g_ecdf
    CS_diffs_ecdf = vecCS_student_t_copula_f_ecdf - vecCS_student_t_copula_g_ecdf
    CLS_diffs_ecdf = vecCLS_student_t_copula_f_ecdf - vecCLS_student_t_copula_g_ecdf

    LogS_diffs_oracle2 = vecLogS_student_t_copula_p_oracle - vecLogS_student_t_copula_g_oracle
    CS_diffs_oracle2 = vecCS_student_t_copula_p_oracle - vecCS_student_t_copula_g_oracle
    CLS_diffs_oracle2 = vecCLS_student_t_copula_p_oracle - vecCLS_student_t_copula_g_oracle

    LogS_diffs_ecdf2 = vecLogS_student_t_copula_p_ecdf - vecLogS_student_t_copula_g_ecdf
    CS_diffs_ecdf2 = vecCS_student_t_copula_p_ecdf - vecCS_student_t_copula_g_ecdf
    CLS_diffs_ecdf2 = vecCLS_student_t_copula_p_ecdf - vecCLS_student_t_copula_g_ecdf

    LogS_diffs_oracle3 = vecLogS_student_t_copula_f_oracle - vecLogS_student_t_copula_p_oracle
    CS_diffs_oracle3 = vecCS_student_t_copula_f_oracle - vecCS_student_t_copula_p_oracle
    CLS_diffs_oracle3 = vecCLS_student_t_copula_f_oracle - vecCLS_student_t_copula_p_oracle

    LogS_diffs_ecdf3 = vecLogS_student_t_copula_f_ecdf - vecLogS_student_t_copula_p_ecdf
    CS_diffs_ecdf3 = vecCS_student_t_copula_f_ecdf - vecCS_student_t_copula_p_ecdf
    CLS_diffs_ecdf3 = vecCLS_student_t_copula_f_ecdf - vecCLS_student_t_copula_p_ecdf

    LogS_diffs_oracle4 = vecLogS_bb7_oracle - vecLogS_student_t_copula_g_oracle
    CS_diffs_oracle4 = vecCS_bb7_oracle - vecCS_student_t_copula_g_oracle
    CLS_diffs_oracle4 = vecCLS_bb7_oracle - vecCLS_student_t_copula_g_oracle

    LogS_diffs_ecdf4 = vecLogS_bb7_ecdf - vecLogS_student_t_copula_g_ecdf
    CS_diffs_ecdf4 = vecCS_bb7_ecdf - vecCS_student_t_copula_g_ecdf
    CLS_diffs_ecdf4 = vecCLS_bb7_ecdf - vecCLS_student_t_copula_g_ecdf

    # For plotting divide CS and CLS by std dev
    LogS_diffs_ecdf = LogS_diffs_ecdf / LogS_diffs_ecdf.std()
    CS_diffs_ecdf = CS_diffs_ecdf / CS_diffs_ecdf.std()
    CLS_diffs_ecdf = CLS_diffs_ecdf / CLS_diffs_ecdf.std()
    LogS_diffs_oracle = LogS_diffs_oracle / LogS_diffs_oracle.std()
    CS_diffs_oracle = CS_diffs_oracle / CS_diffs_oracle.std()
    CLS_diffs_oracle = CLS_diffs_oracle / CLS_diffs_oracle.std()

    LogS_diffs_ecdf2 = LogS_diffs_ecdf2 / LogS_diffs_ecdf2.std()
    CS_diffs_ecdf2 = CS_diffs_ecdf2 / CS_diffs_ecdf2.std()
    CLS_diffs_ecdf2 = CLS_diffs_ecdf2 / CLS_diffs_ecdf2.std()
    LogS_diffs_oracle2 = LogS_diffs_oracle2 / LogS_diffs_oracle2.std()
    CS_diffs_oracle2 = CS_diffs_oracle2 / CS_diffs_oracle2.std()
    CLS_diffs_oracle2 = CLS_diffs_oracle2 / CLS_diffs_oracle2.std()

    LogS_diffs_ecdf3 = LogS_diffs_ecdf3 / LogS_diffs_ecdf3.std()
    CS_diffs_ecdf3 = CS_diffs_ecdf3 / CS_diffs_ecdf3.std()
    CLS_diffs_ecdf3 = CLS_diffs_ecdf3 / CLS_diffs_ecdf3.std()
    LogS_diffs_oracle3 = LogS_diffs_oracle3 / LogS_diffs_oracle3.std()
    CS_diffs_oracle3 = CS_diffs_oracle3 / CS_diffs_oracle3.std()
    CLS_diffs_oracle3 = CLS_diffs_oracle3 / CLS_diffs_oracle3.std()

    LogS_diffs_ecdf4 = LogS_diffs_ecdf4 / LogS_diffs_ecdf4.std()
    CS_diffs_ecdf4 = CS_diffs_ecdf4 / CS_diffs_ecdf4.std()
    CLS_diffs_ecdf4 = CLS_diffs_ecdf4 / CLS_diffs_ecdf4.std()
    LogS_diffs_oracle4 = LogS_diffs_oracle4 / LogS_diffs_oracle4.std()
    CS_diffs_oracle4 = CS_diffs_oracle4 / CS_diffs_oracle4.std()
    CLS_diffs_oracle4 = CLS_diffs_oracle4 / CLS_diffs_oracle4.std()

    score_dict_oracle = {"LogS" : LogS_diffs_oracle, "CS" : CS_diffs_oracle, "CLS" : CLS_diffs_oracle}
    score_dict_ecdf = {"LogS": LogS_diffs_ecdf, "CS": CS_diffs_ecdf, "CLS": CLS_diffs_ecdf}
    score_dict_oracle2 = {"LogS": LogS_diffs_oracle2, "CS": CS_diffs_oracle2, "CLS": CLS_diffs_oracle2}
    score_dict_ecdf2 = {"LogS": LogS_diffs_ecdf2, "CS": CS_diffs_ecdf2, "CLS": CLS_diffs_ecdf2}
    score_dict_oracle3 = {"LogS": LogS_diffs_oracle3, "CS": CS_diffs_oracle3, "CLS": CLS_diffs_oracle3}
    score_dict_ecdf3 = {"LogS": LogS_diffs_ecdf3, "CS": CS_diffs_ecdf3, "CLS": CLS_diffs_ecdf3}
    score_dict_oracle4 = {"LogS": LogS_diffs_oracle4, "CS": CS_diffs_oracle4, "CLS": CLS_diffs_oracle4}
    score_dict_ecdf4 = {"LogS": LogS_diffs_ecdf4, "CS": CS_diffs_ecdf4, "CLS": CLS_diffs_ecdf4}

    # plot_histogram_kde(vecLogS_student_t_copula_f_oracle, vecLogS_student_t_copula_g_oracle,
    #                    vecLogS_student_t_copula_p_oracle, "Logarithmic Score (LogS) for oracle PITs", "oracle")
    # plot_histogram_kde(vecCS_student_t_copula_f_oracle, vecCS_student_t_copula_g_oracle,
    #                    vecCS_student_t_copula_p_oracle, "Censored Log Score (CS) for oracle PITs", "oracle")
    # plot_histogram_kde(vecCLS_student_t_copula_f_oracle, vecCLS_student_t_copula_g_oracle,
    #                    vecCLS_student_t_copula_p_oracle, "Conditional Log Score (CLS) for oracle PITs", "oracle")
    #
    # plot_histogram_kde(vecLogS_student_t_copula_f_ecdf, vecLogS_student_t_copula_g_ecdf,
    #                    vecLogS_student_t_copula_p_ecdf, "Logarithmic Score (LogS) for ECDF PITs", "ECDF")
    # plot_histogram_kde(vecCS_student_t_copula_f_ecdf, vecCS_student_t_copula_g_ecdf,
    #                    vecCS_student_t_copula_p_ecdf, "Censored Log Score (CS) for ECDF PITs", "ECDF")
    # plot_histogram_kde(vecCLS_student_t_copula_f_ecdf, vecCLS_student_t_copula_g_ecdf,
    #                    vecCLS_student_t_copula_p_ecdf, "Conditional Log Score (CLS) for ECDF PITs", "ECDF")

    # plot_score_diff_histogram_kde(LogS_diffs_oracle, LogS_diffs_ecdf, title="Logarithmic Score")
    # plot_score_diff_histogram_kde(CS_diffs_oracle, CS_diffs_ecdf, title="Censored Score")
    # plot_score_diff_histogram_kde(CLS_diffs_oracle, CLS_diffs_ecdf, title="Conditional likelihood Score")

    plot_all_score_diffs_side_by_side(score_dict_oracle, score_dict_ecdf, ["LogS", "CS", "CLS"], "f - g")
    plot_all_score_diffs_side_by_side(score_dict_oracle2, score_dict_ecdf2, ["LogS", "CS", "CLS"], "p - g")
    plot_all_score_diffs_side_by_side(score_dict_oracle3, score_dict_ecdf3, ["LogS", "CS", "CLS"], "f - p")
    plot_all_score_diffs_side_by_side(score_dict_oracle4, score_dict_ecdf4, ["LogS", "CS", "CLS"], "bb7 - g")