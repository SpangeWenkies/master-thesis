from matplotlib.mlab import detrend_linear

from src.utils.utils import sim_clayton_PITs
from utils.utils import *
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from scipy.optimize import minimize

n = 100 # a year of daily observations where the market is open
df = 5
f_rho = -0.3
g_rho = 0.3
p_rho = 0
theta_clayton = 2
# delta = 3
reps = 10000

# Simulate oracle and ECDF PITs using full process
oracle_samples_list = []

for _ in range(reps):
    sim_u_clayton = sim_clayton_PITs(n, theta_clayton)
    oracle_samples_list.append(sim_u_clayton)

pdf_clayton = lambda u: clayton_copula_pdf_from_PITs(u, theta_clayton)
pdf_g = lambda u: student_t_copula_pdf_from_PITs(u, rho=g_rho, df=df)

# === Oracle KL matching ===
kl_oracle_clayton_g_list = [estimate_kl_divergence_copulas(u, pdf_clayton, pdf_g) for u in oracle_samples_list]
target_kl_oracle = np.mean(kl_oracle_clayton_g_list)

def bb1_oracle_objective(params):
    theta, delta = params
    if theta <= 0 or delta < 1:
        return np.inf
    pdf_bb1 = lambda u: bb1_copula_pdf_from_PITs(u, theta, delta)
    kl_vals = [estimate_kl_divergence_copulas(u, pdf_clayton, pdf_bb1) for u in oracle_samples_list]
    return (np.mean(kl_vals) - target_kl_oracle) ** 2

res_oracle = minimize(
    bb1_oracle_objective,
    x0=[2.0, 2.5],
    bounds=[(0.01, 30), (1, 30)],
    method="L-BFGS-B"
)

theta_bb1_oracle, delta_bb1_oracle = res_oracle.x
pdf_bb1_opt = lambda u: bb1_copula_pdf_from_PITs(u, theta_bb1_oracle, delta_bb1_oracle)
print(f"Tuned BB1 (oracle PITs): theta = {theta_bb1_oracle:.4f}, delta = {delta_bb1_oracle:.4f}")
print(f"Target KL(clayton||g) oracle: {target_kl_oracle:.6f}")
print(f"Optimized KL(clayton||bb1): {estimate_kl_divergence_copulas(np.vstack(oracle_samples_list), pdf_clayton, pdf_bb1_opt)}")

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

def plot_score_diff_histogram_kde(diff_oracle, diff_ecdf, title, title2):
    """
    Plots histogram and KDE lines of score differences using oracle and ECDF-based PITs.

    Inputs:
        diff_oracle : np.ndarray – score differences (copula 1 - copula 2) using oracle PITs
        diff_ecdf   : np.ndarray – score differences (copula 1 - copula 2) using ECDF PITs
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

    plt.title(f"{title} – Score Difference {title2}")
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



def simulate_one_rep(n, df, f_rho, g_rho, p_rho, theta_oracle, delta_oracle, theta_clayton):
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
    sim_u_p = student_t.cdf(samples_p, df=df)
    w_p = region_weight_function(sim_u_p, 0.05, df)

    # Define DGP2 (clayton)
    sim_u_clayton = sim_clayton_PITs(n, theta_clayton)
    samples_clayton = np.column_stack([
        student_t.ppf(sim_u_clayton[:, 0], df),
        student_t.ppf(sim_u_clayton[:, 1], df)
    ])
    estim_u_clayton = ecdf_transform(samples_clayton)
    w_clayton = region_weight_function(sim_u_clayton, 0.05, df)

    return {
        "LogS_f_oracle": LogS_student_t_copula(sim_u_p, f_rho, df), #based on DGP1 (indep. student-t)
        "LogS_g_oracle": LogS_student_t_copula(sim_u_p, g_rho, df), #based on DGP1 (indep. student-t)
        "LogS_g_for_KL_matching_oracle": LogS_student_t_copula(sim_u_clayton, g_rho, df), #based on DGP2 (clayton)
        "LogS_p_oracle": LogS_student_t_copula(sim_u_p, p_rho, df), #based on DGP1 (indep. student-t)
        "LogS_bb1_oracle": LogS_bb1(sim_u_clayton, theta_oracle, delta_oracle), #based on DGP2 (clayton)
        "LogS_clayton_oracle": LogS_clayton(sim_u_clayton, theta_clayton), #based on DGP2 (clayton)
        "CS_f_oracle": CS_student_t_copula(sim_u_p, f_rho, df, w_p),
        "CS_g_oracle": CS_student_t_copula(sim_u_p, g_rho, df, w_p),
        "CS_g_for_KL_matching_oracle": CS_student_t_copula(sim_u_clayton, g_rho, df, w_clayton),
        "CS_p_oracle": CS_student_t_copula(sim_u_p, p_rho, df, w_p),
        "CS_bb1_oracle": CS_bb1(sim_u_clayton, theta_oracle, delta_oracle, w_clayton),
        "CS_clayton_oracle": CS_clayton(sim_u_clayton, theta_clayton, w_clayton),
        "CLS_f_oracle": CLS_student_t_copula(sim_u_p, f_rho, df, w_p),
        "CLS_g_oracle": CLS_student_t_copula(sim_u_p, g_rho, df, w_p),
        "CLS_g_for_KL_matching_oracle": CLS_student_t_copula(sim_u_clayton, g_rho, df, w_clayton),
        "CLS_p_oracle": CLS_student_t_copula(sim_u_p, p_rho, df, w_p),
        "CLS_bb1_oracle": CLS_bb1(sim_u_clayton, theta_oracle, delta_oracle, w_clayton),
        "CLS_clayton_oracle": CLS_clayton(sim_u_clayton, theta_clayton, w_clayton),
        "LogS_f_ecdf": LogS_student_t_copula(estim_u_p, f_rho, df),
        "LogS_g_ecdf": LogS_student_t_copula(estim_u_p, g_rho, df),
        "LogS_g_for_KL_matching_ecdf": LogS_student_t_copula(estim_u_clayton, g_rho, df),
        "LogS_p_ecdf": LogS_student_t_copula(estim_u_p, p_rho, df),
        "LogS_bb1_ecdf": LogS_bb1(estim_u_clayton, theta_oracle, delta_oracle),
        "LogS_clayton_ecdf": LogS_clayton(estim_u_clayton, theta_oracle),
        "CS_f_ecdf": CS_student_t_copula(estim_u_p, f_rho, df, w_p),
        "CS_g_ecdf": CS_student_t_copula(estim_u_p, g_rho, df, w_p),
        "CS_g_for_KL_matching_ecdf": CS_student_t_copula(estim_u_clayton, g_rho, df, w_clayton),
        "CS_p_ecdf": CS_student_t_copula(estim_u_p, p_rho, df, w_p),
        "CS_bb1_ecdf": CS_bb1(estim_u_clayton, theta_oracle, delta_oracle, w_clayton),
        "CS_clayton_ecdf": CS_clayton(estim_u_clayton, theta_oracle, w_clayton),
        "CLS_f_ecdf": CLS_student_t_copula(estim_u_p, f_rho, df, w_p),
        "CLS_g_ecdf": CLS_student_t_copula(estim_u_p, g_rho, df, w_p),
        "CLS_g_for_KL_matching_ecdf": CLS_student_t_copula(estim_u_clayton, g_rho, df, w_clayton),
        "CLS_p_ecdf": CLS_student_t_copula(estim_u_p, p_rho, df, w_p),
        "CLS_bb1_ecdf": CLS_bb1(estim_u_clayton, theta_oracle, delta_oracle, w_clayton),
        "CLS_clayton_ecdf": CLS_clayton(estim_u_clayton, theta_oracle, w_clayton),
    }

if __name__ == '__main__':

    results = []

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(simulate_one_rep, n, df, f_rho, g_rho, p_rho, theta_bb1_oracle,
                                   delta_bb1_oracle, theta_clayton) for _ in range(reps)]

        for future in tqdm(as_completed(futures), total=reps, desc="Running simulations"):
            results.append(future.result())

    vecLogS_student_t_copula_f_oracle = np.array([res["LogS_f_oracle"] for res in results]) #DGP1
    vecLogS_student_t_copula_g_oracle = np.array([res["LogS_g_oracle"] for res in results]) #DGP1
    vecLogS_student_t_copula_p_oracle = np.array([res["LogS_p_oracle"] for res in results]) #DGP1
    vecLogS_g_for_KL_matching_oracle = np.array([res["LogS_g_for_KL_matching_oracle"] for res in results]) #DGP2
    vecLogS_bb1_oracle = np.array([res["LogS_bb1_oracle"] for res in results]) #DGP2
    vecLogS_clayton_oracle = np.array([res["LogS_clayton_oracle"] for res in results]) #DGP2

    vecCS_student_t_copula_f_oracle = np.array([res["CS_f_oracle"] for res in results])
    vecCS_student_t_copula_g_oracle = np.array([res["CS_g_oracle"] for res in results])
    vecCS_student_t_copula_p_oracle = np.array([res["CS_p_oracle"] for res in results])
    vecCS_g_for_KL_matching_oracle = np.array([res["CS_g_for_KL_matching_oracle"] for res in results])
    vecCS_bb1_oracle = np.array([res["CS_bb1_oracle"] for res in results])
    vecCS_clayton_oracle = np.array([res["CS_clayton_oracle"] for res in results])

    vecCLS_student_t_copula_f_oracle = np.array([res["CLS_f_oracle"] for res in results])
    vecCLS_student_t_copula_g_oracle = np.array([res["CLS_g_oracle"] for res in results])
    vecCLS_student_t_copula_p_oracle = np.array([res["CLS_p_oracle"] for res in results])
    vecCLS_g_for_KL_matching_oracle = np.array([res["CLS_g_for_KL_matching_oracle"] for res in results])
    vecCLS_bb1_oracle = np.array([res["CLS_bb1_oracle"] for res in results])
    vecCLS_clayton_oracle = np.array([res["CLS_clayton_oracle"] for res in results])

    vecLogS_student_t_copula_f_ecdf = np.array([res["LogS_f_ecdf"] for res in results])
    vecLogS_student_t_copula_g_ecdf = np.array([res["LogS_g_ecdf"] for res in results])
    vecLogS_student_t_copula_p_ecdf = np.array([res["LogS_p_ecdf"] for res in results])
    vecLogS_g_for_KL_matching_ecdf = np.array([res["LogS_g_for_KL_matching_ecdf"] for res in results])
    vecLogS_bb1_ecdf = np.array([res["LogS_bb1_ecdf"] for res in results])
    vecLogS_clayton_ecdf = np.array([res["LogS_clayton_ecdf"] for res in results])

    vecCS_student_t_copula_f_ecdf = np.array([res["CS_f_ecdf"] for res in results])
    vecCS_student_t_copula_g_ecdf = np.array([res["CS_g_ecdf"] for res in results])
    vecCS_student_t_copula_p_ecdf = np.array([res["CS_p_ecdf"] for res in results])
    vecCS_g_for_KL_matching_ecdf = np.array([res["CS_g_for_KL_matching_ecdf"] for res in results])
    vecCS_bb1_ecdf = np.array([res["CS_bb1_ecdf"] for res in results])
    vecCS_clayton_ecdf = np.array([res["CS_clayton_ecdf"] for res in results])

    vecCLS_student_t_copula_f_ecdf = np.array([res["CLS_f_ecdf"] for res in results])
    vecCLS_student_t_copula_g_ecdf = np.array([res["CLS_g_ecdf"] for res in results])
    vecCLS_student_t_copula_p_ecdf = np.array([res["CLS_p_ecdf"] for res in results])
    vecCLS_g_for_KL_matching_ecdf = np.array([res["CLS_g_for_KL_matching_ecdf"] for res in results])
    vecCLS_bb1_ecdf = np.array([res["CLS_bb1_ecdf"] for res in results])
    vecCLS_clayton_ecdf = np.array([res["CLS_clayton_ecdf"] for res in results])

    # vecKL_value_p_bb1_oracle = np.array([res["kl_value_p_bb1_oracle"] for res in results])
    # vecKL_value_p_bb1_ecdf = np.array([res["kl_value_p_bb1_ecdf"] for res in results])
    #
    # vecKL_value_p_g_oracle = np.array([res["kl_value_p_g_oracle"] for res in results])
    # vecKL_value_p_g_ecdf = np.array([res["kl_value_p_g_ecdf"] for res in results])

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

    LogS_diffs_oracle4 = vecLogS_bb1_oracle - vecLogS_g_for_KL_matching_oracle
    CS_diffs_oracle4 = vecCS_bb1_oracle - vecCS_g_for_KL_matching_oracle
    CLS_diffs_oracle4 = vecCLS_bb1_oracle - vecCLS_g_for_KL_matching_oracle

    LogS_diffs_ecdf4 = vecLogS_bb1_ecdf - vecLogS_g_for_KL_matching_ecdf
    CS_diffs_ecdf4 = vecCS_bb1_ecdf - vecCS_g_for_KL_matching_ecdf
    CLS_diffs_ecdf4 = vecCLS_bb1_ecdf - vecCLS_g_for_KL_matching_ecdf



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

    # plot_score_diff_histogram_kde(LogS_diffs_oracle, LogS_diffs_ecdf, "Logarithmic Score", "...-...")
    # plot_score_diff_histogram_kde(CS_diffs_oracle, CS_diffs_ecdf, "Censored Score", "...-...")
    # plot_score_diff_histogram_kde(CLS_diffs_oracle, CLS_diffs_ecdf, "Conditional likelihood Score", "...-...")

    plot_all_score_diffs_side_by_side(score_dict_oracle, score_dict_ecdf, ["LogS", "CS", "CLS"], "f - g (DPG1)")
    plot_all_score_diffs_side_by_side(score_dict_oracle2, score_dict_ecdf2, ["LogS", "CS", "CLS"], "p - g (DGP1)")
    plot_all_score_diffs_side_by_side(score_dict_oracle3, score_dict_ecdf3, ["LogS", "CS", "CLS"], "f - p (DGP1")
    plot_all_score_diffs_side_by_side(score_dict_oracle4, score_dict_ecdf4, ["LogS", "CS", "CLS"], "bb1 - g (DGP2)")

    # print(f"Mean KL divergence, p to bb1, oracle case: {np.mean(vecKL_value_p_bb1_oracle)}")
    # print(f"Mean KL divergence, p to bb1, ecdf case: {np.mean(vecKL_value_p_bb1_ecdf)}")
    #
    # print(f"Mean KL divergence, p to g, oracle case: {np.mean(vecKL_value_p_g_oracle)}")
    # print(f"Mean KL divergence, p to g, ecdf case: {np.mean(vecKL_value_p_g_ecdf)}")