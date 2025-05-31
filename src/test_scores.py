from utils.utils import *
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

n = 1000
df = 5
f_rho = -0.3
g_rho = 0.3
p_rho = 0
reps = 1000

def plot_histogram_kde(data_f, data_g, data_p, title):
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

    plt.title(f"{title} Score Distribution over {len(data_f)} Repetitions")
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
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

def simulate_one_rep(n, df, f_rho, g_rho, p_rho):
    samples = multivariate_t.rvs(loc=[0, 0], shape=[[1, 0], [0, 1]], df=df, size=n)
    sim_u = student_t.cdf(samples, df=df)
    w = region_weight_function(sim_u, 0.05, df)

    return {
        "LogS_f": LogS_student_t_copula(sim_u, f_rho, df),
        "LogS_g": LogS_student_t_copula(sim_u, g_rho, df),
        "LogS_p": LogS_student_t_copula(sim_u, p_rho, df),
        "CS_f": CS_student_t_copula(sim_u, f_rho, df, w),
        "CS_g": CS_student_t_copula(sim_u, g_rho, df, w),
        "CS_p": CS_student_t_copula(sim_u, p_rho, df, w),
        "CLS_f": CLS_student_t_copula(sim_u, f_rho, df, w),
        "CLS_g": CLS_student_t_copula(sim_u, g_rho, df, w),
        "CLS_p": CLS_student_t_copula(sim_u, p_rho, df, w),
    }
if __name__ == '__main__':
    results = []

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(simulate_one_rep, n, df, f_rho, g_rho, p_rho) for _ in range(reps)]

        for future in tqdm(as_completed(futures), total=reps, desc="Running simulations"):
            results.append(future.result())

    vecLogS_student_t_copula_f = np.array([res["LogS_f"] for res in results])
    vecLogS_student_t_copula_g = np.array([res["LogS_g"] for res in results])
    vecLogS_student_t_copula_p = np.array([res["LogS_p"] for res in results])

    vecCS_student_t_copula_f = np.array([res["CS_f"] for res in results])
    vecCS_student_t_copula_g = np.array([res["CS_g"] for res in results])
    vecCS_student_t_copula_p = np.array([res["CS_p"] for res in results])

    vecCLS_student_t_copula_f = np.array([res["CLS_f"] for res in results])
    vecCLS_student_t_copula_g = np.array([res["CLS_g"] for res in results])
    vecCLS_student_t_copula_p = np.array([res["CLS_p"] for res in results])

    plot_histogram_kde(vecLogS_student_t_copula_f, vecLogS_student_t_copula_g, vecLogS_student_t_copula_p, "Logarithmic Score (LogS)")
    plot_histogram_kde(vecCS_student_t_copula_f, vecCS_student_t_copula_g, vecCS_student_t_copula_p, "Censored Log Score (CS)")
    plot_histogram_kde(vecCLS_student_t_copula_f, vecCLS_student_t_copula_g, vecCLS_student_t_copula_p, "Conditional Log Score (CLS)")