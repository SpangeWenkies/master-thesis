from utils.utils import *

n = 1000
df = 5
f_rho = -0.3
g_rho = 0.3
p_rho = 0

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

cov = [[1, 0], [0, 1]]
samples = multivariate_t.rvs(loc=[0, 0], shape=cov, df=df, size=n)
sim_u = student_t.cdf(samples, df=df)  # Convert to PITs

w = region_weight_function(sim_u, 0.05, df)

print(f"LogS of f (DGP = p): ", LogS_student_t_copula(sim_u, f_rho, df))
print(f"LogS of g (DGP = p): ", LogS_student_t_copula(sim_u, g_rho, df))
print(f"LogS of p (DGP = p): ", LogS_student_t_copula(sim_u, p_rho, df))

print(f"CS of f (DGP = p): ", CS_student_t_copula(sim_u, f_rho, df, w))
print(f"CS of g (DGP = p): ", CS_student_t_copula(sim_u, g_rho, df, w))
print(f"CS of p (DGP = p): ", CS_student_t_copula(sim_u, p_rho, df, w))

print(f"CLS of f (DGP = p): ", CLS_student_t_copula(sim_u, f_rho, df, w))
print(f"CLS of g (DGP = p): ", CLS_student_t_copula(sim_u, g_rho, df, w))
print(f"CLS of p (DGP = p): ", CLS_student_t_copula(sim_u, p_rho, df, w))



# CS_student_t_copula(sim_u, rho, df, w)
# CLS_student_t_copula(sim_u, rho, df, w)