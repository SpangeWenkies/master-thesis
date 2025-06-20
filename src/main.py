###########################################################
# main
import matplotlib.pyplot as plt
import numpy as np
from utils.plot_utils import Plot
from utils.util_funcs import simulate_GARCH
from utils.util_funcs import resid_to_unif_PIT_ECDF
from utils.util_funcs import R_bb7
from utils.util_funcs import inverse_ecdf
from utils.util_funcs import simulate_joint_t_marginals
from utils.util_funcs import WhiteNoiseSim
from utils.util_funcs import evaluate_mass_in_region
from utils.util_funcs import compute_scores_over_region
from utils.util_funcs import compare_trueU_ecdfU_score
from utils.util_funcs import compare_trueU_ecdfU_score_test_version
from utils.util_funcs import compare_trueU_ecdfU_score_t_loop
from config import iSeed, n, omega, alpha, beta, dist, nu, theta, delta, grid_size, R, P, H, iterations



def main():
    # Cleaning
    plt.close('all')

    # Set seed
    np.random.seed(iSeed)

    # GARCH(p,q)
    # resid1, variance1, white_noise1, resid2, variance2, white_noise2 = simulate_GARCH(n, omega, alpha, beta, dist, df=nu)

    # Standardize residuals for PIT transformation (is this transformation needed or do we use white noise input???)

    # Pit transform
    # u1 = resid_to_unif_PIT_ECDF(white_noise1)
    # u2 = resid_to_unif_PIT_ECDF(white_noise2)

    # sim_u1, sim_u2, copula_cdf_values, W, empirical_pdf, true_pdf = R_bb7(u1, u2, theta, delta, resid1, resid2)

    # sim_resid1 = inverse_ecdf(sim_u1, resid2)
    # sim_resid2 = inverse_ecdf(sim_u2, resid2)

    # true_y1, true_y2, true_u1, true_u2, true_q, U1, U2, W_true, copula_pdf = simulate_joint_t_marginals(n, nu, theta, delta)

    # print(f"True quantile used to calculate the region:  {true_q:.4f}")
    #
    # # mass_empirical = evaluate_mass_in_region(empirical_pdf, W )
    # mass_theoretical = evaluate_mass_in_region(copula_pdf, W_true)  # here we now use the right true pdf
    #
    # # print(f"Mass in region (Empirical):  {mass_empirical:.4f}")
    # print(f"Mass in region (Theoretical BB7): {mass_theoretical:.4f}")
    #
    # # CSL_emp, SCL_emp = compute_scores_over_region(empirical_pdf, W_true)
    # CS_true, CLS_true = compute_scores_over_region(copula_pdf, W_true)
    # print(f"CLS (Conditional Likelihood Score): {CLS_true:.4f}")
    # print(f"CS (Censored Likelihood Score):   {CS_true:.4f}")


    # results = compare_trueU_ecdfU_score(R, P, H, grid_size, theta, delta, nu)

    # test_results = compare_trueU_ecdfU_score_test_version(R, P, H, grid_size, nu)

    loop_results = compare_trueU_ecdfU_score_t_loop(iterations, grid_size, R, P, H, nu)


###########################################################
# start main
if __name__ == "__main__":
    main()