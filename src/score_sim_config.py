
# Magic numbers
iSeed = 12085278                            # Seed

n = 1000                                     # sample size (n=250 is year of observations where the market is open)
df = 5                                      # degrees of freedom for student-t distributions
f_rho = -0.3                                # candidate correlation coefficient for bivariate student-t copula
g_rho = 0.3                                 # candidate correlation coefficient for bivariate student-t copula
p_rho = 0                                   # DGP correlation coefficient for bivariate student-t copula
theta_sGumbel = 2                           # DGP dependence parameter for survival Gumbel copula
reps = 1000                              # Simulation repetitions
q_threshold = 0.05                          # percentile to create the region mask
kl_match_optim_method = "L-BFGS-B"          # scipy minimize optimization method
bb1_param_bounds = [(0.01, 30), (1.0, 30)]  # R VineCopula bb1 par and par2 bounds

pit_types = ["oracle", "ecdf"]
score_types = ["LogS", "CS", "CLS"]
all_copula_models = ["f", "g", "p", "bb1", "f_for_KL_matching", "sGumbel"]
copula_models_for_plots = ["f", "g", "p", "bb1", "f_for_KL_matching"]


