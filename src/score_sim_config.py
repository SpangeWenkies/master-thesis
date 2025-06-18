
# Magic numbers
iSeed = 12085278                            # Seed

n = 250                                     # sample size (n=250 is year of observations where the market is open)
df = 5                                      # degrees of freedom for student-t distributions
f_rho = -0.3                                # candidate correlation coefficient for bivariate student-t copula
g_rho = 0.3                                 # candidate correlation coefficient for bivariate student-t copula
p_rho = 0                                   # DGP correlation coefficient for bivariate student-t copula
theta_sGumbel = 2                           # DGP dependence parameter for survival Gumbel copula
reps = 5000                             # Simulation repetitions
q_threshold = 0.05                          # percentile to create the region mask
kl_match_optim_method = "L-BFGS-B"          # scipy minimize optimization method
bb1_param_bounds = [(0.001, 7), (1.001, 7)]  # R VineCopula bb1 par and par2 bounds

pit_types = ["oracle", "ecdf"]
score_types = ["LogS", "CS", "CLS"]
all_copula_models = ["f", "g", "p", "bb1", "bb1_local", "bb1_localized", "f_for_KL_matching", "sGumbel"]
copula_models_for_plots = ["f", "g", "p", "bb1", "bb1_local", "bb1_localized", "f_for_KL_matching",]
pair_to_suffixes = {
    "bb1 - f_for_KL_matching": ("oracle_bb1_f_for_KL_matching", "ecdf_bb1_f_for_KL_matching"),
    "bb1_localized - f_for_KL_matching": ("oracle_bb1_localized_f_for_KL_matching", "ecdf_bb1_localized_f_for_KL_matching"),
    "bb1_local - f_for_KL_matching": ("oracle_bb1_local_f_for_KL_matching", "ecdf_bb1_local_f_for_KL_matching"),
    "f - g": ("oracle_f_g", "ecdf_f_g"),
    "f - p": ("oracle_f_p", "ecdf_f_p"),
    "g - p": ("oracle_g_p", "ecdf_g_p")
}
score_score_suffixes = [
    ("LogS",  "bb1 - f_for_KL_matching", "oracle_bb1_f_for_KL_matching", "ecdf_bb1_f_for_KL_matching"),
    ("CS",    "bb1_localized - f_for_KL_matching", "oracle_bb1_localized_f_for_KL_matching", "ecdf_bb1_localized_f_for_KL_matching"),
    ("CLS",   "bb1_local - f_for_KL_matching", "oracle_bb1_local_f_for_KL_matching", "ecdf_bb1_local_f_for_KL_matching"),
]

