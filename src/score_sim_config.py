from utils.structure_defs import DiffKey
# Magic numbers
iSeed = 12085278    # Seed

R = 100    # in-sample window size
P = 100    # out of sample evaluation period size
n = R + P   # total path size (n=250 is year of observations where the market is open)
df = 5  # degrees of freedom for student-t distributions
f_rho = -0.3    # candidate correlation coefficient for bivariate student-t copula
g_rho = 0.3 # candidate correlation coefficient for bivariate student-t copula
p_rho = 0   # DGP correlation coefficient for bivariate student-t copula
theta_sGumbel = 2   # DGP dependence parameter for survival Gumbel copula
theta_Clayton = 3
reps = 1000 # Simulation repetitions
q_threshold = 0.05  # percentile to create the region mask
kl_match_optim_method = "Powell"  # scipy minimize optimization method: Powell, COBYLA (bad option), L-BFGS-B (bad option)
bb1_param_bounds = [(0.001, 7), (1.001, 7)] # R VineCopula bb1 par and par2 bounds
sJoe_param_bounds = [(1.001, 30)]

pit_types = ["oracle", "ecdf"]
score_types = ["LogS", "CS", "CLS"]
all_copula_models = ["f", "g", "p", "sJoe", "sJoe_local", "sJoe_localized", "Clayton", "sGumbel"]
copula_models_for_plots = ["f", "g", "p", "sJoe", "sJoe_local", "sJoe_localized", "Clayton",]
pair_to_keys = {
    "f - g": (DiffKey("oracle", "f", "g"), DiffKey("ecdf", "f", "g")),
    "f - p": (DiffKey("oracle", "f", "p"), DiffKey("ecdf", "f", "p")),
    "g - p": (DiffKey("oracle", "g", "p"), DiffKey("ecdf", "g", "p"))
}
pair_to_keys_size = {
    "f - g": (DiffKey("oracle", "f", "g"), DiffKey("ecdf", "f", "g"))
}
score_score_keys = [
    ("LogS",  "sJoe - Clayton", DiffKey("oracle", "sJoe", "Clayton"), DiffKey("ecdf", "sJoe", "Clayton")),
    ("CS",    "sJoe_localized - Clayton", DiffKey("oracle", "sJoe_localized", "Clayton"), DiffKey("ecdf", "sJoe_localized", "Clayton")),
    ("CLS",   "sJoe_local - Clayton", DiffKey("oracle", "sJoe_local", "Clayton"), DiffKey("ecdf", "sJoe_local", "Clayton")),
]

