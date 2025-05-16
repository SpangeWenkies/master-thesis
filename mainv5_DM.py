import numpy as np
from GarchSimv2 import simulate_GARCH
from dm_test import dm_test

def mainv5_DM():
    iT = 3250

    dOmega = np.array([0.1])
    vAlpha = np.array([0.1])  # np.array([0.1])
    vBeta = np.array([0.6])
    dNu = 6

    MlRange = 750
    Tw = 300
    Tq = 250

    weightFuncEnd = 5 * np.pi
    x1 = np.linspace(0, weightFuncEnd, iT)
    vW = 0.5 * np.sin(x1) + 0.5

    vY1, vY1_var = simulate_GARCH(iT, dOmega, vAlpha, vBeta, "normal", dNu)
    vY2, vY2_var = simulate_GARCH(iT, dOmega, vAlpha, vBeta, "t", dNu)

    actual_sig2 = np.zeros(iT)

    for t in range(iT):
        actual_sig2[t] = vW[t] * vY1_var[t] + (1-vW[t]) * vY2_var[t]

    np.save("actual_sig2_5pi.npy", actual_sig2)

    optim_q_weights = np.load("optim_q_weights_5pi.npy")
    weights_q000 = np.load("weights_q000_5pi.npy")

    forecast_normal = np.load("forecasts_normal_5pi.npy")
    forecast_laplace = np.load("forecasts_laplace_5pi.npy")

    weighted_forecast_optim_q = optim_q_weights * forecast_normal[Tw+Tq:-1] + (1-optim_q_weights) * forecast_laplace[Tw+Tq:-1]
    weighted_forecast_q000 = weights_q000[Tq:-1] * forecast_normal[Tw+Tq:-1] + (1-weights_q000[Tq:-1]) * forecast_laplace[Tw+Tq:-1]

    np.save("weighted_forecast_optim_q_5pi.npy", weighted_forecast_optim_q)
    np.save("weighted_forecast_q000_5pi.npy", weighted_forecast_q000)

    actual_sig2 = actual_sig2[MlRange+Tw+Tq:-1]
    np.save("dm_test_results_5pi.npy", np.asarray(dm_test(actual_sig2, weighted_forecast_optim_q, weighted_forecast_q000)))

    return dm_test(actual_sig2, weighted_forecast_optim_q, weighted_forecast_q000)

mainv5_DM()