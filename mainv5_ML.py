import numpy as np
import matplotlib.pyplot as plt
import datetime
from GarchSimv2 import simulate_GARCH
from GarchEstimv5 import GARCHEstim


def mainv5_ML():
    # Cleaning
    plt.close('all')

    # Magic numbers
    iT = 3250
    iSeed = 12085278
    dOmega = np.array([0.1])
    vAlpha = np.array([0.1])  # np.array([0.1])
    vBeta = np.array([0.6])
    dNu = 6

    dOmega0 = 0.1
    vAlpha0 = np.array([0.3])  # np.array([0.3, 0.1])
    vBeta0 = np.array([0.3])
    vGARCHParams0 = np.hstack((dOmega0, vAlpha0, vBeta0))

    vGARCHLParams0 = np.copy(vGARCHParams0)

    vQ = np.array([0, 0.01, 0.05, 0.10, 0.20, 0.50, 0.75])

    weightFuncEnd = 5 * np.pi  # higher means more dist switches and less gradual change in weights
    x1 = np.linspace(0, weightFuncEnd, iT)
    vW = 0.5 * np.sin(x1) + 0.5
    # vWones = np.ones(iT)

    # Set seed
    np.random.seed(iSeed)

    vY1, vY1_var = simulate_GARCH(iT, dOmega, vAlpha, vBeta, "normal", dNu)
    vY2, vY2_var = simulate_GARCH(iT, dOmega, vAlpha, vBeta, "t", dNu)

    # plt.plot(vY1)
    # plt.show()
    # plt.close()
    #
    # plt.plot(vY2)
    # plt.show()
    # plt.close()

    # Mixture
    vY = np.zeros(iT)
    for t in range(0, iT):
        vY[t] = vW[t]*vY1[t] + (1-vW[t])*vY2[t]

    # plt.plot(vY, color="blue", label="y")
    # plt.plot(vW, color="black", label="Weights")
    # plt.title("Normal Garch(1,1) + Student-t Garch(1,1) (df=6) , N_w = 5pi")
    # plt.legend(loc="upper left")
    # plt.show()

    np.save("vY_5pi.npy", vY)

    MlRange = 750
    osa_forecast_range = iT-MlRange

    vMlOmegaNormal = np.zeros(osa_forecast_range)
    vMlAlphaNormal = np.zeros(osa_forecast_range)
    vMlBetaNormal = np.zeros(osa_forecast_range)

    vMlOmegaLaplace = np.zeros(osa_forecast_range)
    vMlAlphaLaplace = np.zeros(osa_forecast_range)
    vMlBetaLaplace = np.zeros(osa_forecast_range)

    forecasts_normal = {}
    forecasts_laplace = {}

    res = GARCHEstim(vGARCHParams0, vGARCHLParams0, vY[0:MlRange])
    res_normal = res[0]
    res_laplace = res[1]

    vMlOmegaNormal[0] = res_normal[0]
    vMlAlphaNormal[0] = res_normal[1]
    vMlBetaNormal[0] = res_normal[2]

    vMlOmegaLaplace[0] = res_laplace[0]
    vMlAlphaLaplace[0] = res_laplace[1]
    vMlBetaLaplace[0] = res_laplace[2]

    vSig2_normal = np.zeros(MlRange + 1)
    dSig20_normal = vMlOmegaNormal[0] / (1 - vMlAlphaNormal[0] - vMlBetaNormal[0])
    vSig2_normal[0] = dSig20_normal
    for t in range(1, MlRange + 1):
        vSig2_normal[t] = vMlOmegaNormal[0] + vMlAlphaNormal[0] * vY[t - 1] ** 2 + vMlBetaNormal[0] * vSig2_normal[t-1]

    vSig2_laplace = np.zeros(MlRange + 1)
    dSig20_laplace = vMlOmegaLaplace[0] / (1 - vMlAlphaLaplace[0] - vMlBetaLaplace[0])
    vSig2_laplace[0] = dSig20_laplace
    for t in range(1, MlRange + 1):
        vSig2_laplace[t] = vMlOmegaLaplace[0] + vMlAlphaLaplace[0] * vY[t - 1] ** 2 + vMlBetaLaplace[0] * vSig2_laplace[t-1]

    forecasts_normal[0] = vMlOmegaNormal[0] + vMlAlphaNormal[0] * vY[MlRange] ** 2 + vMlBetaNormal[0] * vSig2_normal[-1]
    forecasts_laplace[0] = vMlOmegaLaplace[0] + vMlAlphaLaplace[0] * vY[MlRange] ** 2 + vMlBetaLaplace[0] * vSig2_laplace[-1]

    for t in range(1, osa_forecast_range):

        vGARCHParams0 = np.array([vMlOmegaNormal[t-1], vMlAlphaNormal[t-1], vMlBetaNormal[t-1]])
        vGARCHLParams0 = np.array([vMlOmegaLaplace[t-1], vMlAlphaLaplace[t-1], vMlBetaLaplace[t-1]])

        res = GARCHEstim(vGARCHParams0, vGARCHLParams0, vY[t:t+MlRange])
        res_normal = res[0]
        res_laplace = res[1]

        vMlOmegaNormal[t] = res_normal[0]
        vMlAlphaNormal[t] = res_normal[1]
        vMlBetaNormal[t] = res_normal[2]

        vMlOmegaLaplace[t] = res_laplace[0]
        vMlAlphaLaplace[t] = res_laplace[1]
        vMlBetaLaplace[t] = res_laplace[2]

        vSig2_normal = np.zeros(MlRange + 1)
        dSig20_normal = vMlOmegaNormal[t] / (1 - vMlAlphaNormal[t] - vMlBetaNormal[t])
        vSig2_normal[0] = dSig20_normal
        for i in range(1, MlRange + 1):
            vSig2_normal[i] = vMlOmegaNormal[t] + vMlAlphaNormal[t] * vY[i + t - 1] ** 2 + vMlBetaNormal[t] * vSig2_normal[
                i - 1]

        vSig2_laplace = np.zeros(MlRange + 1)
        dSig20_laplace = vMlOmegaLaplace[t] / (1 - vMlAlphaLaplace[t] - vMlBetaLaplace[t])
        vSig2_laplace[0] = dSig20_laplace
        for i in range(1, MlRange + 1):
            vSig2_laplace[i] = vMlOmegaLaplace[t] + vMlAlphaLaplace[t] * vY[i + t - 1] ** 2 + vMlBetaLaplace[t] * \
                               vSig2_laplace[i - 1]

        forecasts_normal[t] = vMlOmegaNormal[t] + vMlAlphaNormal[t] * vY[MlRange+t] ** 2 + vMlBetaNormal[t] * \
                              vSig2_normal[-1]
        forecasts_laplace[t] = vMlOmegaLaplace[t] + vMlAlphaLaplace[t] * vY[MlRange+t] ** 2 + vMlBetaLaplace[t] * \
                               vSig2_laplace[-1]

    np.save("ML_normal_omega_5pi.npy", vMlOmegaNormal)
    np.save("ML_normal_alpha_5pi.npy", vMlAlphaNormal)
    np.save("ML_normal_beta_5pi.npy", vMlBetaNormal)

    np.save("ML_laplace_omega_5pi.npy", vMlOmegaLaplace)
    np.save("ML_laplace_alpha_5pi.npy", vMlAlphaLaplace)
    np.save("ML_laplace_beta_5pi.npy", vMlBetaLaplace)

    np.save("forecasts_normal_5pi.npy", np.array(list(forecasts_normal.values())))
    np.save("forecasts_laplace_5pi.npy", np.array(list(forecasts_laplace.values())))
