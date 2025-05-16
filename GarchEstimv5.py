import numpy as np
import scipy.optimize as opt


def AvgNLnLGARCH(vGARCHParams, vY):
    """
    Purpose:
        Compute negative average loglikelihood of GARCH(1,1) [normal]

    Inputs:
        vGARCHParams    GARCH parameters: dOmega, vAlpha, vBeta
        vY              data

    Return value:
        dNALL           negative average loglikelihood
    """

    dOmega = vGARCHParams[0]
    vAlpha = vGARCHParams[1]
    vBeta = vGARCHParams[2]
    iT = len(vY)
    vSig2 = np.zeros(iT + 1)
    dSig20 = dOmega / (1 - vAlpha - vBeta)
    vSig2[0] = dSig20
    for t in range(1, iT + 1):
        vSig2[t] = dOmega + vAlpha * vY[t - 1] ** 2 + vBeta * vSig2[t - 1]
    vLL = -0.5 * (np.log(2 * np.pi) + np.log(vSig2[0:-1]) + (vY ** 2) / vSig2[0:-1])
    dALL = np.mean(vLL, axis=0)

    return -dALL

def AvgNLnLGARCHTr(vGARCHParamsTr, vY):
    """
    Purpose:
        Provide wrapper around AvgNLnLGARCH(), accepting transformed parameters

    Inputs:
        vGARCHParamsTr    transformed parameters, with vGARCHParamsTr=log(vGARCHParams)
        vY                data

    Return value:
        dNALL             negative average loglikelihood
    """

    vGARCHParams = np.copy(vGARCHParamsTr)
    vGARCHParams = np.exp(vGARCHParamsTr)
    return AvgNLnLGARCH(vGARCHParams, vY)

def AvgNLnLGARCHL(vGARCHParams, vY):
    """
    Purpose:
        Compute negative average loglikelihood of GARCH(1,1) [Laplace]

    Inputs:
        vGARCHParams    GARCH parameters: dOmega, vAlpha, vBeta
        vY              data

    Return value:
        dNALL           negative average loglikelihood
    """

    dOmega = vGARCHParams[0]
    vAlpha = vGARCHParams[1]
    vBeta = vGARCHParams[2]
    iT = len(vY)
    vSig2 = np.zeros(iT + 1)
    dSig20 = dOmega / (1 - vAlpha - vBeta)
    vSig2[0] = dSig20
    for t in range(1, iT + 1):
        vSig2[t] = dOmega + vAlpha * vY[t - 1] ** 2 + vBeta * vSig2[t - 1]
    vLL = -np.log(2) - 0.5 * np.log(0.5 * vSig2[0:-1]) - (np.sqrt(0.5 * vSig2[0:-1]) ** (-1)) * np.abs(vY)
    dALL = np.mean(vLL, axis=0)

    return -dALL

def AvgNLnLGARCHLTr(vGARCHParamsTr, vY):
    """
    Purpose:
        Provide wrapper around AvgNLnLGARCHL(), accepting transformed parameters

    Inputs:
        vGARCHParamsTr    transformed parameters, with vGARCHParamsTr=log(vGARCHParams)
        vY                data

    Return value:
        dNALL             negative average loglikelihood
    """

    vGARCHParams = np.copy(vGARCHParamsTr)
    vGARCHParams = np.exp(vGARCHParamsTr)
    return AvgNLnLGARCHL(vGARCHParams, vY)

def GARCHEstim(vGARCHParams0, vGARCHLParams0, vY):
    """
    Purpose:
        Estimate GARCH(p,q)

    Inputs:
        vGARCHParams0   initial parameters
        vY              data

    Return value:
        vY              realised time series
    """

    iT = len(vY)
    vGARCHParams0Tr = np.log(vGARCHParams0)
    vGARCHLParams0Tr = np.log(vGARCHLParams0)
    res_normal = opt.minimize(AvgNLnLGARCHTr, vGARCHParams0Tr, args=vY, bounds=((-np.inf, 0), (-np.inf, 0), (-np.inf, 0)), method="Nelder-Mead")
    res_laplace = opt.minimize(AvgNLnLGARCHLTr, vGARCHLParams0Tr, args=vY, bounds=((-np.inf, 0), (-np.inf, 0), (-np.inf, 0)), method="Nelder-Mead")

    vGARCHParamsStarTr = np.copy(res_normal.x)
    vGARCHParamsStar = np.exp(vGARCHParamsStarTr)

    sMess = res_normal.message
    dLL = -iT * res_normal.fun
    print("\nNormal results in ", sMess, "\nParameter estimates (MLE): ", vGARCHParamsStar, "\nLog-likelihood= ", dLL,
          ", f-eval= ", res_normal.nfev)

    vGARCHLParamsStarTr = np.copy(res_laplace.x)
    vGARCHLParamsStar = np.exp(vGARCHLParamsStarTr)

    sMess_laplace = res_laplace.message
    dLL_laplace = -iT * res_laplace.fun
    print("\nLaplace results in ", sMess_laplace, "\nParameter estimates (MLE): ", vGARCHLParamsStar, "\nLog-likelihood= ", dLL_laplace,
          ", f-eval= ", res_laplace.nfev)

    return vGARCHParamsStar, vGARCHLParamsStar

