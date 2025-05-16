import numpy as np
import sys
from Plot import Plot
def WhiteNoiseSim(iT, vDistrParams, sDistrName, sPlot):
    """
    Purpose:
        Generate white noise

    Inputs:
        iT              sample size
        vDistrParams    distributional params
        sDistrName      distribution name
        bPlot           boolean for plotting series, default is 1

    Return value:
        vEps            white noise realisations

    Output:
        Plot realisations process

    """
    lDistrName = ['normal', 't']
    if (sDistrName in lDistrName):
        if (sDistrName == lDistrName[0]):
            vEps = np.random.randn(iT)
        elif (sDistrName == lDistrName[1]):
            vEps = np.random.standard_t(vDistrParams[0], size=iT)

        if (sPlot == 'line'):
            Plot(vEps, "White noise", True)
        elif (sPlot == 'hist'):
            Plot(vEps, "White noise", False)
        return vEps
    else:
        sys.exit(
            "Distribution not supported.")  # print("Error: Distribution not supported. Supported distributions are:"); print(*lDistrName, sep = ", ");
