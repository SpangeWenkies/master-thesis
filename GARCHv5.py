#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GARCH.py

Purpose:
    Simulate and estimate GARCH

Version:
    1       Simulation
    2       Estimation
    3       Robustness checks: identification
    4       Final version
    5       Final version corrected 

Date:
    2020/16/05

Author:
    Ramon de Punder (r.f.a.depunder@uva.nl)
"""
###########################################################
### Imports
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.optimize as opt

## Global settings
lFigureSettings= {'figsize':(10,6), 'dpi':70, 'titlefontsize':16, 'axisfontsize':14} 
iBins= 30 #int(np.floor(len(vX)/25))

###########################################################
### Plot(vX, sTitle, bLine)
def Plot(vX, sTitle, bLine):
    """
    Purpose:
        Plot series vX

    Inputs:
        vX              time series
        bLine           boolean: lineplot (1) or histogram (0)
        

    Output:
        Plot of realisations process
    
    """
    plt.figure(figsize=lFigureSettings['figsize'], dpi=lFigureSettings['dpi'])
    if(bLine == True):
        plt.plot(vX, c='blue') 
        plt.xlabel('$t$', fontsize=lFigureSettings['axisfontsize'])
        plt.ylabel('$y_t$', fontsize=lFigureSettings['axisfontsize'])
    else:
        plt.hist(vX, iBins) 
    plt.title(sTitle, fontsize=lFigureSettings['titlefontsize'])
    plt.show()
    
###########################################################
### vEps= WhiteNoiseSim(iT, vDistrParams, sDistrName, sPlot)
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
    lDistrName= ['normal','t']
    if(sDistrName in lDistrName):
        if(sDistrName == lDistrName[0]):
            vEps=  np.random.randn(iT)
        elif(sDistrName == lDistrName[1]):
            vEps=  np.random.standard_t(vDistrParams[0], size=iT)
    
        if(sPlot == 'line'):
            Plot(vEps,"White noise", True)
        elif(sPlot == 'hist'):
            Plot(vEps,"White noise", False)
        return vEps
    else: sys.exit("Distribution not supported.") #print("Error: Distribution not supported. Supported distributions are:"); print(*lDistrName, sep = ", ");

###########################################################
### vY= GARCHSim(iT, vGARCHParams, iP, vDistrParams, sDistrName, sPlot)
def GARCHSim(iT, vGARCHParams, iP, vDistrParams, sDistrName, sPlot):
    """
    Purpose:
        Simulate GARCH(p,q)

    Inputs:
        iT              sample size
        vGARCHParams    GARCH parameters: dOmega, vAlpha, vBeta
        iP              length vAlpha
        vDistrParams    distributional params
        sDistrName      distribution name
        sPlot           plot type: 'line' or 'hist'

    Return value:
        vY              realised time series

    Output:
        Plot realisations process
    
    """
    dOmega= vGARCHParams[0]
    vAlpha= vGARCHParams[1:iP+1]
    vBeta= vGARCHParams[iP+1:]
    iQ= len(vBeta)
    iR= max(iP, iQ)
    vEps= WhiteNoiseSim(iT+iR,vDistrParams,sDistrName, False)
    vSig2= np.zeros(iT+iR)
    dSig20= dOmega/(1-np.sum(vAlpha)-np.sum(vBeta))
    vSig2[0:iR]= np.repeat(dSig20, iR)
    vY= np.zeros(iT+iR)
    dY0= 0
    vY[0:iR]= np.repeat(dY0, iR)
    for t in range(iR,iT):
        vSig2[t]= dOmega + vAlpha @ vY[t-iP:t][::-1]**2  +  vBeta @ vSig2[t-iQ:t][::-1]
        vY[t] = vEps[t]* np.sqrt(vSig2[t])
    vY= vY[iR:]
    sTitle= "GARCH(%d, %d)" %(iP, iQ)
    if(sPlot == 'line'):
        Plot(vY,sTitle, True)
    elif(sPlot == 'hist'):
        Plot(vY,"sTitle", False)
    return vY

###########################################################
### dLL= AvgNLnLGARCH(vP, vY, mX)
def AvgNLnLGARCH(vGARCHParams, iP, vY):
    """
    Purpose:
        Compute negative average loglikelihood of GARCH(p,q) [normal]

    Inputs:        
        vGARCHParams    GARCH parameters: dOmega, vAlpha, vBeta
        iP              length vAlpha
        vY              data                        

    Return value:
        dNALL           negative average loglikelihood
    """
    
    dOmega= vGARCHParams[0]
    vAlpha= vGARCHParams[1:iP+1]
    vBeta= vGARCHParams[iP+1:]
    iT= len(vY) 
    iQ= len(vBeta)
    iR= max(iP,iQ)
    vSig2= np.zeros(iT+1)
    dSig20= dOmega/(1-np.sum(vAlpha)-np.sum(vBeta))
    vSig2[0:iR]= np.repeat(dSig20, iR)
    for t in range(iR,iT+1):
        vSig2[t]= dOmega + vAlpha @ vY[t-iP:t][::-1]**2  +  vBeta @ vSig2[t-iQ:t][::-1] 
    vLL= -0.5*(np.log(2*np.pi) + np.log(vSig2[0:-1]) + (vY**2)/vSig2[0:-1])
    dALL= np.mean(vLL, axis= 0)

    return -dALL

###########################################################
### dLL= AvgNLnLRegrTr(vPTr, vY, mX)
def AvgNLnLGARCHTr(vGARCHParamsTr, iP, vY):
    """
    Purpose:
        Provide wrapper around AvgNLnLGARCH(), accepting transformed parameters

    Inputs:
        vGARCHParamsTr    transformed parameters, with vGARCHParamsTr=log(vGARCHParams) 
        vY                data

    Return value:
        dNALL             negative average loglikelihood
    """         
    
    vGARCHParams= np.copy(vGARCHParamsTr)
    vGARCHParams= np.exp(vGARCHParamsTr)    
    return AvgNLnLGARCH(vGARCHParams, iP, vY)

###########################################################
### vY= GARCHEstim(vGARCHParams0, iP, vY)
def GARCHEstim(vGARCHParams0, iP, vY):
    """
    Purpose:
        Estimate GARCH(p,q)

    Inputs:
        vGARCHParams0   initial parameters
        vY              data

    Return value:
        vY              realised time series    
    """
    
    iT= len(vY)  
    vGARCHParams0Tr= np.log(vGARCHParams0)
    res= opt.minimize(AvgNLnLGARCHTr, vGARCHParams0Tr, args=(iP, vY), method="BFGS")
    vGARCHParamsStarTr= np.copy(res.x)         
    vGARCHParamsStar= np.exp(vGARCHParamsStarTr)    
    sMess= res.message
    dLL= -iT*res.fun
    print ("\nBFGS results in ", sMess, "\nParameter estimates (MLE): ", vGARCHParamsStar, "\nLog-likelihood= ", dLL, ", f-eval= ", res.nfev)

    return vGARCHParamsStar

###########################################################    
### main
def main():
    # Cleaning
    plt.close('all')
    
    # Magic numbers
    iT= 10000
    sDistrName= 'normal'
    vDistrParams= np.array([0])
    sPlot= 'line'
    iSeed= 1234
    dOmega= np.array([0.01])
    vAlpha= np.array([0.1]) #np.array([0.1]) #np.array([0.05, 0.1])
    vBeta= np.array([0.6])
    vGARCHParams= np.hstack((dOmega, vAlpha, vBeta))
    iP= len(vAlpha)
    dOmega0= 0.1
    vAlpha0= np.array([0.3]) #np.array([0.3, 0.1])
    vBeta0= np.array([0.3])
    vGARCHParams0= np.hstack((dOmega0, vAlpha0, vBeta0))
    
    # Set seed
    np.random.seed(iSeed)
        
    # White noise
    #WhiteNoiseSim(iT,vDistrParams,sDistrName,sPlot)
    
    # GARCH(p,q)
    vY= GARCHSim(iT, vGARCHParams, iP, vDistrParams, sDistrName, sPlot)
    
    GARCHEstim(vGARCHParams0, iP, vY)


###########################################################
### start main
if __name__ == "__main__":
    main()
