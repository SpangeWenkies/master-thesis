#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose: Mainfile size experiment
"""

###########################################################
### Imports

# Fundamentals
import numpy as np
from scipy import stats  # pre-programmed random variables

# MPI
from mpi4py import MPI

# System
import os

# Plots
import matplotlib.pyplot as plt
from matplotlib import rc

os.environ["PATH"] += os.pathsep + "/Library/TeX/texbin"
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': '15'})
rc('text', usetex=True)
# %matplotlib qt

# Dependencies
from SizeBasis import *
from ScoringRulesMC import *
from WeightFunctionsMC import *
from SizePlots import *


###########################################################
### main
def main():
    ########################################
    ### Magic numbers
    ########################################

    ## Goal
    iGoal = 2  # 1: simulate data and calculate DM statistics, 2: plotting

    ## Fundamentals
    iRep = 10000  # number of Monte Carlo replications
    iT = 500  # sample size
    iSeed = 4132  # random seed
    bOrderMPI = False  # order task distribution [sequentially if true]
    sDensClass = 'NormSize'  # 'StudentNorm' #'Laplace'  # StudentNorm or Laplace

    ## Candidate distributions
    dictDGP = {}

    # Normal size: size study advocated by Diks et al. (2011)
    if sDensClass == 'NormSize':
        dMuF = -0.2
        dMuG = 0.2
        dMuP = 0
        dictF = {'randDistr': stats.norm(dMuF, 1), 'sDistr': 'NormalL'}
        dictG = {'randDistr': stats.norm(dMuG, 1), 'sDistr': 'NormalR'}
        dictP = {'randDistr': stats.norm(dMuP, 1), 'sDistr': 'Normal'}
        dictDGP = {'dictP': dictP}

    dictCand = {'dictF': dictF, 'dictG': dictG}

    ## vRGrid
    iRmin = 0.1  # start iR grid
    iRmax = 5  # end iR grid
    iRtot = 288  # step size
    vR = np.linspace(iRmin, iRmax, iRtot)

    ## Rejection rates DM statistics
    lNominalSize = [0.01, 0.05, 0.10]
    dictCritVal = {'FavG': [stats.norm.ppf(dNominalSize) for dNominalSize in lNominalSize],
                   'FavF': [stats.norm.ppf(1 - dNominalSize) for dNominalSize in lNominalSize]}  # H_1 E d < (>) 0

    ## Plots
    lStyles = ['-', '--', ':', '-.']
    lCycle = plt.rcParams['axes.prop_cycle'].by_key()['color']  # blue, orange, green, red, brown, purple

    ## Scoring rules
    dictCRPSNumInt = {'dMinInf': -20, 'dPlusInf': 20, 'dDeltaZ': 0.001, 'iSimE': 1000, 'iSeed': 2314,
                      'CensDistParams': 0, 'iMatrixSizeMax': 5}
    vAlpha = np.array([2])
    dAlpha = 2  # selected alpha norm

    dictScores = {
        # 'LogSSharp': {'fScore': LogSSharp,'vParamsS': None, 'bWeighted': True, 'Label': 'LogS$^\\sharp$', 'Colour': lCycle[0], 'Symbol': lStyles[0]},
        'LogSFlat': {'fScore': LogSFlat, 'vParamsS': None, 'bWeighted': True, 'Label': 'LogS$^\\flat$',
                     'Colour': lCycle[0], 'Symbol': lStyles[0]},
        # 'LogSsbar': {'fScore': LogSsbar,'vParamsS': None, 'bWeighted': True, 'Label': 'LogSsbar', 'Colour': lCycle[2], 'Symbol': lStyles[0]},
        # 'LogSslog': {'fScore': LogSslog,'vParamsS': None, 'bWeighted': True, 'Label': 'LogSslog', 'Colour': lCycle[3], 'Symbol': lStyles[0]},
        # 'PsSphSSharp': {'fScore': PsSphSSharp,'vParamsS': dAlpha, 'bWeighted': True, 'Label': 'SphS$^\\sharp$', 'Colour': lCycle[0], 'Symbol': lStyles[0]},
        'PsSphSFlat': {'fScore': PsSphSFlat, 'vParamsS': dAlpha, 'bWeighted': True, 'Label': 'SphS$^\\flat$',
                       'Colour': lCycle[1], 'Symbol': lStyles[0]},
        # 'PsSphSsbar': {'fScore': PsSphSsbar,'vParamsS': dAlpha, 'bWeighted': True, 'Label': 'SphSsbar', 'Colour': lCycle[2], 'Symbol': lStyles[0]},
        # 'PsSphSslog': {'fScore': PsSphSslog,'vParamsS': dAlpha, 'bWeighted': True, 'Label': 'SphSslog', 'Colour': lCycle[3], 'Symbol': lStyles[0]},
        # 'PowSSharp': {'fScore': PowSSharp,'vParamsS': dAlpha, 'bWeighted': True, 'Label': 'QS$^\\sharp$', 'Colour': lCycle[0], 'Symbol': lStyles[0]},
        # 'PowSFlat': {'fScore': PowSFlat,'vParamsS': dAlpha, 'bWeighted': True, 'Label': 'QS$^\\flat$', 'Colour': lCycle[2], 'Symbol': lStyles[0]},
        # 'PowSsbar': {'fScore': PowSsbar,'vParamsS': dAlpha, 'bWeighted': True, 'Label': 'QSsbar', 'Colour': lCycle[2], 'Symbol': lStyles[0]},
        # 'PowSslog': {'fScore': PowSslog,'vParamsS': dAlpha, 'bWeighted': True, 'Label': 'QSslog', 'Colour': lCycle[3], 'Symbol': lStyles[0]},
        'CRPSSharp': {'fScore': CRPSSharp, 'vParamsS': dictCRPSNumInt, 'bWeighted': True, 'Label': 'CRPS$^\\sharp$',
                      'Colour': lCycle[3], 'Symbol': lStyles[0]},
        'CRPSFlat': {'fScore': CRPSGenFlatGamma, 'vParamsS': dictCRPSNumInt, 'bWeighted': True,
                     'Label': 'CRPS$^\\flat$', 'Colour': lCycle[2], 'Symbol': lStyles[0]},
        # 'CRPSsbar':{'fScore': CRPSsbar,'vParamsS': dictCRPSNumInt, 'bWeighted': True, 'Label': 'CRPSsbar', 'Colour': lCycle[6], 'Symbol': lStyles[0]},
        # 'CRPSslog':{'fScore': CRPSslog,'vParamsS': dictCRPSNumInt, 'bWeighted': True, 'Label': 'CRPSslog', 'Colour': lCycle[3], 'Symbol': lStyles[0]}, #equiv. has also been verified numerically
        'twCRPS': {'fScore': twCRPS, 'vParamsS': dictCRPSNumInt, 'bWeighted': True, 'Label': 'twCRPS',
                   'Colour': lCycle[4], 'Symbol': lStyles[0]},
    }

    ## Selected weight function
    dictW = {'fW': fWIndicatorC, 'vR': vR, 'vParamsW': 2}  # center indicator
    sWeightFunction = dictW['fW'].__name__

    ## Save settings
    sCurrentFileName = 'SizeMain'

    sSelectedScores = ScoringRuleCollectionName(dictScores)
    sSettingsHorizontalGrid = '_iRmin' + str(int(abs(iRmin * 100))) + '_iRmax' + str(
        int(abs(iRmax * 100))) + '_iRtot' + str(int(iRtot))
    sSettings = 'CandF_' + dictCand['dictF']['sDistr'] + '_CandG_' + dictCand['dictG'][
        'sDistr'] + sSettingsHorizontalGrid + '_fW_' + sWeightFunction + '_iT' + str(iT) + '_iRep' + str(
        iRep) + '_' + sCurrentFileName

    ########################################
    ### Calculate or load DM statistics and Rejection rates
    ########################################
    if iGoal == 1:

        ## Load data and IS weights
        iRorCSize = int(vR.size)

        ## Initialisation MPI device
        comm = MPI.COMM_WORLD
        iRank = comm.Get_rank()  # rank process
        iProc = comm.Get_size()  # total number of processes

        if iRank == 0:
            np.random.seed(iSeed)
            mData = dictDGP['dictP']['randDistr'].rvs((iT, iRep))
        else:
            mData = None

        mData = comm.bcast(mData, root=0)

        comm.barrier()

        # Distribute
        vInt = MPITaskDistributor(iRank, iProc, iRorCSize,
                                  bOrder=bOrderMPI)  # calculate indices simulations assigned to this process
        vIntR = vInt

        # Precalculate norms and Fbar for vR grid and vAlpha
        dictPreCalc = {'AlphaNormsF': {dictF['sDistr']: AlphaNormsF(dictF, dictW, vAlpha),
                                       dictG['sDistr']: AlphaNormsF(dictG, dictW, vAlpha)},
                       'AlphaNormsFw': {dictF['sDistr']: AlphaNormsFw(dictF, dictW, vAlpha, vIntR),
                                        dictG['sDistr']: AlphaNormsFw(dictG, dictW, vAlpha, vIntR)},
                       'DistrBar': {dictF['sDistr']: DistrBar(dictF, dictW, vIntR),
                                    dictG['sDistr']: DistrBar(dictG, dictW, vIntR)}}

        comm.barrier()  # wait for everyone
        if iRank == 0: print('Pre-calculations completed')

        # DM calculations
        mDMCalc = np.zeros((1, len(list(dictScores.keys())), iRep, iRorCSize))
        mDMCalcProc = DMCalcMCMPISize(dictCand, mData, dictScores, dictW, dictPreCalc, iRep, iRank, vInt)
        lDMCalc = comm.gather(mDMCalcProc, root=0)
        comm.barrier()  # wait for everyone

        ## Combine data on root
        if iRank == 0:
            iCount = 0
            for rank in range(len(lDMCalc)):
                vIntRank = MPITaskDistributor(rank, iProc, iRorCSize, bOrder=bOrderMPI)
                for i in range(vIntRank.size):
                    mDMCalc[:, :, :, vIntRank[i]] = lDMCalc[rank][:, :, :, i]

            print(mDMCalc[0, 0, 1:10, 0])

            np.save('mDMCalc/mDMCalc_' + sSettings + '.npy', mDMCalc)

    ########################################
    ### Generate plots
    ########################################
    if iGoal == 2:
        ## Cleaning
        plt.close('all')

        ## Load DM statistics and calculate rejection rates
        mDMCalc = np.load('mDMCalc/mDMCalc_' + sSettings + '.npy')

        dictRejRates = RejRates(mDMCalc, dictCritVal)
        dictRIdxGridPlots = {'TrueP': {'FavF': np.arange(0, iRtot)[vR >= -5][vR[vR >= -5] <= 3],
                                       'FavG': np.arange(0, iRtot)[vR > -5][vR[vR > -5] < 3]},
                             'TrueG': {'FavF': np.arange(0, iRtot)[vR > -5][vR[vR > -5] < 5],
                                       'FavG': np.arange(0, iRtot)[vR > -5][vR[vR > -5] < 5]}}

        # All without coinciding ones
        vIdxSelectedScores = np.arange(0, len(dictScores.keys()))

        PlotRejRatesSize({'Values': vR, 'Label': 'threshold $r$'}, dictRejRates, dictScores, vIdxSelectedScores,
                         dictRIdxGridPlots, sSettings + '_All', iLegCol=3, iStepX=1, lNominalSize=lNominalSize)


###########################################################
### start main
if __name__ == "__main__":
    main()
