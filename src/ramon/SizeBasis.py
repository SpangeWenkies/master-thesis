#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose: Helper functions size experiment
"""

###########################################################
### Imports

# Fundamentals
import numpy as np

# Dependencies
from ScoringRulesMC import *


###########################################################
def MPITaskDistributor(iRank, iProc, iTotal, bOrder=True):
    """
    Purpose
    ----------
    Construct vector of indices [integers] for which process with rank iRank should do calculations

    Parameters
    ----------
    iRank :     integer, rank running process
    iProc :     integer, total number of processes in MPI program
    iTotal :    integer, total number of tasks
    bOrder :    boolean, use standard ordering of integers if True

    Output
    ----------
    vInt :      vector, part of total indices 0 to [not incl.] iTotal that must be calculated by process
                    with rank iRank
    """

    # Print error if less than one task per process
    if iTotal / iProc < 1: print('Error: Number of tasks smaller than number of processes')

    # If iTotal/iProc is not an integer, then all processes take on one additional task, except for the last process.
    iCeil = int(np.ceil(iTotal / iProc))
    lInt = []

    # Standard ordering from 0 to iTotal-1
    if bOrder:
        for i in range(iTotal):
            lInt.append(iRank * iCeil + i)
            if len(lInt) == iCeil or iRank * iCeil + i == iTotal - 1:
                break

    # Using iProc steps
    else:
        for i in range(iTotal):
            lInt.append(iRank + iProc * i)
            if iRank + iProc * (i + 1) > iTotal - 1:
                break

    return np.array(lInt)


###########################################################
def ScoringRuleCollectionName(dictScores):
    """
    Purpose
    ----------
    Use short codes to refer to a selection of scoring rules in the file name,
    to prevent for the error of having a too long file name.

    Parameters
    ----------
    dictScores :        dictionary, with scoring rules

    Returns
    ----------
    sSelectedScores :   abbreviated name
    """

    sSelectedScoresFull = ''
    for s in list(dictScores.keys()): sSelectedScoresFull += s

    if sSelectedScoresFull == 'LogSLogSSharpLogSFlatPsSphSPsSphSSharpPsSphSFlatPowSPowSSharpPowSFlatCRPStwCRPSwsCRPSCRPSSharpCRPSFlat':
        sSelectedScores = 'LogSPsSphSPowSCRPSall'
    elif sSelectedScoresFull == 'LogSLogSSharpLogSFlatPsSphSPsSphSSharpPsSphSFlatPowSPowSSharpPowSFlatCRPStwCRPSwsCRPSCRPSSharpCRPSFlatCRPSFlatConstCRPSFlatCentre':
        sSelectedScores = 'LogSPsSphSPowSCRPSallflats'
    elif sSelectedScoresFull == 'LogSLogSSharpLogSFlatPsSphSPsSphSSharpPsSphSFlatPowSPowSSharpPowSFlatCRPStwCRPSwsCRPSCRPSSharpCRPSFlatMinCRPSFlatConstCRPSFlatRandom':
        sSelectedScores = 'LogSPsSphSPowSCRPSallflats'
    else:
        sSelectedScores = sSelectedScoresFull

    return sSelectedScores


###########################################################
def DMCalc(dictCand, mY, fScore, iRidx, dictW, vParamsS, dictPreCalc):
    """
    Purpose
    ----------
    Calculate Diebold Mariano Statistics

    Parameters
    ----------
    dictCand :  dictionary, contains dictionary for F and G distribution:
                    dictF: dictionary, distribution F candidate
                        randDistr:  distribution F, stats object
                        sDistr : name distribution
                    dictG: dictionary, distribution G candidate
                        randDistr:  distribution F, stats object
                        sDistr: name distribution
    mY :            matrix, iT x iRep, observations for which the scoring rule will be calculated
    fScore :        function, selected scoring rule
    dictW :         dictionary,  weight function:
                        fW : selected weighted function
                        dictParamsW : dictionary, parameters of weight function
    vParamsS :      vector, parameter vector scoring rule: alpha parameter
    dictPreCalc :   dictionary, precalculated values

    Returns
    -------
    vDMStat :              vector, DM statistics for each value of r [horizontal axis power plot]
    """

    mDiff = fScore(dictCand['dictF'], mY, iRidx, dictW, vParamsS, dictPreCalc) - fScore(dictCand['dictG'], mY, iRidx,
                                                                                        dictW, vParamsS, dictPreCalc)
    if mDiff.ndim == 1: mDiff = mDiff.reshape((mDiff.size, 1))  # allow for vectors
    iT, iRep = mDiff.shape

    return mDiff.mean(axis=0) / np.sqrt(mDiff.var(axis=0) / iT)


###########################################################
def DMCalcMCMPISize(dictCand, mData, dictScores, dictW, dictPreCalc, iRep, iRank, vInt):
    """
    Purpose:
        Calculate DM statistics for replications with indices in vIntRep [without expected number of observations correction]
        Note: Function is tailored to size study, eliminating some enhancements introduced for the power study

    Parameters:
        dictCand :  dictionary, contains dictionary for F and G distribution:
                        dictF: dictionary, distribution F candidate
                            randDistr:  distribution F, stats object
                            sDistr : name distribution
                        dictG: dictionary, distribution G candidate
                            randDistr:  distribution F, stats object
                            sDistr: name distribution
        mData :         matrix, size 2 x iTmax x iRep
        dictScores :    dictionary, keys are scoring rule names: contains dictionary for each scoring rule with keys
                            fScore: function Scoring rule
                            vParamsS: parameter vector scoring rule
                            bWeighted: boolean, weighted scoring rule logical
        dictW :         dictionary,  weight function:
                            fW : selected weighted function
                            dictParamsW : dictionary, parameters of weight function
        dictPreCalc :   dictionary, precalculated values
        iC :            integer, expected number of observations {w>0}
        vIntR :         vector, selected indices of threshold grid for process with rank iRank

    Returns:
        mDMCalc :       matrix, size 2 x iS x iRep x len(vIntR)  where the `2' refers to rejection in favour of f or g

    """

    lScoresNames = list(dictScores.keys())
    iS = len(lScoresNames)  # number of selected scoring rules
    iInt = vInt.size
    mDMCalc = np.zeros((1, iS, iRep, iInt))

    for i in range(iInt):
        if iRank == 0: print(round(i / iInt * 100), '% completed', sep='')  # status printed on first node
        for d in range(1):
            for s in range(iS):
                mDMCalc[d, s, :, i] = DMCalc(dictCand, mData, dictScores[lScoresNames[s]]['fScore'], vInt[i], dictW,
                                             dictScores[lScoresNames[s]]['vParamsS'], dictPreCalc)

    return mDMCalc


###########################################################
def RejRates(mDMCalc, dictCritVal):
    """
    Purpose
    ----------
    Calculate Rejection Rates

    Parameters
    ----------
    mDMCalc :            matrix, size iF x iS x iRep x iLenPiGrid
    dCriticalValue :     dictionary, keys:
                                FavF:  H_1: E d > 0 [rej in favour of F]
                                FavG:  H_1 E d < 0 [rej in favour of G]
    Returns
    ----------
    dictRejRates :       dictionary, with per keySide
                               mRejRates : matrix, size iF x iS x iLenPiGrid
    """

    iTrues, iS, iRep, iLenRGrid = mDMCalc.shape
    if iTrues == 2:  # power study case
        dictRejRates = {'TrueF': {'FavF': None, 'FavG': None}, 'TrueG': {'FavF': None, 'FavG': None}}
    else:
        dictRejRates = {'TrueP': {'FavF': None, 'FavG': None}}
    lTrueKeys = list(dictRejRates.keys())

    for keySide in list(dictCritVal.keys()):
        for i in range(len(lTrueKeys)):
            if keySide == 'FavF':
                dictRejRates[lTrueKeys[i]][keySide] = []
                for k in range(len(dictCritVal[keySide])):
                    dictRejRates[lTrueKeys[i]][keySide].append(
                        np.mean(mDMCalc[i, :, :, :] > dictCritVal[keySide][k], axis=1))
            elif keySide == 'FavG':
                dictRejRates[lTrueKeys[i]][keySide] = []
                for k in range(len(dictCritVal[keySide])):
                    dictRejRates[lTrueKeys[i]][keySide].append(
                        np.mean(mDMCalc[i, :, :, :] < dictCritVal[keySide][k], axis=1))

    return dictRejRates
