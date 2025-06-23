#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose: Generate plots size experiment
"""

## Imports

# System
import os

# Fundamentals
import numpy as np

# Visualisation
import matplotlib.pyplot as plt


###########################################################
def PlotRejRatesSize(dictX, dictRejRates, dictScores, vIdxScores, dictRIdxGridPlots, sSettings=None, iLegCol=3,
                     iStepX=1, lNominalSize=[]):
    """
    Purpose
    ----------
    Generate plots size experiment

    Parameters
    ----------
    dictX :                 dictionary with 'Values' (x-axis values) and 'Label' (LaTeX-formatted label)
    dictRejRates :          dictionary of empirical rejection rates by true model and favored alternative
    dictScores :            dictionary of scoring rules with plotting specifications and labels
    vIdxScores :            array of integers indicating which scoring rules to include
    dictRIdxGridPlots :     dictionary specifying which grid indices to plot per model configuration
    sSettings :             optional string to include in filename when saving figure
    iLegCol :               integer number of columns in legend layout
    iStepX :                integer step size for x-axis tick marks
    lNominalSize :          list of nominal rejection probabilities used for reference lines

    Returns
    ----------
    None : displays and saves plot in Figures subdirectory
    """

    lKeyTrue = list(dictRejRates.keys())
    lKeySides = list(dictRejRates[lKeyTrue[0]].keys())
    iS, iLenRGrid = dictRejRates[lKeyTrue[0]][lKeySides[0]][0].shape
    lScores = list(dictScores.keys())

    vFigSize = (10, 6)
    sFileLoc = os.getcwd()
    sFileLocation = os.path.join(sFileLoc, 'Figures/')

    fig, ax = plt.subplots(1, 1, figsize=vFigSize)

    for k in range(1):
        for d in range(1):
            for s in vIdxScores:
                for p in range(len(dictRejRates[lKeyTrue[d]][lKeySides[k]])):
                    ax.axhline(y=lNominalSize[p], c='#E6E6EA', zorder=0)
                    vIdx = dictRIdxGridPlots[lKeyTrue[d]][lKeySides[k]]
                    if p == 0:
                        ax.plot(dictX['Values'][vIdx], dictRejRates[lKeyTrue[d]][lKeySides[k]][p][s,][vIdx],
                                c=dictScores[lScores[s]]['Colour'], ls=dictScores[lScores[s]]['Symbol'],
                                label=dictScores[lScores[s]]['Label'])

                    else:
                        ax.plot(dictX['Values'][vIdx], dictRejRates[lKeyTrue[d]][lKeySides[k]][p][s,][vIdx],
                                c=dictScores[lScores[s]]['Colour'], ls=dictScores[lScores[s]]['Symbol'])

                    ax.xaxis.set_ticks(np.arange(int(dictX['Values'][vIdx].min().round()),
                                                 int(dictX['Values'][vIdx].max().round() + 1), iStepX))

                    if p == 0:
                        ax.set_xlabel(dictX['Label'])
                        ax.set_ylabel('rejection rate')
                        ax.yaxis.set_ticks(np.arange(0, 0.12, 0.02))
                        ax.set_title('${\\rm H}_1: {\\rm E}_{p_t} (D_{t+1}) > 0$')

    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15 + 0.05 * int(np.ceil(vIdxScores.size / iLegCol)), hspace=0.45)
    fig.legend(handles, labels, loc='lower center', ncol=iLegCol, frameon=True)  # mode='expand'

    plt.show()
    sName = 'PlotRejRatesSize' + sSettings + '.pdf'
    plt.savefig(sFileLocation + sName, bbox_inches='tight')
