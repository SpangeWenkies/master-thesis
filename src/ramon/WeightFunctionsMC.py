#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose: File containing weight functions

"""

# Fundamentals
import numpy as np


###########################################################
def fWRegular(vY, dictParamsW):
    """
    Purpose
    ----------
    Regular weight function

    Parameters
    ----------
    vParamsW :      vector, parameters weight function

    """

    return np.ones((np.array(vY).size, dictParamsW['vR'].size))


###########################################################
def fWIndicatorL(mY, vParamsW, dR):
    """
    Purpose
    ----------
    Indicator weight function, being one for y < r

    Parameters
    ----------
    mY :            matrix, data
    dictParamsW :   dictionary, weight parameters
                        vR :        vector, threshold values
                        vParamsW :  vector, other parameters weight function
    dR :            double, threshold value

    Return
    ----------
    mY shape x len(vR) matrix with weights
    """

    mWbool = mY < dR
    return np.ones(mWbool.shape) * mWbool


###########################################################
def fWIndicatorC(mY, vParamsW, dR):
    """
    Purpose
    ----------
    Indicator weight function, being one for -r < y < r

    Parameters
    ----------
    mY :            matrix, data
    dictParamsW :   dictionary, weight parameters
                        vR :        vector, threshold values
                        vParamsW :  vector, other parameters weight function
    dR :            double, threshold value

    Return
    ----------
    mY shape x len(vR) matrix with weights
    """

    mWbool = (mY > -dR) * (mY < dR)
    return np.ones(mWbool.shape) * mWbool


###########################################################
def fWIndicatorCgen(mY, vParamsW, dR):
    """
    Purpose
    ----------
    Indicator weight function, being one for r - a/2 < y < r + a/2

    Parameters
    ----------
    mY :            matrix, data
    dictParamsW :   dictionary, weight parameters
                        vR :        vector, threshold values
                        vParamsW :  vector, other parameters weight function
    dR :            double, threshold value

    Return
    ----------
    mY shape x len(vR) matrix with weights
    """

    mWbool = (mY > (dR - vParamsW / 2)) * (mY < (dR + vParamsW / 2))
    return np.ones(mWbool.shape) * mWbool





