import numpy as np
from scipy.stats import norm
from scipy.stats import laplace



def ScoreIndicator(list_of_nums, single_y, left_tail, q):
    if q == 0:
        indicator = np.array([1 if single_y < np.percentile(list_of_nums, left_tail) else 0])
    else:
        indicator = np.array([1 if (
                np.percentile(list_of_nums, 0.5 * q * left_tail) < single_y < np.percentile(list_of_nums,
                                                                                            left_tail - 0.5 * q * left_tail)) else 0])

    return indicator


def pdfCondNormal(y, fcast_var_normal):

    return norm.pdf(y, loc=0, scale=fcast_var_normal)


def pdfCondLaplace(y, fcast_var_laplace):

    return laplace.pdf(y, loc=0, scale=fcast_var_laplace)


def cslScorefunction(weight, vY_subset, forecasts_normal, forecasts_laplace, quantile, q):

    cslScoreNormal = np.zeros(len(vY_subset))
    cslScoreLaplace = np.zeros(len(vY_subset))

    for t in range(0, len(vY_subset)):
        y = vY_subset[t]

        fcast_var_normal = forecasts_normal[t]
        fcast_var_laplace = forecasts_laplace[t]

        cdf_normal = 1 - norm.cdf(np.percentile(vY_subset, quantile), loc=0, scale=fcast_var_normal)
        cdf_laplace = 1 - laplace.cdf(np.percentile(vY_subset, quantile), loc=0, scale=fcast_var_laplace)

        cslScoreNormal[t] = weight * (ScoreIndicator(vY_subset, y, quantile, q) * pdfCondNormal(y, fcast_var_normal)
                                      + (1 - ScoreIndicator(vY_subset, y, quantile, 0)) * cdf_normal)
        cslScoreLaplace[t] = (1 - weight) * (
                ScoreIndicator(vY_subset, y, quantile, q) * pdfCondLaplace(y, fcast_var_laplace)
                + (1 - ScoreIndicator(vY_subset, y, quantile, 0)) * cdf_laplace)

    masked_addition = np.ma.log(np.add(cslScoreLaplace, cslScoreNormal)).filled(0)

    cslScoreTotal = np.sum(masked_addition)

    return -cslScoreTotal


def logScorefunction(weights, vY_subset, forecasts_normal, forecasts_laplace):
    logScoreNormal_q000 = np.zeros(len(vY_subset))
    logScoreNormal_q001 = np.zeros(len(vY_subset))
    logScoreNormal_q005 = np.zeros(len(vY_subset))
    logScoreNormal_q010 = np.zeros(len(vY_subset))
    logScoreNormal_q020 = np.zeros(len(vY_subset))
    logScoreNormal_q050 = np.zeros(len(vY_subset))
    logScoreNormal_q075 = np.zeros(len(vY_subset))

    logScoreLaplace_q000 = np.zeros(len(vY_subset))
    logScoreLaplace_q001 = np.zeros(len(vY_subset))
    logScoreLaplace_q005 = np.zeros(len(vY_subset))
    logScoreLaplace_q010 = np.zeros(len(vY_subset))
    logScoreLaplace_q020 = np.zeros(len(vY_subset))
    logScoreLaplace_q050 = np.zeros(len(vY_subset))
    logScoreLaplace_q075 = np.zeros(len(vY_subset))

    weight_q000 = weights[0]
    weight_q001 = weights[1]
    weight_q005 = weights[2]
    weight_q010 = weights[3]
    weight_q020 = weights[4]
    weight_q050 = weights[5]
    weight_q075 = weights[6]

    for t in range(0, len(vY_subset)):
        y = vY_subset[t]

        fcast_var_normal = forecasts_normal[t]
        fcast_var_laplace = forecasts_laplace[t]

        logScoreNormal_q000[t] = weight_q000[t] * pdfCondNormal(y, fcast_var_normal)
        logScoreNormal_q001[t] = weight_q001[t] * pdfCondNormal(y, fcast_var_normal)
        logScoreNormal_q005[t] = weight_q005[t] * pdfCondNormal(y, fcast_var_normal)
        logScoreNormal_q010[t] = weight_q010[t] * pdfCondNormal(y, fcast_var_normal)
        logScoreNormal_q020[t] = weight_q020[t] * pdfCondNormal(y, fcast_var_normal)
        logScoreNormal_q050[t] = weight_q050[t] * pdfCondNormal(y, fcast_var_normal)
        logScoreNormal_q075[t] = weight_q075[t] * pdfCondNormal(y, fcast_var_normal)

        logScoreLaplace_q000[t] = (1 - weight_q000)[t] * pdfCondLaplace(y, fcast_var_laplace)
        logScoreLaplace_q001[t] = (1 - weight_q001)[t] * pdfCondLaplace(y, fcast_var_laplace)
        logScoreLaplace_q005[t] = (1 - weight_q005)[t] * pdfCondLaplace(y, fcast_var_laplace)
        logScoreLaplace_q010[t] = (1 - weight_q010)[t] * pdfCondLaplace(y, fcast_var_laplace)
        logScoreLaplace_q020[t] = (1 - weight_q020)[t] * pdfCondLaplace(y, fcast_var_laplace)
        logScoreLaplace_q050[t] = (1 - weight_q050)[t] * pdfCondLaplace(y, fcast_var_laplace)
        logScoreLaplace_q075[t] = (1 - weight_q075)[t] * pdfCondLaplace(y, fcast_var_laplace)

    masked_addition_q000 = np.sum(np.ma.log(np.add(logScoreLaplace_q000, logScoreNormal_q000)).filled(0))
    masked_addition_q001 = np.sum(np.ma.log(np.add(logScoreLaplace_q001, logScoreNormal_q001)).filled(0))
    masked_addition_q005 = np.sum(np.ma.log(np.add(logScoreLaplace_q005, logScoreNormal_q005)).filled(0))
    masked_addition_q010 = np.sum(np.ma.log(np.add(logScoreLaplace_q010, logScoreNormal_q010)).filled(0))
    masked_addition_q020 = np.sum(np.ma.log(np.add(logScoreLaplace_q020, logScoreNormal_q020)).filled(0))
    masked_addition_q050 = np.sum(np.ma.log(np.add(logScoreLaplace_q050, logScoreNormal_q050)).filled(0))
    masked_addition_q075 = np.sum(np.ma.log(np.add(logScoreLaplace_q075, logScoreNormal_q075)).filled(0))

    masked_addition = np.array([masked_addition_q000
                                , masked_addition_q001
                                , masked_addition_q005
                                , masked_addition_q010
                                , masked_addition_q020
                                , masked_addition_q050
                                , masked_addition_q075])

    max_score_q = np.amax(masked_addition)
    max_score_q_indicator = np.argmax(masked_addition)

    vQ = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.75]

    print([vQ[max_score_q_indicator], max_score_q])

    return [vQ[max_score_q_indicator], max_score_q]
