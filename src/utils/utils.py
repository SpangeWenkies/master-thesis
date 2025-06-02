import concurrent

import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from scipy.stats import gaussian_kde
from scipy.stats import t as student_t
from scipy.stats import multivariate_t
from scipy.stats import ttest_1samp
from tqdm import tqdm
import scipy.optimize as opt
from concurrent.futures import ProcessPoolExecutor


def WhiteNoiseSim(iT, vDistrParams, sDistrName):
    """
    Purpose:
        Generate white noise

    Inputs:
        iT              sample size
        vDistrParams    distributional params
        sDistrName      distribution name

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

        return vEps
    else:
        sys.exit(
            "Distribution not supported.")  # print("Error: Distribution not supported. Supported distributions are:"); print(*lDistrName, sep = ", ");


###########################################################


def GARCHSim(iT, vGARCHParams, iP, vDistrParams, sDistrName):
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
    dOmega = vGARCHParams[0]
    vAlpha = vGARCHParams[1:iP + 1]
    vBeta = vGARCHParams[iP + 1:]
    iQ = len(vBeta)
    iR = max(iP, iQ)
    vEps = WhiteNoiseSim(iT + iR, vDistrParams, sDistrName, False)
    vSig2 = np.zeros(iT + iR)
    dSig20 = dOmega / (1 - np.sum(vAlpha) - np.sum(vBeta))
    vSig2[0:iR] = np.repeat(dSig20, iR)
    vY = np.zeros(iT + iR)
    dY0 = 0
    vY[0:iR] = np.repeat(dY0, iR)
    for t in range(iR, iT):
        vSig2[t] = dOmega + vAlpha @ vY[t - iP:t][::-1] ** 2 + vBeta @ vSig2[t - iQ:t][::-1]
        vY[t] = vEps[t] * np.sqrt(vSig2[t])
    vY = vY[iR:]
    return vY

###########################################################

def student_t_copula_pdf_from_PITs(u, rho, df, oracle=True):
    """
        Calculate student t copula pdf from given PITs and correlation coefficient rho

        Inputs:
            u :     np.ndarray
                        An (n, 2) array of PITs, each in (0,1)
            rho :   Correlation parameter [-1,1]
            df :    degrees of freedom t distribution
        Outputs:
            student-t copula pdf
    """
    if oracle:
        # Convert uniform PITs to quantiles of t-distribution (known is they come from t-dist)
        x = student_t.ppf(u[:, 0], df)
        y = student_t.ppf(u[:, 1], df)
    else:
        # Convert uniform PITs using ECDF (original distribution is unknown)
        # We observe the true residual series x and y from the copula + specified marginals (or just innovations)
        x = student_t.ppf(u[:, 0], df)
        y = student_t.ppf(u[:, 1], df)
        # We then PIT these x and y into \hat{u_1} and \hat{u_2}

        ...

    # Build covariance matrix
    cov = np.array([[1, rho], [rho, 1]])

    # Multivariate t PDF
    mv_pdf = multivariate_t.pdf(np.stack([x, y], axis=-1), df=df, shape=cov)

    # Marginal t PDFs
    denom = student_t.pdf(x, df) * student_t.pdf(y, df)

    return mv_pdf / denom


def LogS_student_t_copula(u, rho, df):
    """
    Purpose:
        Regular Logarithmic scoring rule on student t copula pdf
    Inputs:
            u :     np.ndarray
                        An (n, 2) array of PITs, each in (0,1)
            rho :   Correlation parameter [-1,1]
            df :    degrees of freedom t distribution

    Return value:
        sum(np.log(mF))     sum of log of all mF matrix points

    Output:
        iT x iRep matrix with calculated log scores
    """

    mF = student_t_copula_pdf_from_PITs(u, rho, df)
    mF[mF == 0] = 1e-100  # avoid numerical zeros
    return sum(np.log(mF))

def CS_student_t_copula(u, rho, df, w):
    """
    Purpose:
        Censored Logarithmic scoring rule on student t copula pdf
    Inputs:
            u :     np.ndarray
                        An (n, 2) array of PITs, each in (0,1)
            rho :   Correlation parameter [-1,1]
            df :    degrees of freedom t distribution
            w :     (n,) array of weights (binary or smooth)

    Return value:
        w * log_mF + (1 - w) * log_Fw_bar  weighted log of mF * w plus inverse weighted log of Fw_bar

    Output:
        scalar of calculated censored log scores
    """

    mF = student_t_copula_pdf_from_PITs(u, rho, df)
    mF[mF == 0] = 1e-100  # avoid numerical zeros
    log_mF = np.log(mF)

    # Censored score uses region average as a constant reference
    Fw_bar = np.sum(mF * w) / np.sum(w)
    Fw_bar = max(Fw_bar, 1e-100)
    log_Fw_bar = np.log(Fw_bar)
    return np.sum(w * log_mF + (1 - w) * log_Fw_bar)

def CLS_student_t_copula(u, rho, df, w):
    """
    Purpose:
        Conditional Logarithmic scoring rule on student t copula pdf
    Inputs:
            u :     np.ndarray
                        An (n, 2) array of PITs, each in (0,1)
            rho :   Correlation parameter [-1,1]
            df :    degrees of freedom t distribution
            w :     (n,) array of weights (binary or smooth)

    Return value:
        ...

    Output:
        scalar of calculated conditional log scores
    """

    mF = student_t_copula_pdf_from_PITs(u, rho, df)
    mF[mF == 0] = 1e-100
    # Estimate mass outside region: bar(Fw) = P(y not in A)
    F_total = np.sum(mF)
    F_outside = np.sum(mF * (1 - w))
    F_outside = min(F_outside, F_total - 1e-100)  # avoid division by 0 or log(0)

    # log(1 - mass_inside) = log(mass_outside / total)
    log_1_minus_Fw = np.log(F_outside / F_total + 1e-100)

    return np.sum(w * (np.log(mF) - log_1_minus_Fw))


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

    dOmega = vGARCHParams[0]
    vAlpha = vGARCHParams[1:iP + 1]
    vBeta = vGARCHParams[iP + 1:]
    iT = len(vY)
    iQ = len(vBeta)
    iR = max(iP, iQ)
    vSig2 = np.zeros(iT + 1)
    dSig20 = dOmega / (1 - np.sum(vAlpha) - np.sum(vBeta))
    vSig2[0:iR] = np.repeat(dSig20, iR)
    for t in range(iR, iT + 1):
        vSig2[t] = dOmega + vAlpha @ vY[t - iP:t][::-1] ** 2 + vBeta @ vSig2[t - iQ:t][::-1]
    vLL = -0.5 * (np.log(2 * np.pi) + np.log(vSig2[0:-1]) + (vY ** 2) / vSig2[0:-1])
    dALL = np.mean(vLL, axis=0)

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

    vGARCHParams = np.copy(vGARCHParamsTr)
    vGARCHParams = np.exp(vGARCHParamsTr)
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

    iT = len(vY)
    vGARCHParams0Tr = np.log(vGARCHParams0)
    res = opt.minimize(AvgNLnLGARCHTr, vGARCHParams0Tr, args=(iP, vY), method="BFGS")
    vGARCHParamsStarTr = np.copy(res.x)
    vGARCHParamsStar = np.exp(vGARCHParamsStarTr)
    sMess = res.message
    dLL = -iT * res.fun
    print("\nBFGS results in ", sMess, "\nParameter estimates (MLE): ", vGARCHParamsStar, "\nLog-likelihood= ", dLL,
          ", f-eval= ", res.nfev)

    return vGARCHParamsStar


###########################################################


def resid_to_unif_PIT_ECDF(residuals):
    """
        Purpose:
            Converts residuals to uniform margins implicitly using empirical CDF
            u_t = F_hat(resid_t) = rank(resid_t) / (n+1)
            Here (n+1) to correct for bias

        Inputs:
            residuals       input series (standardized)

        Output:
            Residuals series mapped to [0,1]

    """
    ranks = rankdata(residuals, method="average")
    return ranks / (len(residuals) + 1)


###########################################################


def R_bb7(u1, u2, theta, delta, resid1, resid2, verbose=True):
    """
        Purpose:
            Uses residuals series their sizes to create bivariate bb7 PITs

        Inputs:
            u1, u2:             input series of uniforms from PIT transformation of marginal residuals
            theta:              bb7 dependence parameter (>1)
            delta:              bb7 tail asymmetry parameter (>1)
            verbose (boolean):  Plot or not
            resid1, resid2:     residual series

        Output:
            Two bivariate PIT series on [0,1] respecting the given bb7 dependence
            The copula CDF based on the AR-GARCH marginals their PITs

    """

    numpy2ri.activate()

    # Inject the u1, u2, theta, delta variables into R
    ro.globalenv['u1'] = u1
    ro.globalenv['u2'] = u2
    ro.globalenv['theta'] = theta
    ro.globalenv['delta'] = delta

    # R code: set BB7 copula and simulate new observations (we can also fit copula params)
    ro.r('''
    library(VineCopula)
    set.seed(12085278)

    # Combine uniform data
    u_data <- cbind(u1, u2)

    # Define BB7 copula with initial parameters
    cop_model <- BiCop(family = 17, par = theta, par2 = delta)

    # Optional: evaluate copula CDF on data
    cop_cdf_values <- BiCopCDF(u_data[,1], u_data[,2], cop_model)

    # Simulate new data from copula
    simulated_uv <- BiCopSim(N = nrow(u_data), obj=cop_model)

    # Export to Python
    sim_u1 <- simulated_uv[,1]
    sim_u2 <- simulated_uv[,2]
    cop_cdf <- cop_cdf_values

    # Evaluate true PDF over a grid
    u_seq <- seq(0.005, 0.995, length.out = 300)
    grid <- expand.grid(u_seq, u_seq)
    true_pdf <- BiCopPDF(grid[,1], grid[,2], cop_model)
    ''')
    # Retrieve simulated values and optional copula CDF evaluations
    sim_u1 = np.array(ro.r('sim_u1'))
    sim_u2 = np.array(ro.r('sim_u2'))
    copula_cdf_values = np.array(ro.r('cop_cdf'))  # Optional
    true_pdf = np.array(ro.r('true_pdf')).reshape((300, 300))
    u_seq = np.linspace(0.005, 0.995, 300)
    U1, U2 = np.meshgrid(u_seq, u_seq)

    # Estimate empirical density using KDE
    uv = np.vstack([sim_u1, sim_u2])
    kde = gaussian_kde(uv)
    empirical_pdf = kde(np.vstack([U1.ravel(), U2.ravel()])).reshape(U1.shape)

    true_pdf_clipped = np.clip(true_pdf, 0, 8)


    if verbose:
        # Plot empirical KDE
        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.plot_surface(U1, U2, empirical_pdf, cmap='viridis', linewidth=0)
        ax1.set_title("Empirical Copula Density (KDE)")
        ax1.set_xlabel("u1")
        ax1.set_ylabel("u2")
        ax1.set_zlabel("Density")

        # Plot theoretical BB7 PDF
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.plot_surface(U1, U2, true_pdf_clipped, cmap='plasma', linewidth=0)
        ax2.set_title("Theoretical BB7 Copula PDF (clipped at 8)")
        ax2.set_xlabel("u1")
        ax2.set_ylabel("u2")
        ax2.set_zlabel("Density")

        plt.tight_layout()
        plt.show()

    # Use sim_resid1 and sim_resid2 as marginals
    F1_inv = lambda u: np.quantile(resid1, u)
    F2_inv = lambda u: np.quantile(resid2, u)

    # Define restriction threshold
    restr = resid1 + resid2
    q_05 = np.quantile(restr, 0.05)

    # Define region W(u1, u2) = 1 if y1 + y2 <= q_0.05
    W = np.zeros_like(U1)
    for i in range(U1.shape[0]):
        for j in range(U1.shape[1]):
            y1_ = F1_inv(U1[i, j])
            y2_ = F2_inv(U2[i, j])
            W[i, j] = 1.0 if (y1_ + y2_) <= q_05 else 0.0

    # 2D contour plot
    if verbose:
        plt.figure(figsize=(7, 6))
        contour = plt.contourf(U1, U2, empirical_pdf, levels=30, cmap="viridis")
        plt.contour(U1, U2, W, levels=[0.5], colors='red', linewidths=2)
        plt.title("Empirical Copula Density with Region y1 + y2 ≤ q₀.₀₅")
        plt.xlabel("u1 (PIT)")
        plt.ylabel("u2 (PIT)")
        plt.colorbar(contour, label="Density")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return sim_u1, sim_u2, copula_cdf_values, W, empirical_pdf, true_pdf


###########################################################


def simulate_joint_t_marginals(n, df, theta, delta, verbose=True):
    """
    Simulate joint sample from a BB7 copula and Student-t marginals.

    Parameters:
        n (int): Number of samples
        df (float): Degrees of freedom for the Student-t marginals
        theta (float): BB7 copula parameter (dependence strength)
        delta (float): BB7 copula parameter (tail asymmetry)
        verbose (boolean): Plot or not

    Returns:
        y1, y2: Simulated marginal series
        u1, u2: Copula PIT values
        q_true: True 5% quantile of y1 + y2
    """

    numpy2ri.activate()

    # Step 1: Simulate PITs from BB7 copula via R
    ro.globalenv['n'] = n
    ro.globalenv['theta'] = theta
    ro.globalenv['delta'] = delta

    ro.r('''
    library(VineCopula)
    set.seed(123)

    # Simulate PITs
    cop_model <- BiCop(family = 17, par = theta, par2 = delta)
    u_sim <- BiCopSim(N = n, obj = cop_model)
    u1 <- u_sim[, 1]
    u2 <- u_sim[, 2]

    # Create evaluation grid
    grid_size <- 300
    u_seq <- seq(0.005, 0.995, length.out = grid_size)
    grid <- expand.grid(u_seq, u_seq)

    # Evaluate PDF on grid
    pdf_vals <- BiCopPDF(grid[,1], grid[,2], cop_model)
    ''')

    u1 = np.array(ro.r('u1'))
    u2 = np.array(ro.r('u2'))

    pdf_vals = np.array(ro.r('pdf_vals'))
    u_seq = np.linspace(0.005, 0.995, 300)
    U1, U2 = np.meshgrid(u_seq, u_seq)
    copula_pdf = pdf_vals.reshape(U1.shape)

    # Step 2: Apply Student-t inverse CDF to get marginals
    y1 = student_t.ppf(u1, df)
    y2 = student_t.ppf(u2, df)

    # Step 3: Compute quantile of portfolio sum
    q_true = np.quantile(y1 + y2, 0.05)

    # Step 4: Compute mask / region
    Y1 = student_t.ppf(U1, df)
    Y2 = student_t.ppf(U2, df)
    W_true = ((Y1 + Y2) <= q_true).astype(int)

    # --- Step 4: Plot ---
    if verbose:
        plt.figure(figsize=(8, 6))
        contour = plt.contourf(U1, U2, np.clip(copula_pdf, 0, 8), levels=30, cmap="viridis")
        plt.contour(U1, U2, W_true, levels=[0.5], colors="red", linewidths=2)
        plt.title("BB7 Copula PDF with True Region Overlay")
        plt.xlabel("u1")
        plt.ylabel("u2")
        plt.colorbar(contour, label="Copula Density")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return y1, y2, u1, u2, q_true, U1, U2, W_true, copula_pdf


###########################################################


def inverse_ecdf(u, original_resid):
    """
        Purpose:
            Inverts the PITs back to residuals using inverse ECDF

        Inputs:
            u                   input series of PITs
            original_resid      original specified marginal residuals

        Output:
            specified marginal style residuals respecting copula dependence

    """
    sorted_resid = np.sort(original_resid)
    n = len(sorted_resid)
    indices = np.minimum((u * n).astype(int), n - 1)
    return sorted_resid[indices]


###########################################################


def simulate_GARCH(n, omega, alpha, beta, dist, df=np.array([5])):
    """
    Purpose:
        Simulate two series from same GARCH process

    Inputs:
        n               series length
        omega           long term variance level
        alpha           impact of new shocks (ARCH term)
        beta            persistence of new volatility (GARCH term)
        dist            boolean: either "normal" or "t"
        nu              degrees of freedom when dist="t"

    Output:
        Two residuals series and their two variance series

    """
    # Initialize the parameters
    if dist == "normal":
        white_noise1 = np.random.normal(size=n)
        white_noise2 = np.random.normal(size=n)
    elif dist == "t":
        white_noise1 = np.random.standard_t(df, size=n)
        white_noise2 = np.random.standard_t(df, size=n)
    resid1 = np.zeros_like(white_noise1)
    variance1 = np.zeros_like(white_noise1)
    resid2 = np.zeros_like(white_noise2)
    variance2 = np.zeros_like(white_noise2)

    for t in range(1, n):
        # Simulate the variance (sigma squared)
        variance1[t] = omega + alpha * resid1[t - 1] ** 2 + beta * variance1[t - 1]
        variance2[t] = omega + alpha * resid2[t - 1] ** 2 + beta * variance2[t - 1]
        # Simulate the residuals
        resid1[t] = np.sqrt(variance1[t]) * white_noise1[t]
        resid2[t] = np.sqrt(variance2[t]) * white_noise2[t]

    return resid1, variance1, white_noise1, resid2, variance2, white_noise2


###########################################################


def evaluate_mass_in_region(density, W):
    """
    Purpose:
        Compute the proportion of copula density (empirical or theoretical)
        that falls within a PIT-space restriction region.

    Inputs:
        density     : 2D array of copula density values
        W           : 2D binary mask (1 = in region, 0 = out)

    Output:
        Proportion of total density mass that lies inside the masked region
    """
    weighted_mass = np.sum(density * W)
    total_mass = np.sum(density)
    frac = weighted_mass / total_mass if total_mass != 0 else np.nan
    return frac


###########################################################
def compute_scores_over_region(density, W, eps=1e-12):
    """
    Compute localized scoring rules (CS and CLS) over a given region mask.

    Inputs:
        density : 2D array of copula PDF values
        W       : 2D binary mask (1 = in region, 0 = out)
        eps     : small number to prevent log(0)

    Outputs:
        CSL : 2D array of censored log-score values (outside = 0)
        SCL : 2D array of normalized conditional log-scores
    """
    density = density + eps
    log_density = np.log(density)
    CS = np.sum(log_density * density * W)

    # Total probability mass in the region
    p_A = np.sum(density * W)

    # Conditional density inside region
    p_cond = (density * W) / p_A

    # SCL = expectation under conditional
    CLS = np.sum(log_density * p_cond)

    return CS, CLS


###########################################################
def compute_ecdf_inverse(sample):
    """
        Return ECDF-based inverse function using quantile interpolation.

        Inputs:
            sample  : standardized sample of which to use the ECDF

        Outputs:
            ECDF as a function of the PITs
    """
    return lambda u: np.quantile(sample, u)


###########################################################
def compute_true_t_inverse(df):
    """
        Return student-t CDF inverse function

        Inputs:
            df  : degrees of freedom

        Outputs:
            Inverse student-t CDF as function of PITs
    """
    return lambda u: student_t.ppf(u, df)

def simulate_independent_t_copula(n, df):
    """
        Simulate PITs of product copula with student-t marginals

        Inputs:
            n   : sample size
            df  : degrees of freedom

        Outputs:
            PITs
    """
    eps1 = student_t.rvs(df, size=n)
    eps2 = student_t.rvs(df, size=n)
    u1 = student_t.cdf(eps1, df)
    u2 = student_t.cdf(eps2, df)
    return u1, u2

###########################################################
def create_region(U1, U2, F1_inv, F2_inv, q_alpha):
    """
        Create a binary mask for the region where F1_inv(U1) + F2_inv(U2) <= q_alpha.

        Inputs:
            U1, U2          : Meshgrid same shape as copula_pdf
            F1_inv, F2_inv  : inverse of either analytical CDF or emperical CDF
            q_alpha         : the threshold quantile of the restriction (in this case Y1+Y2)
        Outputs:
            ECDF as a function of the PITs
    """
    Y1 = F1_inv(U1)
    Y2 = F2_inv(U2)
    W = ((Y1 + Y2) <= q_alpha).astype(int)
    return W


###########################################################
def copula_pdf_student_t(U1, U2, rho, df):
    """
        Create a binary mask for the region where F1_inv(U1) + F2_inv(U2) <= q_alpha.

        Inputs:
            U1, U2  : Meshgrid same shape as copula_pdf
            rho     : Correlation parameter [-1,1]
            df      : the threshold quantile of the restriction (in this case Y1+Y2)
        Outputs:
            copola pdf of a student-t copula
    """


    # Inverse transform
    x = student_t.ppf(U1, df)
    y = student_t.ppf(U2, df)

    # Build covariance matrix
    cov = np.array([[1, rho], [rho, 1]])

    # Multivariate t PDF
    mv_pdf = multivariate_t.pdf(np.stack([x, y], axis=-1), df=df, shape=cov)

    # Marginal t PDFs
    denom = student_t.pdf(x, df) * student_t.pdf(y, df)

    return mv_pdf / denom


def compute_score_differences(pdf_f, pdf_g, W_ecdf, W_oracle):
    """
    Compute score differences between two copulas f and g over ECDF and oracle regions.

    Parameters:
        pdf_f     : 2D numpy array, density of copula f on the grid
        pdf_g     : 2D numpy array, density of copula g on the grid
        W_ecdf    : 2D numpy binary mask for ECDF-based region
        W_oracle  : 2D numpy binary mask for Oracle region

    Returns:
        A dictionary with CLS and CS differences (f - g) for both regions
    """
    CS_f_oracle, CLS_f_oracle = compute_scores_over_region(pdf_f, W_oracle)
    CS_g_oracle, CLS_g_oracle = compute_scores_over_region(pdf_g, W_oracle)
    CS_f_ecdf, CLS_f_ecdf = compute_scores_over_region(pdf_f, W_ecdf)
    CS_g_ecdf, CLS_g_ecdf = compute_scores_over_region(pdf_g, W_ecdf)

    results = {
        "CLS_diff_ecdf": CLS_f_ecdf - CLS_g_ecdf,
        "CLS_diff_oracle": CLS_f_oracle - CLS_g_oracle,
        "CS_diff_ecdf": CS_f_ecdf - CS_g_ecdf,
        "CS_diff_oracle": CS_f_oracle - CS_g_oracle,
    }

    return results

def rejection_rate(differences, alpha=0.05):
    t_stat, p_val = ttest_1samp(differences, popmean=0)
    reject = p_val < alpha
    return reject, p_val

###########################################################
def compare_trueU_ecdfU_score(R, P, H, grid_size, theta, delta, df, verbose=True):
    """
    Compare scoring results using the ECDF-based region vs the true region
    defined from known marginal inverse CDFs.

    Parameters:
        R                   : sample size
        P                   : number of (H-day-ahead) forecasts to evaluate
        H                   : forecast horizon
        theta, delta        : bb7 copula parameters
        df                  : Degrees of freedom for Student-t marginals
        verbose             : If True, print score comparisons and show region plots

    Returns:
        A dictionary with CLS and CSL scores for both region types and region masks
    """

    numpy2ri.activate()

    n = R + P + H - 1

    # Step 1: Simulate PITs from BB7 copula via R
    ro.globalenv['n'] = n
    ro.globalenv['grid_size'] = grid_size
    ro.globalenv['theta'] = theta
    ro.globalenv['delta'] = delta

    ro.r('''
    library(VineCopula)
    set.seed(123)

    # Simulate PITs
    cop_model <- BiCop(family = 17, par = theta, par2 = delta)
    u_sim <- BiCopSim(N = n, obj = cop_model)
    u1 <- u_sim[, 1]
    u2 <- u_sim[, 2]

    # Create evaluation grid
    margin <- 1 / ( 2 * grid_size)
    u_seq <- seq(margin, 1 - margin, length.out = grid_size)
    grid <- expand.grid(u_seq, u_seq)

    # Evaluate PDF on grid
    pdf_vals <- BiCopPDF(grid[,1], grid[,2], cop_model)
    ''')

    # True PIT values used to compute the true region quantile
    true_u1 = np.array(ro.r('u1'))
    true_u2 = np.array(ro.r('u2'))

    pdf_vals = np.array(ro.r('pdf_vals'))

    # Create meshgrid of u1 and u2 (to be same shape as copula_pdf)
    u_seq = np.linspace(0, 1, grid_size)
    U1, U2 = np.meshgrid(u_seq, u_seq)

    # 2D numpy array of copula density on [0,1]^2
    copula_pdf = pdf_vals.reshape(U1.shape)

    # Create y1 and y2 using the true PITs and analytical inverse
    y1 = student_t.ppf(true_u1, df)
    y2 = student_t.ppf(true_u2, df)

    # Now from with these y1 and y2 create the ECDF function of the PITs

    F1_inv_ecdf = compute_ecdf_inverse(y1)
    F2_inv_ecdf = compute_ecdf_inverse(y2)

    # Compute empirical quantile
    q_emp = np.quantile(y1 + y2, 0.05)

    # Define empirical region

    W_emp = create_region(U1, U2, F1_inv=F1_inv_ecdf, F2_inv=F2_inv_ecdf, q_alpha=q_emp)

    # Define true inverse CDFs (analytical)
    F1_inv_true = compute_true_t_inverse(df)
    F2_inv_true = compute_true_t_inverse(df)

    # Compute true quantile
    q_true = np.quantile(F1_inv_true(true_u1) + F2_inv_true(true_u2), 0.05)

    # Define true region
    W_true = create_region(U1, U2, F1_inv=F1_inv_true, F2_inv=F2_inv_true, q_alpha=q_true)

    # Compute scoring rules for both regions
    results = {
        "CLS_emp": compute_scores_over_region(copula_pdf, W_emp)[1],
        "CLS_true": compute_scores_over_region(copula_pdf, W_true)[1],
        "CS_emp": compute_scores_over_region(copula_pdf, W_emp)[0],
        "CS_true": compute_scores_over_region(copula_pdf, W_true)[0],
        "W_true": W_true,
        "q_true": q_true
    }

    for t in range(R, R + P):
        u1_train = true_u1[t - R:t]
        u2_train = true_u2[t - R:t]

        # We pretend we predicted these because we had a correct copula specification
        u1_target = true_u1[t + H - 1]
        u1_target = true_u1[t + H - 1]

        y1_train = student_t.ppf(u1_train, df)
        y2_train = student_t.ppf(u2_train, df)

        F1_inv_ecdf_train = compute_ecdf_inverse(y1_train)
        F2_inv_ecdf_train = compute_ecdf_inverse(y2_train)

        q_emp_train = np.quantile(y1_train + y2_train, 0.05)


        W_emp_train = create_region(U1, U2, F1_inv=F1_inv_ecdf_train, F2_inv=F2_inv_ecdf_train, q_alpha=q_emp_train)

        F1_inv_true = compute_true_t_inverse(df)
        F2_inv_true = compute_true_t_inverse(df)

        q_true_train = np.quantile(F1_inv_true(u1_train) + F2_inv_true(u2_train), 0.05)

        W_true_train = create_region(U1, U2, F1_inv=F1_inv_true, F2_inv=F2_inv_true, q_alpha=q_true_train)

        CS_emp_train, CLS_emp_train = compute_scores_over_region(copula_pdf, W_emp_train)
        CS_true_train, CLS_true_train = compute_scores_over_region(copula_pdf, W_true_train)

    if verbose:
        print(f"CLS (ECDF region):  {results['CLS_emp']:.4f}")
        print(f"CLS (True region):  {results['CLS_true']:.4f}")
        print(f"CS (ECDF region):  {results['CS_emp']:.4f}")
        print(f"CS (True region):  {results['CS_true']:.4f}")

        plt.figure(figsize=(8, 6))
        plt.contour(U1, U2, W_emp, levels=[0.5], colors='red', linewidths=2, label="ECDF Region")
        plt.contour(U1, U2, W_true, levels=[0.5], colors='blue', linewidths=2, label="True Region")
        plt.title("True vs ECDF Region Boundaries in PIT Space")
        plt.xlabel("u1")
        plt.ylabel("u2")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return results


###########################################################
def compare_trueU_ecdfU_score_test_version(R, P, H, grid_size, df, verbose=True):
    """
    Compare scoring results using the ECDF-based region vs the true region
    defined from known marginal inverse CDFs.

    Parameters:
        R                   : sample size
        P                   : number of (H-day-ahead) forecasts to evaluate
        H                   : forecast horizon
        df                  : Degrees of freedom for Student-t
        verbose             : If True, print score comparisons and/or show region plots

    Returns:
        A dictionary with CLS and CSL scores for both region types and region masks
    """
    # Param setup
    n = R + P + H - 1
    margin = 1 / (2 * grid_size)
    u_seq = np.linspace(margin, 1 - margin, grid_size)
    U1, U2 = np.meshgrid(u_seq, u_seq)

    # Simulate true PITs from independent DGP
    u1, u2 = simulate_independent_t_copula(n, df)

    # Precompute t-inverse values once for oracle case
    t_inv = student_t.ppf(u_seq, df)
    Y1_oracle = np.tile(t_inv, (grid_size, 1))  # columns = u_seq
    Y2_oracle = np.tile(t_inv[:, np.newaxis], (1, grid_size))  # rows = u_seq

    # Create inverse oracle CDF and inverse ECDF funtions
    y1 = student_t.ppf(u1, df)
    y2 = student_t.ppf(u2, df)

    # Precompute 1D inverse for ECDF
    t_inv_ecdf = np.quantile(y1, u_seq)  # same for y2 in symmetric case

    # Broadcast to grid
    Y1_emp = np.tile(t_inv_ecdf, (grid_size, 1))
    Y2_emp = np.tile(t_inv_ecdf[:, np.newaxis], (1, grid_size))

    # Create quantiles
    q_oracle = np.quantile(y1 + y2, 0.05)
    q_ecdf = np.quantile(y1 + y2, 0.05)

    # Create region masks
    W_oracle = ((Y1_oracle + Y2_oracle) <= q_oracle).astype(int)
    W_ecdf = ((Y1_emp + Y2_emp) <= q_ecdf).astype(int)

    # Create pdf of two equally misspecified Student-t copulas
    pdf_f = copula_pdf_student_t(U1.ravel(), U2.ravel(), rho=0.3, df=df).reshape(grid_size, grid_size)
    pdf_g = copula_pdf_student_t(U1.ravel(), U2.ravel(), rho=-0.3, df=df).reshape(grid_size, grid_size)

    # Compute scores for both copulas and both regions
    CS_f_oracle, CLS_f_oracle = compute_scores_over_region(pdf_f, W_oracle)
    CS_g_oracle, CLS_g_oracle = compute_scores_over_region(pdf_g, W_oracle)
    CS_f_ecdf, CLS_f_ecdf = compute_scores_over_region(pdf_f, W_ecdf)
    CS_g_ecdf, CLS_g_ecdf = compute_scores_over_region(pdf_g, W_ecdf)
    CLS_diff_ecdf = CLS_f_ecdf - CLS_g_ecdf
    CLS_diff_oracle = CLS_f_oracle - CLS_g_oracle
    CS_diff_ecdf = CS_f_ecdf - CS_g_ecdf
    CS_diff_oracle = CS_f_oracle - CS_g_oracle

    results = [
        CS_f_oracle, CLS_f_oracle,
        CS_g_oracle, CLS_g_oracle,
        CS_f_ecdf, CLS_f_ecdf,
        CS_g_ecdf, CLS_g_ecdf,
        CLS_diff_ecdf, CLS_diff_oracle,
        CS_diff_ecdf, CS_diff_oracle,
        W_oracle, W_ecdf
    ]

    if verbose:
        print(f"CLS of f (oracle region):  {results[1]:.4f}")
        print(f"CLS of f (ECDF region):  {results[5]:.4f}")
        print(f"CLS of g (oracle region):  {results[3]:.4f}")
        print(f"CLS of g (ECDF region):  {results[7]:.4f}")
        print("")
        print(f"CS of f (oracle region):  {results[0]:.4f}")
        print(f"CS of f (ECDF region):  {results[4]:.4f}")
        print(f"CS of g (oracle region):  {results[2]:.4f}")
        print(f"CS of g (ECDF region):  {results[6]:.4f}")
        print("")
        print(f"CLS difference of ecdf case (f-g):  {results[8]:.4f}")
        print(f"CLS difference of oracle case (f-g):  {results[9]:.4f}")
        print(f"CS difference of ecdf case (f-g):  {results[10]:.4f}")
        print(f"CS difference of oracle case (f-g):  {results[11]:.4f}")



        plt.figure(figsize=(8, 6))
        plt.contourf(U1, U2, np.log(pdf_f), cmap='Blues', levels=30)
        plt.contour(U1, U2, W_ecdf, levels=[0.5], colors='red', linewidths=2, label="ECDF Region")
        plt.contour(U1, U2, W_oracle, levels=[0.5], colors='blue', linewidths=2, label="Oracle Region")
        plt.title("True vs ECDF Region Boundaries in PIT Space")
        plt.xlabel("u1")
        plt.ylabel("u2")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return results


###########################################################
def run_single_t_loop(i, R, P, H, grid_size, df):
    return i, *compare_trueU_ecdfU_score_test_version(R, P, H, grid_size, df, verbose=False)[-6:]


###########################################################
def compare_trueU_ecdfU_score_t_loop(iterations, grid_size, R, P, H, df, verbose=True):
    """
    Iterates the boundry creation and makes a mean boundary and confidence intervals

    Parameters:
        iterations          : number of iterations
        verbose             : If True, print score comparisons and/or show region plots

    Returns:
        A plot with mean boundaries and confidence intervals
    """
    margin = 1 / (2 * grid_size)
    u_seq = np.linspace(margin, 1 - margin, grid_size)
    U1, U2 = np.meshgrid(u_seq, u_seq)

    # Storage for region masks across repetitions
    oracle_masks = np.zeros((iterations, grid_size, grid_size), dtype=int)
    ecdf_masks = np.zeros((iterations, grid_size, grid_size), dtype=int)
    CLS_diffs_ecdf = np.zeros(iterations)
    CLS_diffs_oracle = np.zeros(iterations)
    CS_diffs_ecdf = np.zeros(iterations)
    CS_diffs_oracle = np.zeros(iterations)

    pairwise_diffs = []

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_single_t_loop, i, R, P, H, grid_size, df) for i in range(iterations)]

        # tracks feature completion
        for future in tqdm(concurrent.futures.as_completed(futures), total=iterations, desc="Running simulations"):
            i, CLS_diff_ecdf, CLS_diff_oracle, CS_diff_ecdf, CS_diff_oracle, W_oracle, W_ecdf = future.result()
            oracle_masks[i] = W_oracle
            ecdf_masks[i] = W_ecdf
            diff = ecdf_masks[i] - oracle_masks[i]
            mean_diff = np.mean(diff)
            pairwise_diffs.append(mean_diff)
            CLS_diffs_ecdf[i] = CLS_diff_ecdf
            CLS_diffs_oracle[i] = CLS_diff_oracle
            CS_diffs_ecdf[i] = CS_diff_ecdf
            CS_diffs_oracle[i] = CS_diff_oracle


    # Mean and CI bounds (across simulations)
    mean_oracle = np.mean(oracle_masks, axis=0)
    mean_ecdf = np.mean(ecdf_masks, axis=0)

    # Compute 95% CI bounds
    lower_oracle = np.percentile(oracle_masks, 2.5, axis=0)
    upper_oracle = np.percentile(oracle_masks, 97.5, axis=0)

    lower_ecdf = np.percentile(ecdf_masks, 2.5, axis=0)
    upper_ecdf = np.percentile(ecdf_masks, 97.5, axis=0)

    if verbose:
        # Plot mean boundary with confidence interval overlays
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.contour(U1, U2, mean_oracle, levels=[0.5], colors='blue', linewidths=2, label='Mean Oracle')
        ax.contour(U1, U2, mean_ecdf, levels=[0.5], colors='red', linewidths=2, label='Mean ECDF')

        # Plot CI boundaries
        ax.contour(U1, U2, lower_oracle, levels=[0.5], colors='blue', linestyles='dashed', linewidths=1)
        ax.contour(U1, U2, upper_oracle, levels=[0.5], colors='blue', linestyles='dashed', linewidths=1)
        ax.contour(U1, U2, lower_ecdf, levels=[0.5], colors='red', linestyles='dashed', linewidths=1)
        ax.contour(U1, U2, upper_ecdf, levels=[0.5], colors='red', linestyles='dashed', linewidths=1)

        ax.set_title("Mean Region Boundaries and 95% CIs (Oracle vs ECDF)")
        ax.set_xlabel("u1")
        ax.set_ylabel("u2")
        ax.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.hist(pairwise_diffs, bins=30, color='gray', edgecolor='black')
        plt.axvline(np.mean(pairwise_diffs), color='red', linestyle='--', label=f"Mean = {np.mean(pairwise_diffs):.4f}")
        plt.title("Distribution of Mean Pairwise ECDF - Oracle Region Differences")
        plt.xlabel("Mean difference per simulation (ECDF - Oracle)")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.hist(CLS_diffs_ecdf, bins=30, color='gray', edgecolor='black')
        plt.axvline(np.mean(CLS_diffs_ecdf), color='red', linestyle='--', label=f"Mean = {np.mean(CLS_diffs_ecdf):.4f}")
        plt.title("Distribution of ECDF CLS Score Differences (f-g)")
        plt.xlabel("Difference per simulation (f-g)")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.hist(CLS_diffs_oracle, bins=30, color='gray', edgecolor='black')
        plt.axvline(np.mean(CLS_diffs_oracle), color='red', linestyle='--', label=f"Mean = {np.mean(CLS_diffs_oracle):.4f}")
        plt.title("Distribution of Oracle CLS Score Differences (f-g)")
        plt.xlabel("Difference per simulation (f-g)")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return mean_oracle, mean_ecdf, lower_oracle, upper_oracle, lower_ecdf, upper_ecdf

