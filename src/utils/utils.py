import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from scipy.stats import gaussian_kde
from scipy.stats import t as student_t
import scipy.optimize as opt


def WhiteNoiseSim(iT, vDistrParams, sDistrName):
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


def to_unif_PIT(residuals):
    """
        Purpose:
            Converts residuals to uniform margins using empirical CDF

        Inputs:
            residuals       input series

        Output:
            Residuals series mapped to [0,1]

    """
    ranks = rankdata(residuals, method="average")
    return ranks / (len(residuals) + 1)


###########################################################


def R_bb7(u1, u2, theta, delta, resid1, resid2):
    """
        Purpose:
            Uses residuals series their sizes to create bivariate bb7 PITs

        Inputs:
            u1, u2          input series of uniforms from PIT transformation of marginal residuals
            theta           bb7 dependence parameter (>1)
            delta           bb7 tail asymmetry parameter (>1)

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
    u_seq <- seq(0.01, 0.99, length.out = 100)
    grid <- expand.grid(u_seq, u_seq)
    true_pdf <- BiCopPDF(grid[,1], grid[,2], cop_model)
    ''')
    # Retrieve simulated values and optional copula CDF evaluations
    sim_u1 = np.array(ro.r('sim_u1'))
    sim_u2 = np.array(ro.r('sim_u2'))
    copula_cdf_values = np.array(ro.r('cop_cdf'))  # Optional
    true_pdf = np.array(ro.r('true_pdf')).reshape((100, 100))
    u_seq = np.linspace(0.01, 0.99, 100)
    U1, U2 = np.meshgrid(u_seq, u_seq)

    # Estimate empirical density using KDE
    uv = np.vstack([sim_u1, sim_u2])
    kde = gaussian_kde(uv)
    empirical_pdf = kde(np.vstack([U1.ravel(), U2.ravel()])).reshape(U1.shape)

    # Prepare for plotting
    fig = plt.figure(figsize=(14, 6))

    # Plot empirical KDE
    # ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    # ax1.plot_surface(U1, U2, empirical_pdf, cmap='viridis', linewidth=0)
    # ax1.set_title("Empirical Copula Density (KDE)")
    # ax1.set_xlabel("u1")
    # ax1.set_ylabel("u2")
    # ax1.set_zlabel("Density")

    # Plot theoretical BB7 PDF

    true_pdf_clipped = np.clip(true_pdf, 0, 40)
    # ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    # ax2.plot_surface(U1, U2, true_pdf_clipped, cmap='plasma', linewidth=0)
    # ax2.set_title("Theoretical BB7 Copula PDF (clipped at 40)")
    # ax2.set_xlabel("u1")
    # ax2.set_ylabel("u2")
    # ax2.set_zlabel("Density")

    # plt.tight_layout()
    # plt.show()

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

    # # 2D contour plot
    # plt.figure(figsize=(7, 6))
    # contour = plt.contourf(U1, U2, empirical_pdf, levels=30, cmap="viridis")
    # plt.contour(U1, U2, W, levels=[0.5], colors='red', linewidths=2)
    # plt.title("Empirical Copula Density with Region y1 + y2 ≤ q₀.₀₅")
    # plt.xlabel("u1 (PIT)")
    # plt.ylabel("u2 (PIT)")
    # plt.colorbar(contour, label="Density")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    return sim_u1, sim_u2, copula_cdf_values, W, empirical_pdf, true_pdf


###########################################################


def simulate_joint_t_marginals(n, df, theta, delta):
    """
    Simulate joint sample from a BB7 copula and Student-t marginals.

    Parameters:
        n (int): Number of samples
        df (float): Degrees of freedom for the Student-t marginals
        theta (float): BB7 copula parameter (dependence strength)
        delta (float): BB7 copula parameter (tail asymmetry)

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
    # plt.figure(figsize=(8, 6))
    # contour = plt.contourf(U1, U2, np.clip(copula_pdf, 0, 40), levels=30, cmap="viridis")
    # plt.contour(U1, U2, W_true, levels=[0.5], colors="red", linewidths=2)
    # plt.title("BB7 Copula PDF with True Region Overlay")
    # plt.xlabel("u1")
    # plt.ylabel("u2")
    # plt.colorbar(contour, label="Copula Density")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    return y1, y2, u1, u2, q_true, U1, U2, W_true, copula_pdf


###########################################################


def inverse_ecdf(u, original_resids):
    """
        Purpose:
            Inverts the PITs back to residuals using inverse ECDF

        Inputs:
            u                   input series of PITs
            original_resids     original AR-GARCH residuals

        Output:
            GARCH style residuals respecting BB7 dependence

    """
    sorted_resids = np.sort(original_resids)
    n = len(sorted_resids)
    indices = np.minimum((u * n).astype(int), n - 1)
    return sorted_resids[indices]


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

    return resid1, variance1, resid2, variance2


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