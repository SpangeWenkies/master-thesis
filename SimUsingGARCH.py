import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D

# Global settings
lFigureSettings = {'figsize': (10, 6), 'dpi': 70, 'titlefontsize': 16, 'axisfontsize': 14}
iBins = 30  # int(np.floor(len(vX)/25))
numpy2ri.activate()


###########################################################
# Plot


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
    if (bLine == True):
        plt.plot(vX, c='blue')
        plt.xlabel('$t$', fontsize=lFigureSettings['axisfontsize'])
        plt.ylabel('$y_t$', fontsize=lFigureSettings['axisfontsize'])
    else:
        plt.hist(vX, iBins)
    plt.title(sTitle, fontsize=lFigureSettings['titlefontsize'])
    plt.show()


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


def R_bb7(u1,u2, theta, delta, resid1, resid2):
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

    # Compute shared z-axis range for both surfaces
    zmin = 0
    zmax = max(empirical_pdf.max(), true_pdf.max())

    # Plot empirical KDE
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(U1, U2, empirical_pdf, cmap='viridis', linewidth=0)
    ax1.set_title("Empirical Copula Density (KDE)")
    ax1.set_xlabel("u1")
    ax1.set_ylabel("u2")
    ax1.set_zlabel("Density")

    # Plot theoretical BB7 PDF
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(U1, U2, true_pdf, cmap='plasma', linewidth=0)
    ax2.set_title("Theoretical BB7 Copula PDF")
    ax2.set_xlabel("u1")
    ax2.set_ylabel("u2")
    ax2.set_zlabel("Density")
    ax2.set_zlim(zmin, zmax)

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

###########################################################
# main


def main():
    # Cleaning
    plt.close('all')

    # Magic numbers
    sPlot = 'line'
    iSeed = 12085278

    # GARCH params
    n = 10000
    dist = 't'
    nu = np.array([5])
    omega = np.array([0.01])
    alpha = np.array([0.1])  # np.array([0.1]) #np.array([0.05, 0.1])
    beta = np.array([0.6])

    # bb7 params
    theta = np.array([3])
    delta = np.array([2])

    # Set seed
    np.random.seed(iSeed)

    # White noise
    # WhiteNoiseSim(iT,vDistrParams,dist,sPlot)

    # GARCH(p,q)
    resid1, variance1, resid2, variance2 = simulate_GARCH(n, omega, alpha, beta, dist, df=nu)

    u1 = to_unif_PIT(resid1)
    u2 = to_unif_PIT(resid2)

    sim_u1, sim_u2, copula_cdf_values, W, empirical_pdf, true_pdf = R_bb7(u1, u2, theta, delta, resid1, resid2)

    sim_resid1 = inverse_ecdf(sim_u1, resid2)
    sim_resid2 = inverse_ecdf(sim_u2, resid2)

    mass_empirical = evaluate_mass_in_region(empirical_pdf, W)
    mass_theoretical = evaluate_mass_in_region(true_pdf, W)

    print(f"Mass in region (Empirical):  {mass_empirical:.4f}")
    print(f"Mass in region (Theoretical BB7): {mass_theoretical:.4f}")

    iQ = len(beta)
    iP = len(alpha)

    # sTitle = "GARCH(%d, %d)" % (iP, iQ)
    sTitle = "residuals mapped to [0,1]"
    if (sPlot == 'line'):
        Plot(u1, sTitle, True)
    elif (sPlot == 'hist'):
        Plot(u1, sTitle, False)


###########################################################
# start main
if __name__ == "__main__":
    main()
