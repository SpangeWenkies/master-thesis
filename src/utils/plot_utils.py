import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


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

    lFigureSettings = {'figsize': (10, 6), 'dpi': 70, 'titlefontsize': 16, 'axisfontsize': 14}
    iBins = 30  # int(np.floor(len(vX)/25))

    plt.figure(figsize=lFigureSettings['figsize'], dpi=lFigureSettings['dpi'])
    if (bLine == True):
        plt.plot(vX, c='blue')
        plt.xlabel('$t$', fontsize=lFigureSettings['axisfontsize'])
        plt.ylabel('$y_t$', fontsize=lFigureSettings['axisfontsize'])
    else:
        plt.hist(vX, iBins)
    plt.title(sTitle, fontsize=lFigureSettings['titlefontsize'])
    plt.show()

def plot_histogram_kde(data_f, data_g, data_p, title, pit_type):
    """
        Plots histogram and KDE lines

        Inputs:
            data_f, data_g, data_p  :   The two candidate dist and true dist
            title :                     The name of the score for which the histogram will be plotted

        Returns:
            histograms and KDE lines overlays
        """
    kde_f = gaussian_kde(data_f)
    kde_g = gaussian_kde(data_g)
    kde_p = gaussian_kde(data_p)

    x_min = min(data_f.min(), data_g.min(), data_p.min())
    x_max = max(data_f.max(), data_g.max(), data_p.max())
    x_grid = np.linspace(x_min, x_max, 500)

    plt.figure(figsize=(10, 6))
    plt.hist(data_f, bins=30, alpha=0.3, density=True, label='Model f (rho=-0.3)', color='blue')
    plt.hist(data_g, bins=30, alpha=0.3, density=True, label='Model g (rho=0.3)', color='green')
    plt.hist(data_p, bins=30, alpha=0.3, density=True, label='Model p (rho=0)', color='red')

    plt.plot(x_grid, kde_f(x_grid), label='KDE f', linewidth=2, color='blue')
    plt.plot(x_grid, kde_g(x_grid), label='KDE g', linewidth=2, color='green')
    plt.plot(x_grid, kde_p(x_grid), label='KDE p', linewidth=2, color='red')

    plt.title(f"{title} Score Distribution over {len(data_f)} Repetitions for {pit_type} PITs ")
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_score_diff_histogram_kde(diff_oracle, diff_ecdf, title, title2):
    """
    Plots histogram and KDE lines of score differences using oracle and ECDF-based PITs.

    Inputs:
        diff_oracle : np.ndarray – score differences (copula 1 - copula 2) using oracle PITs
        diff_ecdf   : np.ndarray – score differences (copula 1 - copula 2) using ECDF PITs
        title       : str – title of the score type (e.g., "LogS", "CS", "CLS")

    Output:
        Histogram and KDE plot
    """
    kde_oracle = gaussian_kde(diff_oracle)
    kde_ecdf = gaussian_kde(diff_ecdf)

    x_min = min(diff_oracle.min(), diff_ecdf.min())
    x_max = max(diff_oracle.max(), diff_ecdf.max())
    x_grid = np.linspace(x_min, x_max, 500)

    plt.figure(figsize=(10, 6))
    plt.hist(diff_oracle, bins=30, alpha=0.3, density=True, label='Oracle PITs', color='purple')
    plt.hist(diff_ecdf, bins=30, alpha=0.3, density=True, label='ECDF PITs', color='orange')

    plt.plot(x_grid, kde_oracle(x_grid), label='KDE Oracle', linewidth=2, color='purple')
    plt.plot(x_grid, kde_ecdf(x_grid), label='KDE ECDF', linewidth=2, color='orange')

    plt.title(f"{title} – Score Difference {title2}")
    plt.xlabel("Score Difference")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_score_differences(score_dicts, score_names, pair_names, pair_label=None):
    """
    Plot histograms and KDEs of score differences.

    Parameters:
    - score_dicts: dict of {suffix: {score_name: array}}, e.g., {'oracle': {'LogS': ...}, ...}
    - score_names: list of score names (e.g., ["LogS", "CS", "CLS"])
    - pair_names: dict mapping suffixes (e.g., 'oracle', 'ecdf2') to pair labels (e.g., 'f - g')
    - pair_label: optional string; if specified, plot only that copula pair (e.g., "f - g")
    """
    # Determine which pairs to include
    if pair_label is None:
        selected_pairs = pair_names
    elif isinstance(pair_label, str):
        selected_pairs = {suffix: label for suffix, label in pair_names.items() if label == pair_label}
    elif isinstance(pair_label, list):
        selected_pairs = {suffix: label for suffix, label in pair_names.items() if label in pair_label}
    else:
        raise ValueError("pair_label must be None, a string, or a list of strings.")

    # Get unique groupings like "f - g", "p - g", etc.
    unique_pair_labels = sorted(set(selected_pairs.values()))

    for label in unique_pair_labels:
        # Get the two suffixes (oracle, ecdf) for the current pair
        suffix_group = [s for s, l in selected_pairs.items() if l == label]

        oracle_suffix = next((s for s in suffix_group if "oracle" in s), None)
        ecdf_suffix = next((s for s in suffix_group if "ecdf" in s), None)

        if not oracle_suffix or not ecdf_suffix:
            continue  # Skip if one is missing

        oracle_scores = score_dicts[oracle_suffix]
        ecdf_scores = score_dicts[ecdf_suffix]

        num_scores = len(score_names)
        fig, axes = plt.subplots(1, num_scores, figsize=(6 * num_scores, 5), sharey=True)

        if num_scores == 1:
            axes = [axes]

        for ax, score in zip(axes, score_names):
            data_oracle = oracle_scores[score]
            data_ecdf = ecdf_scores[score]

            kde_oracle = gaussian_kde(data_oracle)
            kde_ecdf = gaussian_kde(data_ecdf)

            x_min = min(data_oracle.min(), data_ecdf.min())
            x_max = max(data_oracle.max(), data_ecdf.max())
            x_grid = np.linspace(x_min, x_max, 500)

            ax.hist(data_oracle, bins=30, alpha=0.3, density=True, label='Oracle', color='blue')
            ax.hist(data_ecdf, bins=30, alpha=0.3, density=True, label='ECDF', color='orange')
            ax.plot(x_grid, kde_oracle(x_grid), label='KDE Oracle', linewidth=2, color='blue')
            ax.plot(x_grid, kde_ecdf(x_grid), label='KDE ECDF', linewidth=2, color='orange')

            ax.set_title(f"{score}: Oracle vs ECDF")
            ax.set_xlabel("Score Difference")
            ax.set_ylabel("Density")
            ax.legend()
            ax.grid(True)

        fig.suptitle(f"Score Differences for {label}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
