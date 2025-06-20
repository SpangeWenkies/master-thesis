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

def plot_score_diff_cdf(diff_oracle, diff_ecdf, title, title2):
    """
    Plot empirical CDFs of score differences using oracle and ECDF-based PITs.

    Parameters
    ----------
    diff_oracle : np.ndarray
        Score differences (copula 1 - copula 2) calculated with oracle PITs.
    diff_ecdf : np.ndarray
        Score differences (copula 1 - copula 2) calculated with ECDF PITs.
    title : str
        Name of the score type (e.g. "LogS", "CS", "CLS").
    title2 : str
        Additional title descriptor, typically the copula pair.
    """
    sorted_oracle = np.sort(diff_oracle)
    sorted_ecdf = np.sort(diff_ecdf)

    cdf_oracle = np.arange(1, len(sorted_oracle) + 1) / len(sorted_oracle)
    cdf_ecdf = np.arange(1, len(sorted_ecdf) + 1) / len(sorted_ecdf)

    plt.figure(figsize=(10, 6))
    plt.step(sorted_oracle, cdf_oracle, where="post", label="Oracle PITs", color="purple")
    plt.step(sorted_ecdf, cdf_ecdf, where="post", label="ECDF PITs", color="orange")

    plt.title(f"{title} – Score Difference {title2}")
    plt.xlabel("Score Difference")
    plt.ylabel("Empirical CDF")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def validate_plot_data(score_dicts, pair_names, score_names, pair_label=None):
    """
    Validates that both oracle and ecdf data exist and are valid for each selected pair.
    """
    print("=== Validating Plot Data ===")

    # Apply optional filter
    if pair_label is None:
        selected_pairs = pair_names
    elif isinstance(pair_label, str):
        selected_pairs = {k: v for k, v in pair_names.items() if v == pair_label}
    elif isinstance(pair_label, list):
        selected_pairs = {k: v for k, v in pair_names.items() if v in pair_label}
    else:
        raise ValueError("pair_label must be None, a string, or a list of strings.")

    issues_found = False

    for label in sorted(set(selected_pairs.values())):
        key_group = [k for k, l in selected_pairs.items() if l == label]
        oracle_key = next((k for k in key_group if getattr(k, "pit", "")=="oracle"), None)
        ecdf_key = next((k for k in key_group if getattr(k, "pit", "")=="ecdf"), None)

        if not oracle_key or not ecdf_key:
            print(f"Missing oracle or ecdf key for pair: {label}")
            issues_found = True
            continue

        if oracle_key not in score_dicts or ecdf_key not in score_dicts:
            print(f"Missing score_dict entries for: {oracle_key}, {ecdf_key}")
            issues_found = True
            continue

        for score in score_names:
            data_o = score_dicts[oracle_key].get(score, None)
            data_e = score_dicts[ecdf_key].get(score, None)

            if data_o is None or data_e is None:
                print(f"Missing score '{score}' for pair {label}")
                issues_found = True
            elif len(data_o) == 0 or len(data_e) == 0:
                print(f"Empty data for '{score}' in pair {label}")
                issues_found = True
            elif not np.all(np.isfinite(data_o)) or not np.all(np.isfinite(data_e)):
                print(f"Non-finite values in '{score}' for pair {label}")
                issues_found = True

    if not issues_found:
        print("All selected pairs are valid for plotting.")

def plot_score_differences(score_dicts, score_names, pair_to_keys):
    """
    Plot histograms and KDEs of score differences using explicitly passed keys per pair.

    Parameters:
    - score_dicts: dict of {DiffKey: {score_name: array}}
    - score_names: list of score names
    - pair_to_keys: dict mapping plot label (e.g. "f - g") -> (oracle_key, ecdf_key)
    """
    for label, (oracle_key, ecdf_key) in pair_to_keys.items():
        if oracle_key not in score_dicts or ecdf_key not in score_dicts:
            print(f"Skipping {label} — missing: "
                  f"{'oracle' if oracle_key not in score_dicts else ''} "
                  f"{'ecdf' if ecdf_key not in score_dicts else ''}")
            continue

        oracle_scores = score_dicts[oracle_key]
        ecdf_scores = score_dicts[ecdf_key]

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

def plot_score_differences_cdf(score_dicts, score_names, pair_to_keys):
    """
    Plot empirical CDFs of score differences using explicitly passed suffixes per pair.

    Parameters
    ----------
    score_dicts : dict
        Dictionary mapping DiffKey -> {score_name: array}.
    score_names : list
        List of score names to plot.
    pair_to_keys : dict
        Mapping plot label (e.g. "f - g") -> (oracle_key, ecdf_key).
    """
    for label, (oracle_key, ecdf_key) in pair_to_keys.items():
        if oracle_key not in score_dicts or ecdf_key not in score_dicts:
            print(
                f"Skipping {label} — missing: "
                f"{'oracle' if oracle_key not in score_dicts else ''} "
                f"{'ecdf' if ecdf_key not in score_dicts else ''}"
            )
            continue

        oracle_scores = score_dicts[oracle_key]
        ecdf_scores = score_dicts[ecdf_key]

        num_scores = len(score_names)
        fig, axes = plt.subplots(1, num_scores, figsize=(6 * num_scores, 5), sharey=True)
        if num_scores == 1:
            axes = [axes]

        for ax, score in zip(axes, score_names):
            data_oracle = np.sort(oracle_scores[score])
            data_ecdf = np.sort(ecdf_scores[score])

            cdf_oracle = np.arange(1, len(data_oracle) + 1) / len(data_oracle)
            cdf_ecdf = np.arange(1, len(data_ecdf) + 1) / len(data_ecdf)

            ax.step(data_oracle, cdf_oracle, where="post", label="Oracle", color="blue")
            ax.step(data_ecdf, cdf_ecdf, where="post", label="ECDF", color="orange")

            ax.set_title(f"{score}: Oracle vs ECDF")
            ax.set_xlabel("Score Difference")
            ax.set_ylabel("Empirical CDF")
            ax.legend()
            ax.grid(True)

        fig.suptitle(f"Score Differences for {label}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

def plot_aligned_kl_matched_scores(score_dicts, score_score_keys):
    """
    Plots three score difference histograms/KDEs aligned horizontally.

    Parameters:
    - score_dicts: dict of {DiffKey: {score_name: np.ndarray}}, as constructed from your diffs.
    - score_score_keys: list of tuples:
        [
            (score_name, label, oracle_key, ecdf_key),
            ...
        ]
      For example:
        [
            ("LogS", "bb1 - f_for_KL_matching", "oracle_bb1_f_for_KL_matching", "ecdf_bb1_f_for_KL_matching"),
            ("CS", "bb1_localized - f_for_KL_matching", "oracle_bb1_localized_f_for_KL_matching", "ecdf_bb1_localized_f_for_KL_matching"),
            ("CLS", "bb1_local - f_for_KL_matching", "oracle_bb1_local_f_for_KL_matching", "ecdf_bb1_local_f_for_KL_matching"),
        ]
    """
    num_scores = len(score_score_keys)
    fig, axes = plt.subplots(1, num_scores, figsize=(6 * num_scores, 5), sharey=True)

    if num_scores == 1:
        axes = [axes]

    for ax, (score, label, oracle_key, ecdf_key) in zip(axes, score_score_keys):
        if oracle_key not in score_dicts or ecdf_key not in score_dicts:
            print(f" Skipping {label} — missing: "
                  f"{'oracle' if oracle_key not in score_dicts else ''} "
                  f"{'ecdf' if ecdf_key not in score_dicts else ''}")
            continue

        oracle_data = score_dicts[oracle_key][score]
        ecdf_data = score_dicts[ecdf_key][score]

        kde_oracle = gaussian_kde(oracle_data)
        kde_ecdf = gaussian_kde(ecdf_data)

        x_min = min(oracle_data.min(), ecdf_data.min())
        x_max = max(oracle_data.max(), ecdf_data.max())
        x_grid = np.linspace(x_min, x_max, 500)

        ax.hist(oracle_data, bins=30, alpha=0.3, density=True, label='Oracle', color='blue')
        ax.hist(ecdf_data, bins=30, alpha=0.3, density=True, label='ECDF', color='orange')
        ax.plot(x_grid, kde_oracle(x_grid), label='KDE Oracle', linewidth=2, color='blue')
        ax.plot(x_grid, kde_ecdf(x_grid), label='KDE ECDF', linewidth=2, color='orange')

        ax.set_title(f"{score} — {label}")
        ax.set_xlabel("Score Difference")
        ax.grid(True)

    axes[0].set_ylabel("Density")
    axes[0].legend()
    fig.suptitle("KL-Matched Score Differences", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_aligned_kl_matched_scores_cdf(score_dicts, score_score_keys):
    """
    Plot three score difference CDFs aligned horizontally.

    Parameters
    ----------
    score_dicts : dict
        Mapping DiffKey -> {score_name: np.ndarray}.
    score_score_keys : list of tuples
        [(score_name, label, oracle_key, ecdf_key), ...]
    """
    num_scores = len(score_score_keys)
    fig, axes = plt.subplots(1, num_scores, figsize=(6 * num_scores, 5), sharey=True)

    if num_scores == 1:
        axes = [axes]

    for ax, (score, label, oracle_key, ecdf_key) in zip(axes, score_score_keys):
        if oracle_key not in score_dicts or ecdf_key not in score_dicts:
            print(
                f" Skipping {label} — missing: "
                f"{'oracle' if oracle_key not in score_dicts else ''} "
                f"{'ecdf' if ecdf_key not in score_dicts else ''}"
            )
            continue

        oracle_data = np.sort(score_dicts[oracle_key][score])
        ecdf_data = np.sort(score_dicts[ecdf_key][score])

        cdf_oracle = np.arange(1, len(oracle_data) + 1) / len(oracle_data)
        cdf_ecdf = np.arange(1, len(ecdf_data) + 1) / len(ecdf_data)

        ax.step(oracle_data, cdf_oracle, where="post", label="Oracle", color="blue")
        ax.step(ecdf_data, cdf_ecdf, where="post", label="ECDF", color="orange")

        ax.set_title(f"{score} — {label}")
        ax.set_xlabel("Score Difference")
        ax.grid(True)

    axes[0].set_ylabel("Empirical CDF")
    axes[0].legend()
    fig.suptitle("KL-Matched Score Differences", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_size_test_results(size_results, score_names, alpha=0.05):
    """Visualize p-values from size tests across score types."""

    for label, res_pair in size_results.items():
        oracle_pvals = [res_pair[sc]["oracle"]["p_value"] for sc in score_names]
        ecdf_pvals = [res_pair[sc]["ecdf"]["p_value"] for sc in score_names]

        indices = np.arange(len(score_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(indices - width / 2, oracle_pvals, width, label="Oracle")
        ax.bar(indices + width / 2, ecdf_pvals, width, label="ECDF")
        ax.axhline(alpha, color="red", linestyle="--", label=f"alpha={alpha}")
        ax.set_xticks(indices)
        ax.set_xticklabels(score_names)
        ax.set_ylabel("p-value")
        ax.set_title(f"Size Test p-values – {label}")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()

def plot_size_curves(size_curve_dict, pair_labels, plot_type="discrepancy", title=None):
    """Plot size discrepancy or rejection rate curves.

    Parameters
    ----------
    size_curve_dict : dict
        Mapping DiffKey -> (alpha_grid, rejection_rates, discrepancies).
    pair_labels : dict
        Mapping DiffKey -> human-readable pair label.
    plot_type : {"discrepancy", "rejection"}
        Select y-axis content.
    title : str, optional
        Title for the plot.
    """

    if plot_type not in {"discrepancy", "rejection"}:
        raise ValueError("plot_type must be 'discrepancy' or 'rejection'")

    fig, ax = plt.subplots(figsize=(6, 4))

    for key, (alpha_grid, rejection_rates, discrepancies) in size_curve_dict.items():
        label = pair_labels.get(key, key)
        y = discrepancies if plot_type == "discrepancy" else rejection_rates
        ax.plot(alpha_grid, y, marker="o", label=label)

    ax.set_xlabel("alpha")
    if plot_type == "discrepancy":
        ax.set_ylabel("rejection rate - alpha")
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    else:
        ax.set_ylabel("rejection rate")
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)

    if title is None:
        title = "Size Curves"
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()