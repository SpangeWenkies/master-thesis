import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


###########################################################
# Plot

def Plot(vX: np.ndarray, sTitle: str, bLine: bool) -> None:
    """Plot a time series or histogram.

    Parameters
    ----------
    vX : ndarray of shape (n,)
        Data to plot.
    sTitle : str
        Title of the figure.
    bLine : bool
        If ``True`` draw a line plot, otherwise draw a histogram.

    Returns
    -------
    None
        The plot is shown on screen.
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


def plot_histogram_kde(
    data_f: np.ndarray,
    data_g: np.ndarray,
    data_p: np.ndarray,
    title: str,
    pit_type: str,
) -> None:
    """Plot histograms and KDE overlays for three distributions.

    Parameters
    ----------
    data_f, data_g, data_p : ndarray of shape (n,)
        Samples from two candidate models and the true distribution.
    title : str
        Title of the score being displayed.
    pit_type : str
        Indicates whether the PITs are oracle or ECDF based.

    Returns
    -------
    None
        The plot is shown on screen.
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

def plot_score_diff_histogram_kde(
    diff_oracle: np.ndarray,
    diff_ecdf: np.ndarray,
    title: str,
    title2: str,
) -> None:
    """Plot histograms of score differences for oracle and ECDF PITs.

    Parameters
    ----------
    diff_oracle : ndarray of shape (n,)
        Score differences computed with oracle PITs.
    diff_ecdf : ndarray of shape (n,)
        Score differences computed with ECDF PITs.
    title : str
        Name of the score type.
    title2 : str
        Additional descriptor for the copula pair.

    Returns
    -------
    None
        The plot is shown on screen.
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

def plot_score_diff_cdf(
    diff_oracle: np.ndarray,
    diff_ecdf: np.ndarray,
    title: str,
    title2: str,
) -> None:
    """Plot empirical CDFs of score differences.

    Parameters
    ----------
    diff_oracle : ndarray of shape (n,)
        Score differences computed with oracle PITs.
    diff_ecdf : ndarray of shape (n,)
        Score differences computed with ECDF PITs.
    title : str
        Name of the score type.
    title2 : str
        Additional descriptor for the copula pair.

    Returns
    -------
    None
        The plot is shown on screen.
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

def validate_plot_data(
    score_dicts: dict,
    pair_names: dict,
    score_names: list[str],
    pair_label: str | list[str] | None = None,
) -> None:
    """Validate that plot data are present for all selected pairs.

    Parameters
    ----------
    score_dicts : dict
        Mapping DiffKey -> {score_name: ndarray of shape (n,)}.
    pair_names : dict
        Mapping DiffKey -> human readable label.
    score_names : list of str
        Score names expected in ``score_dicts``.
    pair_label : str or list of str or None, optional
        Restrict validation to these labels.

    Returns
    -------
    None
        Prints messages about missing data.
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

def plot_score_differences(
    score_dicts: dict,
    score_names: list[str],
    pair_to_keys: dict,
) -> None:
    """Plot histograms of score differences for multiple pairs.

    Parameters
    ----------
    score_dicts : dict
        Mapping DiffKey -> {score_name: ndarray of shape (n,)}.
    score_names : list of str
        Score names to display.
    pair_to_keys : dict
        Mapping plot label -> (oracle_key, ecdf_key).

    Returns
    -------
    None
        The plots are shown on screen.
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

def plot_score_differences_cdf(
    score_dicts: dict,
    score_names: list[str],
    pair_to_keys: dict,
) -> None:
    """Plot empirical CDFs of score differences for multiple pairs.

    Parameters
    ----------
    score_dicts : dict
        Mapping DiffKey -> {score_name: ndarray of shape (n,)}.
    score_names : list of str
        Names of the scores to plot.
    pair_to_keys : dict
        Mapping label -> (oracle_key, ecdf_key).

    Returns
    -------
    None
        The plots are shown on screen.
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

def plot_aligned_kl_matched_scores(
    score_dicts: dict,
    score_score_keys: list[tuple[str, str, str, str]],
) -> None:
    """Plot aligned histograms of KL-matched score differences.

    Parameters
    ----------
    score_dicts : dict
        Mapping DiffKey -> {score_name: ndarray of shape (n,)}.
    score_score_keys : list of tuple
        ``(score_name, label, oracle_key, ecdf_key)`` for each subplot.

    Returns
    -------
    None
        The plots are shown on screen.
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

def plot_aligned_kl_matched_scores_cdf(
    score_dicts: dict,
    score_score_keys: list[tuple[str, str, str, str]],
) -> None:
    """Plot aligned CDFs of KL-matched score differences.

    Parameters
    ----------
    score_dicts : dict
        Mapping DiffKey -> {score_name: ndarray of shape (n,)}.
    score_score_keys : list of tuple
        ``(score_name, label, oracle_key, ecdf_key)`` describing each subplot.

    Returns
    -------
    None
        The plots are shown on screen.
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


def plot_size_test_results(
    size_results: dict,
    score_names: list[str],
    alpha: float = 0.05,
) -> None:
    """Visualize p-values from size tests across score types.

    Parameters
    ----------
    size_results : dict
        Mapping label -> {score_name: {"oracle": {"p_value": float}, "ecdf": {"p_value": float}}}.
    score_names : list of str
        Names of the scores to visualize.
    alpha : float, default 0.05
        Significance threshold drawn as a horizontal line.

    Returns
    -------
    None
        The plots are shown on screen.
    """

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

def plot_dm_size_discrepancy(
    alpha_grid: np.ndarray,
    rates_oracle: np.ndarray,
    rates_ecdf: np.ndarray,
    score: str,
    label: str = "f - g",
) -> None:
    """Plot size discrepancy curves for oracle and ECDF DM tests.

   Parameters
   ----------
   alpha_grid : ndarray of shape (m,)
       Significance levels for the DM test.
   rates_oracle : ndarray of shape (m,)
       Rejection rates based on oracle PITs.
   rates_ecdf : ndarray of shape (m,)
       Rejection rates based on ECDF PITs.
   score : str
       Score type used for the DM test.
   label : str, default "f - g"
       Identifier of the model pair shown in the title.
   """

    discrepancy_oracle = rates_oracle - alpha_grid
    discrepancy_ecdf = rates_ecdf - alpha_grid

    plt.figure(figsize=(6, 4))
    plt.plot(alpha_grid, discrepancy_oracle, marker="o", label="oracle")
    plt.plot(alpha_grid, discrepancy_ecdf, marker="s", label="ecdf")
    plt.plot(alpha_grid, 1.96 * np.sqrt(alpha_grid * (1 - alpha_grid) / len(discrepancy_oracle)), color="gray",
             linestyle="--", linewidth=1)
    plt.plot(alpha_grid, -1.96 * np.sqrt(alpha_grid * (1 - alpha_grid) / len(discrepancy_oracle)), color="gray",
             linestyle="--", linewidth=1)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("alpha")
    plt.ylabel("rejection rate - alpha")
    plt.title(f"Size discrepancy (two-sided DM, {score}, {label})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
