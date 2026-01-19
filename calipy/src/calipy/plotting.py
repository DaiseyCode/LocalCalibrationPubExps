import sklearn
import sklearn.metrics
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox

from .experiment_results import ExperimentResults


def postage_bar(
    exp_results: ExperimentResults,
    ax: Axes,
    show_axis: bool = False,
    add_counts_scaled: bool = False,
    add_counts_unscaled: bool = False,
    show_unscaled: bool = True,
    show_scaled: bool = True,
    show_quantiles: bool = True,
    annotate: bool = True,
):
    if exp_results.num_samples == 0 or sum(exp_results.true_labels) == 0:
        return  # noop

    bin_edges = exp_results.bins
    num_bins = exp_results.num_bins
    all_pass_rate = exp_results.true_labels.mean()
    b_ref = all_pass_rate * (1 - all_pass_rate)
    skill_score = (b_ref - exp_results.brier_score) / b_ref
    prob_true, prob_pred = exp_results.calibration_curve()
    rescaling = "None" if show_unscaled else "Platt"

    ax.plot([0, 1], [0, 1], color="black", linestyle="--")

    scaled_exp = exp_results.to_platt_scaled()

    if show_unscaled:
        unscaled_bars = ax.bar(
            bin_edges[:-1] + (0.5 / num_bins),
            prob_true,
            width=1 / num_bins,
            align="center",
            alpha=0.5,
            # yerr=confidence_intervals,
            # set edge color to black
            edgecolor="k",
            color="C0",
        )

        skill_score = (b_ref - exp_results.brier_score) / b_ref
        text_annotations = "\n".join(
            [
                f"Rescaling: {rescaling}",
                f"All Pass @1: {all_pass_rate:.0%}",
                "$\\mathcal{B}_{ref}$" + f": {b_ref:.2f}",
                f"$ECE$: {exp_results.ece:.2f}",
                "$\\mathcal{B}$" + f": {exp_results.brier_score:.2f}",
                f"$SS$: {skill_score:.2f}",
            ],
        )

        if annotate:
            ax.text(
                0.02,
                0.98,
                text_annotations,
                verticalalignment="top",
                horizontalalignment="left",
                transform=ax.transAxes,
            )
    if show_scaled:
        scale_prob_true, scale_prob_pred = scaled_exp.calibration_curve()
        scaled_bars = ax.bar(
            bin_edges[:-1] + (0.5 / num_bins),
            scale_prob_true,
            width=1 / num_bins,
            align="center",
            alpha=0.5,
            # set edge color to black
            edgecolor="k",
            color="C1",
        )
        skill_score = (b_ref - scaled_exp.brier_score) / b_ref
        text_annotations = "\n".join(
            [
                f"Rescaling: {rescaling}",
                f"All Pass @1: {all_pass_rate:.0%}",
                "$\\mathcal{B}_{ref}$" + f": {b_ref:.2f}",
                f"$ECE$: {scaled_exp.ece:.2f}",
                "$\\mathcal{B}$" + f": {scaled_exp.brier_score:.2f}",
                f"$SS$: {skill_score:.2f}",
            ],
        )

        if annotate:
            ax.text(
                0.02,
                0.98,
                text_annotations,
                verticalalignment="top",
                horizontalalignment="left",
                transform=ax.transAxes,
            )
    if show_quantiles:
        quant_true, quant_prob = sklearn.calibration.calibration_curve(
            scaled_exp.true_labels,
            scaled_exp.predicted_probabilities,
            n_bins=5,
            strategy="quantile",
        )
        ax.plot(
            quant_prob,
            quant_true,
            # make it wider
            linewidth=3,
            # Make it red
            color="red",
            # make each point be a circle
            marker="o",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # Remove tick numbers
    if not show_axis:
        ax.set_xticks([])
        ax.set_yticks([])

    if add_counts_scaled:
        num_samples = scaled_exp.samples_per_bin
        for i, bar in enumerate(scaled_bars):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                # bar.get_height() + 0.02,
                0.06,
                str(num_samples[i]),
                ha="center",
                va="bottom",
                fontsize=10,
            )
    if add_counts_unscaled and show_unscaled:
        num_samples = exp_results.samples_per_bin
        for i, bar in enumerate(unscaled_bars):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                # bar.get_height() + 0.02,
                0.06,
                str(num_samples[i]),
                ha="center",
                va="bottom",
                fontsize=10,
            )


def full_extent(ax, pad=0.0):
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels()
    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])
    return bbox.expanded(1.0 + pad, 1.0 + pad)


def reliability_single(exp_result, fig, ax, show_unscaled=False):
    ax.set_xlabel("P(estimate)")
    ax.set_ylabel("P(correct)")
    postage_bar(
        exp_result,
        ax,
        show_axis=True,
        # add_counts_scaled=True,
        show_unscaled=show_unscaled,
        show_scaled=not show_unscaled,
        add_counts_scaled=not show_unscaled,
        add_counts_unscaled=show_unscaled,
        show_quantiles=True,
    )
    # Add counts


def roc_plot_single(
    exp_results: ExperimentResults,
    key: str,
    fig: Figure,
    ax: Axes,
):
    if exp_results.num_samples == 0:
        return
    scaled_exp = exp_results.to_platt_scaled()
    acc = scaled_exp.true_labels
    score = scaled_exp.predicted_probabilities
    # make ROC
    auc = exp_results.roc_auc
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(acc, score)
    ax.plot(fpr, tpr, label=key)
    ax.text(
        0.02,
        0.98,
        f"AUC: {auc:.2f}",
        verticalalignment="top",
        horizontalalignment="left",
        transform=ax.transAxes,
    )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
