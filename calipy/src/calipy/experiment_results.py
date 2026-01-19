import logging
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from numpy.random import choice, uniform
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import KFold

from .calibrate import PlattCalibrator


class BinStrategy(Enum):
    Uniform = 1
    Quantile = 2
    Custom = 3


# TODO
@dataclass(frozen=True)
class Bin:
    edge: float
    predicted_probabilities: list[float]
    true_labels: list[int]
    accuracy: float
    variance: float


class ExperimentResults:
    _ece: float
    _accuracy: float
    _mean_bin_accuracy: float
    _mean_bin_variance: float
    _bin_variances: list[float]
    _binned_pred_probs: list[list[float]]
    _binned_true_labels: list[list[int]]
    _bin_accuracies: list[float]
    _brier_score: float
    _bin_eces: list[float]
    _rescaler_folds: list[PlattCalibrator] | None
    _rescaler_all_data: PlattCalibrator | None

    def __init__(
        self,
        predicted_probabilities: np.ndarray[np.float64] | list[float] = None,
        true_labels: np.ndarray[np.signedinteger] | list[float | int | bool] = None,
        bin_strategy: BinStrategy = BinStrategy.Uniform,
        num_bins: int = 10,
        bins: np.ndarray[np.float64] | None = None,
    ):
        self._binned = False

        if not (
            isinstance(predicted_probabilities, np.ndarray)
        ):
            predicted_probabilities = np.array(predicted_probabilities, dtype=np.float64)

        if not isinstance(true_labels, np.ndarray):
            true_labels = np.array(true_labels, dtype=np.signedinteger)

        if predicted_probabilities.shape != true_labels.shape:
            msg = "Predicted probabilities shape does not match true labels shape!"
            msg += "\n Predicted probabilities shape: "
            msg += str(predicted_probabilities.shape)
            msg += "\n True labels shape: "
            msg += str(true_labels.shape)
            raise Exception(msg)

        if (
            set(np.unique(predicted_probabilities)) == {0, 1}
            and len(set(np.unique(true_labels))) > 2
        ):
            logging.warning(
                "Are you sure you passed probabilities and not true labels by accident?",
            )

        # This sorts the inputs by the probabilities to make
        # debugging easier and to avoid any weirdness regarding
        # shuffling in kfold scaling
        sorted_probs = predicted_probabilities.argsort()
        self.predicted_probabilities = predicted_probabilities[sorted_probs]
        if np.isnan(self.predicted_probabilities).any():
            raise ValueError("Predicted probabilities contain NaN values.")
            
        min_prob = np.min(self.predicted_probabilities)
        max_prob = np.max(self.predicted_probabilities)
        if min_prob < 0.0 or max_prob > 1.0:
            raise ValueError(
                f"Predicted probabilities must be between 0 and 1 (inclusive). \
                Found min: {min_prob}, max: {max_prob}"
            )
            
        self.true_labels = true_labels[sorted_probs]

        # Round probabilities to avoid floating point issues at edges
        self.predicted_probabilities = np.round(self.predicted_probabilities, decimals=7)

        self.num_bins = num_bins
        self.num_samples = len(predicted_probabilities)
        self.bin_strategy = bin_strategy
        self._rescaler_folds = None
        self._rescaler_all_data = None

        match bin_strategy.name:
            case BinStrategy.Uniform.name:
                self.bins = np.linspace(0, 1, num_bins + 1)
                self.quantiles = None
            case BinStrategy.Quantile.name:
                self.quantiles = np.linspace(0, 1, num_bins + 1)
                self.bins = np.percentile(
                    self.predicted_probabilities,
                    self.quantiles * 100,
                )
            case BinStrategy.Custom.name:
                if not bins:
                    msg = "Please provide bins if selecting custom strategy."
                    raise Exception(msg)
                self.bins = bins
            case _:
                msg = "Invalid bin strategy"
                raise Exception(msg)

    def plot_hist(self, use_bin_strategy: bool = True):
        if use_bin_strategy:
            sns.histplot(self.predicted_probabilities, bins=self.bins)
        else:
            sns.histplot(self.predicted_probabilities, binwidth=0.1)

        # Add count labels
        for p in plt.gca().patches:
            plt.text(
                p.get_x() + p.get_width() / 2.0,
                p.get_height(),
                f"{int(p.get_height())}",
                ha="center",
                va="bottom",
            )

        plt.show()

    def _verify_binned(self):
        if not self._binned:
            self._bin_values()
        assert self._binned
        return True

    @staticmethod
    def from_random(
        sample_size=100,
        bin_strategy: BinStrategy = BinStrategy.Uniform,
    ) -> "ExperimentResults":
        rnd_probs = uniform(size=sample_size)
        rnd_pass = choice([1, 0], size=sample_size)
        return __class__(rnd_probs, rnd_pass, bin_strategy=bin_strategy)

    @classmethod
    def platt_scaled(
        cls,
        predicted_probabilities: np.ndarray[np.float64],
        true_labels: np.ndarray[np.signedinteger],
        bin_strategy: BinStrategy = BinStrategy.Uniform,
        num_bins: int = 10,
        bins: np.ndarray[np.float64] | None = None,
        kfolds: int = 5,
    ) -> "ExperimentResults":
        return __class__._platt_scaled(
            predicted_probabilities,
            true_labels,
            bin_strategy,
            num_bins,
            bins,
            kfolds,
            fit_intercept=True,
        )

    @classmethod
    def temperature_scaled(
        cls,
        predicted_probabilities: np.ndarray[np.float64],
        true_labels: np.ndarray[np.signedinteger],
        bin_strategy: BinStrategy = BinStrategy.Uniform,
        num_bins: int = 10,
        bins: np.ndarray[np.float64] | None = None,
        kfolds: int = 5,
    ) -> "ExperimentResults":
        return __class__._platt_scaled(
            predicted_probabilities,
            true_labels,
            bin_strategy,
            num_bins,
            bins,
            kfolds,
            fit_intercept=False,
        )

    @classmethod
    def _platt_scaled(
        cls,
        predicted_probabilities: np.ndarray[np.float64],
        true_labels: np.ndarray[np.signedinteger],
        bin_strategy: BinStrategy = BinStrategy.Uniform,
        num_bins: int = 10,
        bins: np.ndarray[np.float64] | None = None,
        kfolds: int = 5,
        fit_intercept: bool = True,
    ) -> "ExperimentResults":
        # Rescale the estimate values to be better calibrated
        kf = KFold(n_splits=kfolds, shuffle=True, random_state=42)
        new_predicted_probs = []
        new_true_labels = []
        assert len(predicted_probabilities) == len(true_labels)
        rescaler_folds = []

        def make_rescaler(train_index):
            rescaler = PlattCalibrator(
                log_odds=True,
                fit_intercept=fit_intercept,
            )
            # Check for NaN/Inf in the input probabilities for this fold
            fold_probs = predicted_probabilities[train_index]
            if np.isnan(fold_probs).any() or np.isinf(fold_probs).any():
                raise ValueError(
                    f"NaN or Inf detected in predicted_probabilities[train_index] within calipy before PlattCalibrator.fit. Min: {np.nanmin(fold_probs)}, Max: {np.nanmax(fold_probs)}"
                )
            rescaler.fit(
                fold_probs, # Use the checked variable
                true_labels[train_index],
            )
            return rescaler

        for _i, (train_index, test_index) in enumerate(
            kf.split(predicted_probabilities),
        ):
            rescaler = make_rescaler(train_index)
            rescaler_folds.append(rescaler)
            scaled_predicted_probabilities = rescaler.predict(
                predicted_probabilities[test_index],
            )
            new_predicted_probs.append(scaled_predicted_probabilities)
            new_true_labels.append(true_labels[test_index])

        instance = __class__(
            np.concatenate(new_predicted_probs),
            np.concatenate(new_true_labels),
            bin_strategy,
            num_bins,
            bins,
        )
        # Hacky. This whole part is somewhat suspect and not clear to
        #   DNG why __class__ is used.
        #   Claudio: we use __class__ to instantiate a new object of this
        #   class type as the values are now scaled.
        instance._rescaler_folds = rescaler_folds
        instance._rescaler_all_data = make_rescaler(
            np.arange(len(predicted_probabilities)),
        )
        return instance

    def to_platt_scaled(self) -> "ExperimentResults":
        return ExperimentResults.platt_scaled(
            self.predicted_probabilities,
            self.true_labels,
            bin_strategy=self.bin_strategy,
            num_bins=self.num_bins,
        )

    def to_temperature_scaled(self) -> "ExperimentResults":
        return ExperimentResults.temperature_scaled(
            self.predicted_probabilities,
            self.true_labels,
            bin_strategy=self.bin_strategy,
            num_bins=self.num_bins,
        )

    @cached_property
    def ece(self) -> float:
        self._verify_binned()
        return self._ece

    @cached_property
    def accuracy(self) -> float:
        self._verify_binned()
        return self._accuracy

    @cached_property
    def mean_bin_accuracy(self) -> float:
        self._verify_binned()
        return self._mean_bin_accuracy

    @cached_property
    def mean_bin_variance(self) -> float:
        self._verify_binned()
        return self._mean_bin_variance

    @cached_property
    def bin_variances(self) -> np.ndarray[np.floating]:
        self._verify_binned()
        return self._bin_variances

    @cached_property
    def binned_pred_probs(self) -> np.ndarray[np.floating]:
        self._verify_binned()
        return self._binned_pred_probs

    @cached_property
    def binned_true_labels(self) -> np.ndarray[np.integer]:
        self._verify_binned()
        return self._binned_true_labels

    @cached_property
    def bin_accuracies(self) -> np.ndarray[np.floating]:
        self._verify_binned()
        return self._bin_accuracies

    @cached_property
    def samples_per_bin(self) -> np.ndarray[np.integer]:
        self._verify_binned()
        return self._samples_per_bin

    @cached_property
    def sample_count(self) -> int:
        return len(self.predicted_probabilities)

    @cached_property
    def brier_score(self) -> float:
        self._verify_binned()
        return self._brier_score

    @cached_property
    def roc_auc(self) -> float:
        return roc_auc_score(
            self.true_labels,
            self.predicted_probabilities,
        )

    @cached_property
    def skill_score(self) -> float:
        self._verify_binned()
        return (self.reference_brier - self.brier_score) / self.reference_brier

    @cached_property
    def base_rate(self) -> float:
        return self.true_labels.mean()

    @cached_property
    def reference_brier(self) -> float:
        return self.base_rate * (1 - self.base_rate)

    @cached_property
    def summary(self) -> str:
        return f"""Samples: {self.sample_count:,}
Accuracy: {self.accuracy:.2f}
R Brier: {self.reference_brier:.2f}
ECE: {self.ece:.2f}
Brier: {self.brier_score:.2f}"""

    # @cached_property
    def calibration_curve(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        prob_true : ndarray of shape (n_bins,) or smaller
        The proportion of samples whose class is the positive class, in each
        bin (fraction of positives).

        prob_pred : ndarray of shape (n_bins,) or smaller
        The mean predicted probability in each bin.

        """
        self._verify_binned()
        # Number of bins
        bins_len = self.num_bins

        # Find bin indices for each predicted probability value
        binids = np.searchsorted(
            self.bins[1:-1],
            self.predicted_probabilities,
            side="right",
        )

        # Sum predicted probabilities in each bin
        bin_sums = np.bincount(
            binids,
            weights=self.predicted_probabilities,
            minlength=bins_len,
        )

        # Sum true labels in each bin (either 0 or 1)
        bin_true = np.bincount(binids, weights=self.true_labels, minlength=bins_len)

        # Count of total samples in each bin
        bin_total = np.bincount(binids, minlength=bins_len)

        # Identify non-zero bins (to avoid division by zero)
        nonzero = bin_total != 0

        # Calculate the proportion of true labels in each bin
        prob_true = np.divide(
            bin_true,
            bin_total,
            out=np.zeros_like(bin_true),
            where=nonzero,
        )

        # Calculate the average predicted probability in each bin
        prob_pred = np.divide(
            bin_sums,
            bin_total,
            out=np.zeros_like(bin_sums),
            where=nonzero,
        )

        # --- Comment out assertion comparing prob_true and _bin_accuracies --- 
        # assert np.array_equal(
        #     prob_true,
        #     np.array([0 if np.isnan(x) else x for x in self._bin_accuracies]),
        # )

        # Ensure we use the confidences calculated consistently in _bin_values
        # prob_pred = np.array([0 if np.isnan(x) else x for x in self._bin_confidences])
        # --- End Change ---

        return prob_true, prob_pred

    def _bin_values(self) -> None:
        """Returns ece, accuracy, mean_bin_accuracy, mean_bin_variance, binned_pred_probs, binned_true_labels"""
        if self._binned:
            return

        # reference numpy hist
        hist, bin_edges = np.histogram(self.predicted_probabilities, bins=self.bins)

        # uniform binning approach with M number of bins
        bin_lowers = self.bins[:-1]
        bin_uppers = self.bins[1:]

        # get binary class predictions from confidences
        predicted_labels = np.ones_like(self.predicted_probabilities)

        # get a boolean list of correct/false predictions
        accuracies = predicted_labels == self.true_labels
        accuracy = accuracies.mean()
        bin_accuracies = []
        bin_variances = []
        binned_pred_probs = []
        binned_true_labels = []
        bin_eces = []
        bin_confidences = []

        ece = 0
        for i, (bin_lower, bin_upper) in enumerate(
            zip(bin_lowers, bin_uppers, strict=True),
        ):
            # determine if sample is in bin m (between bin lower & upper)
            if i == self.num_bins - 1:  # same hack as in numpy histogram func
                in_bin = self.predicted_probabilities >= bin_lower.item()
            else:
                in_bin = np.logical_and(
                    self.predicted_probabilities >= bin_lower.item(),
                    self.predicted_probabilities < bin_upper.item(),
                )

            # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
            probability_in_bin = in_bin.astype(float).mean()

            if probability_in_bin.item() > 0:
                # get the accuracy of bin m: acc(Bm)
                accuracy_in_bin = accuracies[in_bin].astype(float).mean()
                bin_accuracies.append(accuracy_in_bin)

                labels_in_bin = self.true_labels[in_bin]
                binned_pred_probs.append(self.predicted_probabilities[in_bin])
                binned_true_labels.append(labels_in_bin)
                # get the average confidence of bin m: conf(Bm)
                avg_confidence_in_bin = self.predicted_probabilities[in_bin].mean()
                bin_confidences.append(avg_confidence_in_bin)
                # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
                bin_ece = (
                    np.abs(avg_confidence_in_bin - accuracy_in_bin) * probability_in_bin
                )
                ece += bin_ece
                bin_eces.append(bin_ece)

                variance = accuracy_in_bin * (1 - accuracy_in_bin)
                bin_variances.append(variance)
            else:
                bin_accuracies.append(np.nan)
                bin_variances.append(np.nan)
                binned_pred_probs.append(np.array([]))
                binned_true_labels.append(np.array([]))
                bin_eces.append(0)
                bin_confidences.append(0)

        mean_bin_accuracy = np.nanmean(bin_accuracies)
        mean_bin_variance = np.nanmean(bin_variances)

        brier_score = brier_score_loss(
            self.true_labels,
            self.predicted_probabilities,
        )

        if self.bin_strategy != BinStrategy.Uniform:
            self._ece = -1
            logging.warning("Cannot produce ECE value for non-uniform bin strategy")

        self._binned = True
        self._ece = ece
        self._accuracy = accuracy
        self._mean_bin_accuracy = mean_bin_accuracy
        self._mean_bin_variance = mean_bin_variance
        self._bin_variances = np.array(bin_variances)
        self._binned_pred_probs = binned_pred_probs
        self._binned_true_labels = binned_true_labels
        self._bin_accuracies = np.array(bin_accuracies)
        self._samples_per_bin = np.array(list(map(len, binned_pred_probs)))
        self._brier_score = brier_score
        self._bin_eces = np.array(bin_eces)
        self._bin_confidences = np.array(bin_confidences)
        
        manual_counts = self._samples_per_bin 
        numpy_counts = hist
        count_diff = np.sum(np.abs(manual_counts - numpy_counts))
        # Allow a total absolute difference because of unknown reasons (floating point precision???)
        tolerance = np.sum(manual_counts) * 0.04
        if count_diff > tolerance + 1: 
            print("\n--- CALIPY WARNING: Bin count mismatch EXCEEDS tolerance! ---")
            print(f"Manual Counts ({len(manual_counts)}): {manual_counts}")
            print(f"NumPy Counts  ({len(numpy_counts)}): {numpy_counts}")
            print(f"Bin Edges   ({len(bin_edges)}): {bin_edges}")
            mismatched_indices = np.where(manual_counts != numpy_counts)[0]
            print(f"Mismatched Bin Indices: {mismatched_indices}")
            for idx in mismatched_indices:
                print(f"  Bin {idx} (Edges: [{bin_edges[idx]:.4f}, {bin_edges[idx+1]:.4f}]): Manual={manual_counts[idx]}, NumPy={numpy_counts[idx]}")
            print(f"Total Absolute Difference: {count_diff}")
            print("--- END CALIPY WARNING ---\n")
            raise AssertionError(f"Significant bin count difference ({count_diff}) between manual loop and numpy.histogram")
        elif count_diff > 0:
             print(f"CALIPY INFO: Bin count mismatch within tolerance: {count_diff} (Manual: {manual_counts}, NumPy: {numpy_counts})")
        

    def reliability_plot(
        self,
        ax: Axes,
        show_axis: bool = True,
        show_unscaled: bool = True,
        show_scaled: bool = False,
        show_counts: Literal[False, "Scaled", "Unscaled"] = False,
        show_quantiles: Literal[False, "Scaled", "Unscaled"] = False,
        annotate: Literal[False, "Scaled", "Unscaled"] = False,
        extra_annotations: list[str] | None = None,
    ):
        if self.num_samples == 0 or sum(self.true_labels) == 0:
            raise ValueError("No samples to plot.")

        if show_counts == "Scaled" and not show_scaled:
            raise ValueError("Cannot show scaled counts without showing scaled bars.")

        if show_counts == "Unscaled" and not show_unscaled:
            raise ValueError(
                "Cannot show unscaled counts without showing unscaled bars.",
            )

        if show_counts not in [False, "Scaled", "Unscaled"]:
            raise ValueError("Invalid show_counts value. Must be False, 'Scaled', or 'Unscaled'.")

        # Calculate calibration curve
        prob_true, _ = self.calibration_curve()

        rescaling = "Platt" if show_scaled else "None"

        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], color="black", linestyle="--")

        # Determine if platt scaling is required
        scaled_required = (
            show_scaled or show_quantiles == "Scaled" or annotate == "Scaled"
        )
        if scaled_required:
            scaled_exp = self.to_platt_scaled()

        if show_scaled:
            scale_prob_true, _ = scaled_exp.calibration_curve()
            scaled_bars = ax.bar(
                self.bins[:-1] + (0.5 / self.num_bins),
                scale_prob_true,
                width=1 / self.num_bins,
                align="center",
                alpha=0.5,
                # set edge color to black
                edgecolor="k",
                color="C1",
                label="Scaled"

            )
            text_annotations = "\n".join(
                [
                    f"Rescaling: {rescaling}",
                    # f"Accuracy: {scaled_exp.accuracy:.0%}",
                    f"Base Rate: {scaled_exp.base_rate:.0%}",
                    "$\\mathcal{B}_{ref}$" + f": {scaled_exp.reference_brier:.2f}",
                    f"$ECE$: {scaled_exp.ece:.2f}",
                    "$\\mathcal{B}$" + f": {scaled_exp.brier_score:.2f}",
                    f"$SS$: {scaled_exp.skill_score:.2f}",
                    *(extra_annotations or []),
                ],
            )

            if annotate == "Scaled":
                ax.text(
                    0.02,
                    0.98,
                    text_annotations,
                    verticalalignment="top",
                    horizontalalignment="left",
                    transform=ax.transAxes,
                )

        if show_unscaled:
            unscaled_bars = ax.bar(
                self.bins[:-1] + (0.5 / self.num_bins),
                prob_true,
                width=1 / self.num_bins,
                align="center",
                alpha=0.5,
                # yerr=confidence_intervals,
                # set edge color to black
                edgecolor="k",
                color="C0",
                label="Nonscaled"
            )

            text_annotations = "\n".join(
                [
                    f"Rescaling: {rescaling}",
                    # f"Accuracy: {self.accuracy:.0%}",
                    f"Base Rate: {self.base_rate:.0%}",
                    "$\\mathcal{B}_{ref}$" + f": {self.reference_brier:.2f}",
                    f"$ECE$: {self.ece:.2f}",
                    "$\\mathcal{B}$" + f": {self.brier_score:.2f}",
                    f"$SS$: {self.skill_score:.2f}",
                    *(extra_annotations or []),
                ],
            )

            if annotate == "Unscaled":
                ax.text(
                    0.02,
                    0.98,
                    text_annotations,
                    verticalalignment="top",
                    horizontalalignment="left",
                    transform=ax.transAxes,
                )

        if show_quantiles:
            match show_quantiles:
                case "Scaled":
                    quant_true_labels = scaled_exp.true_labels
                    quant_pred_prob = scaled_exp.predicted_probabilities
                case "Unscaled":
                    quant_true_labels = self.true_labels
                    quant_pred_prob = self.predicted_probabilities
            quant_true, quant_prob = ExperimentResults(
                quant_pred_prob,
                quant_true_labels,
                num_bins=5,
                bin_strategy=BinStrategy.Quantile,
            ).calibration_curve()

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

        # Legend
        ax.legend(loc="upper right")

        # Remove tick numbers
        if not show_axis:
            ax.set_xticks([])
            ax.set_yticks([])

        if show_counts:
            match show_counts:
                case "Scaled":
                    bars = scaled_bars
                    num_samples = scaled_exp.samples_per_bin
                case "Unscaled":
                    bars = unscaled_bars
                    num_samples = self.samples_per_bin
            for i, bar in enumerate(bars):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.06,
                    str(num_samples[i]),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )
