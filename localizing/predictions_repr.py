from dataclasses import dataclass
from typing import Literal
from calipy.experiment_results import ExperimentResults
import numpy as np
import pickle
from pathlib import Path
from localizing.localizing_structs import TokenizedLocalization
from localizing.probe.agg_models.agg_config import ProbeConfig


@dataclass
class ProblemPredictions:
    problem_id: str
    token_prediction_raw_probs: np.ndarray
    line_prediction_raw_probs: np.ndarray
    prob_level_pred: float
    
    # Labels stored directly instead of derived from localization
    token_labels: np.ndarray  # Ground truth binary labels for each token
    line_labels: np.ndarray   # Ground truth binary labels for each line
    problem_label: float      # Ground truth binary label for the problem (0 or 1)

    def get_token_prediction_label(self) -> np.ndarray:
        return self.token_labels
    
    def get_line_prediction_label(self) -> np.ndarray:
        return self.line_labels
    
    def get_problem_prediction_label(self) -> float:
        return self.problem_label

    def __post_init__(self):
        if len(self.token_prediction_raw_probs) != len(self.token_labels):
            raise ValueError(
                f"Token prediction raw probs length ({len(self.token_prediction_raw_probs)}) must match "
                f"token labels length ({len(self.token_labels)})"
            )
        if len(self.line_prediction_raw_probs) != len(self.line_labels):
            raise ValueError(
                f"Line predictions raw probs length ({len(self.line_prediction_raw_probs)}) must match "
                f"line labels length ({len(self.line_labels)})"
            )
        if not isinstance(self.prob_level_pred, float):
            raise ValueError(
                f"Prob level pred must be a float, got {type(self.prob_level_pred)}"
            )
        if not isinstance(self.problem_label, (int, float)):
            raise ValueError(
                f"Problem label must be a number, got {type(self.problem_label)}"
            )
        # Make sure probs are between 0 and 1 (with some tolerance)
        eps = 1e-6
        if self.prob_level_pred < -eps or self.prob_level_pred > 1 + eps:
            raise ValueError(
                f"Prob level pred must be between 0 and 1, got {self.prob_level_pred}"
            )
        if not np.all(self.token_prediction_raw_probs >= -eps) or not np.all(self.token_prediction_raw_probs <= 1 + eps):
            raise ValueError(
                f"Token prediction raw probs must be between 0 and 1, got {self.token_prediction_raw_probs}"
            )
        if not np.all(self.line_prediction_raw_probs >= -eps) or not np.all(self.line_prediction_raw_probs <= 1 + eps):
            raise ValueError(
                f"Line prediction raw probs must be between 0 and 1, got {self.line_prediction_raw_probs}"
            )
        # Make sure labels are binary (with some tolerance for floats)
        if not np.all(np.isin(self.token_labels, [0, 1])):
            raise ValueError(
                f"Token labels must be binary (0 or 1), got {np.unique(self.token_labels)}"
            )
        if not np.all(np.isin(self.line_labels, [0, 1])):
            raise ValueError(
                f"Line labels must be binary (0 or 1), got {np.unique(self.line_labels)}"
            )
        if self.problem_label not in [0, 1, 0.0, 1.0]:
            raise ValueError(
                f"Problem label must be 0 or 1, got {self.problem_label}"
            )


def get_combined_vector_from_problem_preds(
    preds: list[ProblemPredictions],
    level: Literal["token", "line", "problem"],
    use_gt: bool,
) -> np.ndarray:
    """Get a flat vector for a given label. Gets either the predictions or the ground truth labels."""
    if not preds:
        return np.array([])
    
    vectors = []
    
    for pred in preds:
        if level == "token":
            if use_gt:
                vector = pred.get_token_prediction_label()
            else:
                vector = pred.token_prediction_raw_probs
        elif level == "line":
            if use_gt:
                vector = pred.get_line_prediction_label()
            else:
                vector = pred.line_prediction_raw_probs
        elif level == "problem":
            if use_gt:
                vector = np.array([pred.get_problem_prediction_label()])
            else:
                vector = np.array([pred.prob_level_pred])
        else:
            raise ValueError(f"Invalid level: {level}. Must be 'token', 'line', or 'problem'")
        
        vectors.append(vector)
    
    # Concatenate all vectors into a single flat array
    vec = np.concatenate(vectors)
    # Add eps if <0.5 and subtract eps if >0.5, but only for predictions (not ground truth labels)
    # Use epsilon compatible with 4-decimal rounding in ExperimentResults
    if not use_gt:
        vec = np.where(vec < 0.5, vec + 0.001, vec - 0.001)
    return vec


@dataclass
class FoldResultsPreds:
    config: ProbeConfig
    test_preds_each_fold: list[list[ProblemPredictions]]
    train_preds_each_fold: list[list[ProblemPredictions]]
    fold_names: list[str] = None


def calc_exp_results_per_fold(
    fold_results_preds: FoldResultsPreds,
    level: Literal["token", "line", "problem"],
    scaled_from_train: bool = False,
) -> list[ExperimentResults]:
    """Calculate experiment results for each fold from FoldResultsPreds."""
    results = []
    if scaled_from_train:
        raise NotImplementedError("Scaled from train not implemented")
    for fold_preds in fold_results_preds.test_preds_each_fold:
        exp_results = ExperimentResults(
            predicted_probabilities=get_combined_vector_from_problem_preds(fold_preds, level, use_gt=False),
            true_labels=get_combined_vector_from_problem_preds(fold_preds, level, use_gt=True),
        )
        results.append(exp_results)
    return results


def serialize_fold_results_preds(
    fold_results_preds: FoldResultsPreds,
    path: str,
) -> None:
    filename = Path(path)
    filename.parent.mkdir(parents=True, exist_ok=True)
    if str(filename).endswith('.lz4'):
        import lz4.frame
        with lz4.frame.open(filename, "wb") as f:
            pickle.dump(fold_results_preds, f, protocol=5)
    else:
        with open(filename, "wb") as f:
            pickle.dump(fold_results_preds, f, protocol=5)


def deserialize_fold_results_preds(
    path: str,
) -> FoldResultsPreds | None:
    filename = Path(path)
    if not filename.exists():
        return None
    if str(filename).endswith('.lz4'):
        import lz4.frame
        with lz4.frame.open(filename, "rb") as f:
            return pickle.load(f)
    else:
        with open(filename, "rb") as f:
            return pickle.load(f)

    

BSS = "BSS $\\uparrow$"
ECE = "ECE $\\downarrow$"
AUC = "AUC $\\uparrow$"

def make_metrics_from_fold_results_for_level(
    results: FoldResultsPreds,
    level: Literal["token", "line", "problem"] = "line",
) -> dict[str, dict[str, str]]:
    def avg_metric(vals) -> str:
        return f"{sum(vals) / len(vals):.2f}"
    metrics = {
        "Unscaled": {
            BSS: avg_metric([r.skill_score for r in calc_exp_results_per_fold(results, level)]),
            ECE: avg_metric([r.ece for r in calc_exp_results_per_fold(results, level)]),
            AUC: avg_metric([r.roc_auc for r in calc_exp_results_per_fold(results, level)]),
        },
        "Scaled": {
            BSS: avg_metric([r.to_platt_scaled().skill_score for r in calc_exp_results_per_fold(results, level)]),
            ECE: avg_metric([r.to_platt_scaled().ece for r in calc_exp_results_per_fold(results, level)]),
        },
    }
    return metrics


def make_metrics_from_fold_results_for_level_not_averaged(
    results: FoldResultsPreds,
    level: Literal["token", "line", "problem"] = "line",
) -> dict[str, dict[str, str]]:
    def proc_metric(val) -> str:
        return f"{val:.2f}"
    exp_results = calc_exp_results_per_fold(results, level)
    metrics = []
    for r, fold_name in zip(exp_results, results.fold_names):
        scaled = r.to_platt_scaled()
        if fold_name.startswith("EvalDataset__"):
            fold_name = fold_name.split("EvalDataset__")[1]
        metrics.append({
            "Fold": fold_name,
            "Unscaled": {
                BSS: proc_metric(r.skill_score),
                ECE: proc_metric(r.ece),
                AUC: proc_metric(r.roc_auc),
            },
            "Scaled": {
                BSS: proc_metric(scaled.skill_score),
                ECE: proc_metric(scaled.ece),
            },
        })
    return metrics


def make_metrics_from_fold_results_multilevel(
    results: FoldResultsPreds,
    include_problem_level: bool = False,
) -> dict[str, dict[str, str]]:
    out = {
        "Line-Level": make_metrics_from_fold_results_for_level(results, "line"),
        "Token-Level": make_metrics_from_fold_results_for_level(results, "token"),
    }
    if include_problem_level:
        out["Problem-Level"] = make_metrics_from_fold_results_for_level(results, "problem")
    return out


def flatten_dict(d: dict, prefix: str = "") -> dict:
    """Flatten a dict prepending on the outer key"""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten_dict(v, f"{prefix}{k}__"))
        else:
            out[f"{prefix}{k}"] = v
    return out