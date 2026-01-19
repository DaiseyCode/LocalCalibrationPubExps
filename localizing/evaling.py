from typing import Callable

from lmwrapper.openai_wrapper import OpenAiModelNames
from numpy.typing import NDArray
from synthegrator.few_shotting import FewShotLibrary
from synthegrator.synthdatasets.dypybench import yield_dypybench

from calipy.calibrate import PlattCalibrator
from calipy.experiment_results import ExperimentResults
import pandas as pd

from localizing.line_vis import get_for_all_eg_line, print_vis_line_localization
from localizing.localizing_structs import LocalizationList, MultisampleTokenEqualsLocalization, \
    BaseLocalization, MultiSamplingConfig, MultiSampleMode
import numpy as np
import matplotlib.pyplot as plt

from localizing.multi_data_gathering import multis_equals_from_scratch, filter_localization_by_min_lines, \
    FIX_REFERENCE_GT
from synthegrator.synthdatasets import DatasetName
from plot_manager import plot_and_save, plot

T_PROB_ACCESSOR = Callable[[BaseLocalization], NDArray[np.float64]]


def token_equals_to_exp_token_agg(
    localizations: LocalizationList[BaseLocalization],
    rescaller: PlattCalibrator = None,
    pred_accessor: T_PROB_ACCESSOR = lambda loc: loc.estimated_keeps,
    gt_accessor: T_PROB_ACCESSOR = lambda loc: loc.gt_base_token_keeps,
) -> ExperimentResults:
    """Mixes all the tokens together into one calibration experiment"""
    use_est = []
    all_actuals = []
    for loc in localizations.iter_passed_filtered():
        vals = pred_accessor(loc)
        if rescaller is not None:
            vals = rescaller.predict(vals)
        gt = gt_accessor(loc)
        use_est.extend(vals)
        all_actuals.extend(gt)
    return ExperimentResults(
        np.array(use_est, np.float64),
        all_actuals
    )


def make_rescale_and_baseline(
    localizations: LocalizationList[BaseLocalization],
    pred_accessor: T_PROB_ACCESSOR = lambda loc: loc.estimated_keeps,
    gt_accessor: T_PROB_ACCESSOR = lambda loc: loc.gt_base_token_keeps,
):
    all_ests = []
    all_gts = []
    for loc in localizations.iter_passed_filtered():
        exp = token_equals_to_exp_token_agg(
            LocalizationList([loc]), None, pred_accessor, gt_accessor)
        # Take a random sample of the predicted_probabilities and true
        combined_vec = np.stack([exp.predicted_probabilities, exp.true_labels], axis=1)
        np.random.shuffle(combined_vec)
        combined_vec = combined_vec[:min(20, len(combined_vec))]
        all_ests.extend(combined_vec[:, 0])
        all_gts.extend(combined_vec[:, 1])
    rescaler = PlattCalibrator(
        log_odds=True,
        fit_intercept=True,
    )
    all_ests, all_gts = np.array(all_ests), np.array(all_gts)
    rescaler.fit(all_ests, all_gts)
    base_rate = all_gts.mean()
    return rescaler, base_rate


def get_problem_exps(
    localizations: LocalizationList[BaseLocalization],
    pred_accessor: T_PROB_ACCESSOR = lambda loc: loc.estimated_keeps,
    gt_accessor: T_PROB_ACCESSOR = lambda loc: loc.gt_base_token_keeps,
) -> pd.DataFrame:
    vals = []
    rescaler, base_rate = make_rescale_and_baseline(
        localizations, pred_accessor, gt_accessor)
    for loc in localizations.iter_passed_filtered():
        exp = token_equals_to_exp_token_agg(
            LocalizationList([loc]), None, pred_accessor, gt_accessor)
        exp_rescale = token_equals_to_exp_token_agg(
            LocalizationList([loc]), rescaler, pred_accessor, gt_accessor)
        exp_baseline = ExperimentResults(
            np.repeat(base_rate, len(exp.predicted_probabilities)),
            exp.true_labels
        )
        if len(exp.predicted_probabilities) == 0:
            print("NO PRED")
            print("bas")
            print(loc.get_base_text())
            print("gt")
            print(loc.get_gt_fix_text())
            continue
        reference_brier = exp_baseline.brier_score
        vals.append({
            #"brier": exp.brier_score,
            #"ece": exp.ece,
            #"ss": (reference_brier - exp.brier_score) / reference_brier,
            "s_brier": exp_rescale.brier_score,
            "s_ece": exp_rescale.ece,
            "s_ss": (reference_brier - exp_rescale.brier_score) / reference_brier,
            #"auroc": exp.roc_auc
        })
    return pd.DataFrame(vals)


def plot_exp(exp: ExperimentResults, title: str = "Reliability", context=None):
    """
    Plot experiment results with reliability, ROC, and PR curves
    
    Args:
        exp: The experiment results
        title: Plot title
        context: Optional dict with context info for directory structure (dataset, model, etc.)
    """
    # Default context if none provided
    if context is None:
        context = {}
    
    # Create directory components based on available context
    dir_components = ["evaling","reliability_plots"]
    if "dataset" in context:
        dir_components.append(str(context["dataset"]))
    if "model" in context:
        dir_components.append(str(context["model"]))
    if "mode" in context:
        dir_components.append(str(context["mode"]))
    
    # RELIABILITY PLOT
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    exp.reliability_plot(
        show_scaled=True,
        ax=axs,
        show_counts="Scaled",
        show_quantiles="Scaled",
        annotate="Scaled",
    )
    pred = exp.predicted_probabilities
    gt = exp.true_labels
    axs.set_title(title)
    
    # Save reliability plot with simplified properties
    properties = {
        "type": "reliability"
    }
    plot_and_save(dir_components, title, properties)
    
    # ROC CURVE
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(gt, pred)
    roc_auc = np.trapz(tpr, fpr)
    axs.plot(fpr, tpr)
    axs.set_xlabel("False Positive Rate")
    axs.set_ylabel("True Positive Rate")
    # add text for AUC
    axs.text(0.6, 0.2, f"AUC = {roc_auc:.2f}")
    roc_title = f"{title} ROC"
    axs.set_title(roc_title)
    
    # Save ROC plot
    roc_dir_components = dir_components.copy()
    properties = {
        "type": "roc"
    }
    plot_and_save(roc_dir_components, roc_title, properties)
    
    # PR CURVE
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(gt, pred)
    axs.plot(recall, precision)
    axs.set_xlabel("Recall")
    axs.set_ylabel("Precision")
    pr_title = f"{title} PR"
    axs.set_title(pr_title)
    
    # Save PR curve
    pr_dir_components = dir_components.copy()
    properties = {
        "type": "pr_curve"
    }
    plot_and_save(pr_dir_components, pr_title, properties)


def plot_problem_exps(problem_exps: pd.DataFrame, title="Problem Metrics", context=None):
    """
    Plot problem experiment metrics as a boxplot
    
    Args:
        problem_exps: DataFrame with problem metrics
        title: Plot title
        context: Optional dict with context info for directory structure
    """
    # Default context if none provided
    if context is None:
        context = {}
    
    # Create directory components based on available context
    dir_components = ["evaling", "boxplots"]
    if "dataset" in context:
        dir_components.append(str(context["dataset"]))
    if "model" in context:
        dir_components.append(str(context["model"]))
    if "mode" in context:
        dir_components.append(str(context["mode"]))
    
    # plot a box and whisker plot of the metrics
    import seaborn as sns
    sns.boxplot(data=problem_exps)
    plt.ylim(-0.5, 1.0)
    plt.title(title)
    
    # Save boxplot
    properties = {}
    plot_and_save(dir_components, "boxplot", properties)


def gah_why():
    problems = list(yield_dypybench(max_problems=3))
    print(problems[0].working_directory.files.pretty_print())


def eval_token_level():
    #gah_why()
    #exit()
    # Define variables to ensure consistency between function args and context
    datasets = [DatasetName.humaneval_plus, DatasetName.mbpp_plus, DatasetName.livecodebench]
    gen_model = OpenAiModelNames.gpt_4o_mini
    fix_ref = OpenAiModelNames.o4_mini_2025_04_16
    tokenizer = "Qwen/Qwen2.5-Coder-0.5B"
    sample_mode = MultiSampleMode.from_prompt
    num_samples = 10
    temperature = 1.0
    
    localizations = multis_equals_from_scratch(
        dataset=datasets,
        gen_model_name=gen_model,
        filter_to_original_fails=True,
        fix_reference=fix_ref,
        #fix_reference=FIX_REFERENCE_GT,
        max_problems=1000,
        max_gen_tokens=1000,
        multi_config=MultiSamplingConfig(
            multi_temperature=temperature,
            target_num_samples=num_samples,
            mode=sample_mode,
        ),
        tokenizer_key=tokenizer,
    )
    
    # Context for organizing plots
    context = {
        "model": gen_model,
        "level": "token",
        "tokenizer": tokenizer.split("/")[-1],
        "mode": sample_mode
    }
    
    exp = token_equals_to_exp_token_agg(localizations)
    plot_exp(exp, "Token-level mixed together calibration", context)
    
    problem_exps = get_problem_exps(localizations)
    plot_problem_exps(problem_exps, "Token-level problem metrics", context)


def gen_accessor(name: str):
    return (
        name,
        lambda loc: getattr(loc.get_line_probs(), name),
        lambda loc: getattr(loc.get_gold_line_info(), name),
    )


def eval_line_level():
    vals = []
    filter_to_original_fails = True
    for dataset in (
        [DatasetName.humaneval_plus, DatasetName.mbpp_plus, DatasetName.livecodebench],
        #DatasetName.humaneval_plus,
        #DatasetName.mbpp_plus,
        #DatasetName.livecodebench,
    ):
        for mode in (
            MultiSampleMode.from_prompt,
            MultiSampleMode.repair,
        ):
            localizations = get_for_all_eg_line(
                dataset=dataset,
                fix_reference=OpenAiModelNames.o4_mini_2025_04_16,
                multi_config = MultiSamplingConfig(
                    multi_temperature=1.0,
                    target_num_samples=10,
                    mode=mode,
                ),
                filter_to_original_fails=filter_to_original_fails,
                #max_problems=10,
            )
            print("got localizations")
            localizations = filter_localization_by_min_lines(localizations, 3)
            l_list = list(localizations.iter_passed_filtered())
            #print("---")
            #print("BASE")
            #print("---")
            #print(l_list[0].get_base_text())
            #print("---")
            #print("FIX")
            #print("---")
            #print(l_list[0].get_gt_fix_text())
            #exit()
            all_pred = []
            all_gt = []

            if isinstance(dataset, list):
                dataset_name = "_".join([str(d.display_name)[:min(10, len(str(d.display_name)))] for d in dataset])
            else:
                dataset_name = str(dataset.display_name)

            # Context for plot directory organization
            context = {
                "dataset": dataset_name,
                "mode": mode,
                "level": "line",
                "filter_fails": filter_to_original_fails
            }

            for accessor_name, pred_accessor, gt_accessor in (
                gen_accessor("prob_rewrites"),
                #gen_accessor("prob_deletes"),
                #gen_accessor("prob_insert_before"),
            ):
                exp = token_equals_to_exp_token_agg(
                    localizations,
                    pred_accessor=pred_accessor,
                    gt_accessor=gt_accessor
                )
                exp_scaled = exp.to_platt_scaled()
                filt_fail = "" if filter_to_original_fails else "w/pass"
                title = f"{mode} {accessor_name}{filt_fail} line mixed together"
                
                # Update context with accessor info
                context["accessor"] = accessor_name
                
                # Plot with updated plotting function
                plot_exp(exp, title, context)
                
                print("localizations for problem exps")
                print(localizations)
                problem_exps = get_problem_exps(
                    localizations,
                    pred_accessor=pred_accessor,
                    gt_accessor=gt_accessor
                )
                title = f"{mode} {accessor_name} line problem-agg"
                
                # Plot with updated plotting function
                plot_problem_exps(problem_exps, title, context)
                
                has_val = sum(
                    1 if sum(gt_accessor(loc)) > 0 else 0
                    for loc in localizations.iter_passed_filtered()
                )
                vals.append({
                    "dataset": dataset_name,
                    "mode": mode,
                    "estimate": accessor_name,
                    "filter_to_original_fails": filter_to_original_fails,
                    "s_brier median": problem_exps["s_brier"].median(),
                    "s_ece median": problem_exps["s_ece"].median(),
                    "s_ss median": problem_exps["s_ss"].median(),
                    "mixed_together_s_brier": exp_scaled.brier_score,
                    "mixed_together_s_ece": exp_scaled.ece,
                    "mixed_together_s_ss": exp_scaled.skill_score,
                    "mixed_together_rocauc": exp.roc_auc,
                    #"count": len(list(localizations.iter_passed_filtered())),
                    "prob_count": has_val,
                })
    # Make a nice ascii table
    df = pd.DataFrame(vals)
    print(df)
    df.to_csv("line_results_agg.csv", index=False)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        print(pd.DataFrame(vals).groupby(["estimate", "dataset", "mode"]).mean())
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        print(pd.DataFrame(vals).groupby(["estimate", "mode"]).mean())


if __name__ == "__main__":
    eval_token_level()
    #eval_line_level()