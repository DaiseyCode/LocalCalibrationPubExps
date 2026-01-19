import dataclasses
from typing import Optional, Generic, TypeVar
import os
from calipy.calibrate import PlattCalibrator
from calipy.experiment_results import ExperimentResults

from calipy.calibrate import PlattCalibrator
from contigent import compute_token_keep_correlations, visualize_token_correlations
import colorama
import time
from pprint import pprint
from typing import Literal
from dataclasses import dataclass
from typing import Sequence, Optional

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from lmwrapper.openai_wrapper import get_open_ai_lm, OpenAiModelNames
import math
import difflib

from lmwrapper.structs import LmPrompt, LmPrediction
from synthegrator.code_problems import LmCodeSolution
from synthegrator.code_solver import DummyCodeSolverAutoRegressive
from synthegrator.execution_threading import solve_and_evaluate_problems
from synthegrator.lang_specs.lang_spec_python import PythonLangSpec
from synthegrator.solution_eval import SolutionEvaluation, evaluate_code_problem_execution

from calipy.experiment_results import ExperimentResults
from fixsolver import RewriteFixSolver, solution_to_repair_problem
#from localizing.manual_label_maybe import get_some_solves
from protogrator import LmCodeSolutionSet, LmCodeSolverTemp
import transformers
from pathlib import Path
from synthegrator.synthdatasets import DatasetName, DatasetSpec

cur_path = Path(__file__).parent.absolute()

gen_model_name: str = "mistralai/Mistral-7B-v0.1"
#gen_model_name: str = "HuggingFaceTB/SmolLM-135M-Instruct"
#gen_model_name: str = "HuggingFaceTB/SmolLM-135M"
#gen_model_name: str = "HuggingFaceTB/SmolLM-360M"
#gen_model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
#gen_model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"

# Make a tokenizer
#gen_tokenizer = transformers.AutoTokenizer.from_pretrained(gen_model_name)
#lang_spec = PythonLangSpec()


max_gen_tokens = 1000


def main():
    run_multi_explore(
        True,
        DatasetName.humaneval,
        multi_samples=5,
    )
    #single_explore()
    #temperature_sweep()


@dataclass
class FilterStats:
    """Tracks counts at each filtering stage"""
    total_initial: int
    none_metric_count: int  # Evals with no metric
    existing_successes: int  # Already solved correctly
    no_parsed_function: int  # Couldn't parse expected function
    junk_solves: int  # Original solve was junk
    logprobs_not_alignable: int  # Couldn't align logprobs
    no_extracted_tokens: int  # Couldn't extract tokens from fix
    no_valid_multis: int  # No valid multi-samples
    final_used: int  # Final number used in analysis

    def print_filter_pipeline(self):
        """Print the filter pipeline showing each stage"""
        remaining = self.total_initial
        print(f"Initial evaluations: {remaining}")

        print(f"---v Remove {self.none_metric_count} with no metric")
        remaining -= self.none_metric_count
        print(f"   > {remaining}")

        print(f"---v Remove {self.existing_successes} existing successes")
        remaining -= self.existing_successes
        print(f"   > {remaining}")

        print(f"---v Filter no parsed expected function: {self.no_parsed_function}")
        remaining -= self.no_parsed_function
        print(f"   > {remaining}")

        print(f"---v Filter original solve was junk: {self.junk_solves}")
        remaining -= self.junk_solves
        print(f"   > {remaining}")

        print(f"---v Filter logprobs not alignable: {self.logprobs_not_alignable}")
        remaining -= self.logprobs_not_alignable
        print(f"   > {remaining}")

        print(f"---v Filter no extracted tokens in fix: {self.no_extracted_tokens}")
        remaining -= self.no_extracted_tokens
        print(f"   > {remaining}")

        print(f"---v Filter no valid multis: {self.no_valid_multis}")
        remaining -= self.no_valid_multis
        print(f"   > {remaining}")

        assert remaining == self.final_used, f"Expected {self.final_used} but got {remaining}"
        print(f"\nFinal measurements on {self.final_used} problems")


@dataclass
class TokenAnalysis:
    """Analysis of tokens from one problem"""
    old_tokens: list[str]
    old_logprobs: list[float]
    token_keep_tally: np.ndarray  # Shape: (num_tokens,)
    new_tokens: list[str]
    old_keeps: list[int]  # Binary indicators for kept tokens
    problem_id: str


@dataclass
class FilteredAnalysis:
    """Results after filtering and analysis"""
    filter_stats: FilterStats
    token_analyses: list[TokenAnalysis]
    # These are kept for debugging/analysis
    was_fixed: list[tuple[SolutionEvaluation, LmCodeSolutionSet]]
    new_evals: list[SolutionEvaluation]
    existing_successes: list[SolutionEvaluation]
    old_evals_fixed: list[SolutionEvaluation]




@dataclass
class MultiExploreResults:
    """Results from running multi-sample exploration"""
    experiment_results: ExperimentResults
    fix_rate: float  # Fraction of problems fixed
    orig_solve_rate: float  # Original solve rate
    filter_stats: FilterStats  # Detailed filtering statistics
    filtered_data: FilteredAnalysis  # The filtered data for reuse
    dataset: DatasetName
    temperature: float
    multi_samples: int


def get_filtered_data(
    use_eval_plus: bool,
    dataset: DatasetName,
    multi_samples: int = 5,
    temperature: float = 1.0,
    max_problems: Optional[int] = None,
    fix_reference: str = "gpt4"
) -> FilteredAnalysis:
    """
    Run the data collection and filtering pipeline.

    Args:
        use_eval_plus: Whether to use eval plus
        dataset: Which dataset to analyze
        multi_samples: Number of samples per problem
        temperature: Temperature for sampling
        max_problems: Maximum number of problems to analyze
        fix_reference: Reference to use for fixes ("gpt4" or "gt")
    """

    # Get multi-sample evaluations
    multi_evals: list[LmCodeSolutionSet] = get_some_solves(
        model_name=gen_model_name,
        internals=False,
        max_problems=max_problems,
        solves_per_problem=multi_samples,
        temperature=temperature,
        max_gen_tokens=max_gen_tokens,
        use_eval_plus=use_eval_plus,
        dataset=dataset,
        run_eval=False,
    )

    base_and_multis = [
        BaseAndMultis(base_eval, multi_eval)
        for base_eval, multi_eval in zip(base_evals, multi_evals)
    ]

    # Run filtering and analysis
    filtered = filter_and_analyze(base_and_multis, dataset, fix_reference)

    # Print filtering pipeline
    filtered.filter_stats.print_filter_pipeline()

    return filtered


def run_multi_explore(
    use_eval_plus: bool,
    dataset: DatasetName,
    multi_samples: int = 5,
    temperature: float = 1.0,
    max_problems: Optional[int] = None,
    fix_reference: str = "gpt4",
    output_dir: Optional[Path] = None
) -> MultiExploreResults:
    """
    Run the complete multi-sample exploration analysis pipeline.
    """
    # Get filtered data
    filtered = get_filtered_data(
        use_eval_plus=use_eval_plus,
        dataset=dataset,
        multi_samples=multi_samples,
        temperature=temperature,
        max_problems=max_problems,
        fix_reference=fix_reference
    )

    # Compute metrics
    metrics = compute_metrics(filtered)
    all_ests = metrics['estimates']
    all_actuals = metrics['actuals']

    # Calculate rates
    fix_rate = len(filtered.was_fixed) / len(filtered.new_evals) if filtered.new_evals else 0
    orig_solve_rate = len(filtered.existing_successes) / filtered.filter_stats.total_initial

    # Create experiment results with the multisample estimates
    res = ExperimentResults(all_ests, all_actuals)

    results = MultiExploreResults(
        experiment_results=res,
        fix_rate=fix_rate,
        orig_solve_rate=orig_solve_rate,
        filter_stats=filtered.filter_stats,
        filtered_data=filtered,
        dataset=dataset,
        temperature=temperature,
        multi_samples=multi_samples
    )

    # Generate enhanced plots
    plot_reliability_curves(filtered, results, output_dir)

    return results



def compute_metrics(filtered: FilteredAnalysis) -> dict:
    """Compute metrics from filtered analysis results"""
    all_ests = []
    all_actuals = []
    all_old_probs = []

    for analysis in filtered.token_analyses:
        all_ests.extend(analysis.token_keep_tally)
        all_actuals.extend(analysis.old_keeps)
        all_old_probs.extend([
            math.exp(p) for p in analysis.old_logprobs
        ])

    return {
        'estimates': np.array(all_ests),
        'actuals': np.array(all_actuals),
        'old_probs': np.array(all_old_probs)
    }


def filter_and_analyze(
    base_evals: Sequence[SolutionEvaluation],
    multi_evals: Sequence[LmCodeSolutionSet],
    dataset: DatasetName,
    fix_reference: str = "gpt4"
) -> FilteredAnalysis:
    """Main function to filter and analyze evaluations"""
    total = len(base_evals)
    none_count = sum(1 for e in base_evals if e.main_metric is None)

    # Get fixes using existing get_fixes function
    was_fixed, new_evals, existing_successes, old_evals_fixed = get_fixes(
        base_evals, multi_evals, fix_reference
    )

    # Process each evaluation
    token_analyses = []
    filter_counts = {
        "no_parsed": 0,
        "junk": 0,
        "logprobs_unaligned": 0,
        "no_tokens": 0,
        "no_multis": 0
    }

    # Process each fixed solution
    for (fix, multis), old_eval in zip(was_fixed, old_evals_fixed):
        analysis, stage = process_single_evaluation(fix, multis, old_eval, dataset)
        if analysis is None:
            # Use the stage data to determine which filter rejected it
            failure = stage.get_failure_stage()
            if failure:
                filter_counts[failure] += 1
        else:
            token_analyses.append(analysis)

    stats = FilterStats(
        total_initial=total,
        none_metric_count=none_count,
        existing_successes=len(existing_successes),
        no_parsed_function=filter_counts["no_parsed"],
        junk_solves=filter_counts["junk"],
        logprobs_not_alignable=filter_counts["logprobs_unaligned"],
        no_extracted_tokens=filter_counts["no_tokens"],
        no_valid_multis=filter_counts["no_multis"],
        final_used=len(token_analyses)
    )

    return FilteredAnalysis(
        filter_stats=stats,
        token_analyses=token_analyses,
        was_fixed=was_fixed,
        new_evals=new_evals,
        existing_successes=existing_successes,
        old_evals_fixed=old_evals_fixed
    )


def token_and_value(tokens, values):
    for tok, val in zip(tokens, values):
        print(f"{tok}:{val}", end="")


def is_func_body_junk_solve(
    text_func: str,
    tokens: list[str] = None,
    dataset: DatasetSpec | None = None,
) -> bool:
    if dataset.get_base_collection_name() in (
        DatasetName.humaneval.get_base_collection_name(), 
        DatasetName.humaneval_plus.get_base_collection_name(), 
        DatasetName.mbpp.get_base_collection_name(), 
        DatasetName.mbpp_plus.get_base_collection_name(),
        DatasetName.livecodebench.get_base_collection_name(),
        DatasetName.repocod.get_base_collection_name(),
    ):
        frac_lines_comments, num_lines = get_frac_lines_that_are_comments(text_func)
        if num_lines == 0:
            # No lines here
            return True
        if frac_lines_comments > 0.9:
            # Mostly comments
            return True
        elif frac_lines_comments >= 0.5 and num_lines < 4:
            # Short and mostly comments. Likely just a "TODO" comment or something
            return True
        if tokens is not None:
            if len(tokens) > max_gen_tokens * 0.95:
                # It likely just got clipped by the max gen limit
                return True
        return False
    elif dataset.get_base_collection_name() == DatasetName.dypy_line_completion.get_base_collection_name():
        return len(text_func) < 5 or len(tokens) < 2
    elif dataset.get_base_name() in ("NonSynthDataset", "HaluEval"):
        return False
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    

def has_odd_indentation(text_func: str, dataset: DatasetSpec) -> bool:
    if dataset.get_base_collection_name() == DatasetName.repocod.get_base_collection_name():
        indent_levels = [
            len(line) - len(line.lstrip())
            for line in text_func.split("\n")
            if line.strip()
        ]
        if len(indent_levels) > 1 and min(indent_levels) < indent_levels[0]:
            return True
    return False


def single_explore():
    dataset = DatasetName.humaneval
    evals_for_problems = get_some_solves(
        model_name=gen_model_name,
        internals=False,
        max_problems=100,
        solves_per_problem=1,
        max_gen_tokens=max_gen_tokens,
        use_eval_plus=True,
        dataset=dataset,
    )

    lm = get_open_ai_lm(OpenAiModelNames.gpt_4o)
    solver = RewriteFixSolver(lm)
    existing_successes = []
    new_problems = []
    old_evals_needing_fix = []
    for eval in evals_for_problems:
        if eval.main_metric.is_success:
            existing_successes.append(eval)
            continue
        solution = eval.solution
        new_problems.append(solution_to_repair_problem(eval))
        old_evals_needing_fix.append(eval)
    new_evals = list(solve_and_evaluate_problems(solver, new_problems))
    was_fixed = []
    for eval in new_evals:
        if eval.main_metric.is_success:
            was_fixed.append(eval)
    print("Fixed", len(was_fixed), "out of", len(new_evals), f" (originally solved {len(existing_successes)})")
    was_fix_closer_than_gt(was_fixed, dataset)


def diff_tokens(old_tokens, new_tokens, old_logprobs):
    matcher = difflib.SequenceMatcher(None, old_tokens, new_tokens)
    token_rows = []

    # Print unified diff-like output
    print("Unified diff-like output:")
    def print_diff():
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                for token in old_tokens[i1:i2]:
                    print(f"  {token}")
            elif tag == 'replace':
                for token in old_tokens[i1:i2]:
                    print(f"r- {token}")
                for token in new_tokens[j1:j2]:
                    print(f"r+ {token}")
            elif tag == 'delete':
                for token in old_tokens[i1:i2]:
                    print(f"- {token}")
            elif tag == 'insert':
                for token in new_tokens[j1:j2]:
                    print(f"+ {token}")

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        equal_in_diff = tag == "equal"
        if tag == "insert":
            for new_idx, token in zip(new_tokens[j1:j2], range(j1, j2)):
                token_rows.append({
                    "token": token,
                    "logprob": None,
                    "prob": None,
                    "equal_in_diff": equal_in_diff,
                    "tag": tag,
                    "new_idx": new_idx,
                    "old_idx": None,
                })
        elif tag == "replace":
            for old_idx in range(i1, i2):
                token = old_tokens[old_idx]
                logprob = old_logprobs[old_idx]
                token_rows.append({
                    "token": token,
                    "logprob": logprob,
                    "prob": math.exp(logprob) if logprob is not None else None,
                    "equal_in_diff": equal_in_diff,
                    "tag": tag,
                    "old_idx": old_idx,
                    "new_idx": None,
                })
            for new_idx in range(j1, j2):
                token = new_tokens[new_idx]
                token_rows.append({
                    "token": token,
                    "logprob": logprob,
                    "prob": None,
                    "equal_in_diff": equal_in_diff,
                    "tag": tag,
                    "old_idx": None,
                    "new_idx": new_idx,
                })
        elif tag == "equal":
            for old_idx, new_idx in zip(range(i1, i2), range(j1, j2), strict=True):
                token = old_tokens[old_idx]
                logprob = old_logprobs[old_idx]
                token_rows.append({
                    "token": token,
                    "logprob": logprob,
                    "prob": math.exp(logprob) if logprob is not None else None,
                    "equal_in_diff": equal_in_diff,
                    "tag": tag,
                    "old_idx": old_idx,
                    "new_idx": new_idx,
                })
        elif tag == "delete":
            for old_idx in range(i1, i2):
                logprob = old_logprobs[old_idx]
                token_rows.append({
                    "token": old_tokens[old_idx],
                    "logprob": None,
                    "prob": math.exp(logprob) if logprob is not None else None,
                    "equal_in_diff": equal_in_diff,
                    "tag": tag,
                    "old_idx": old_idx,
                    "new_idx": None,
                })
        else:
            raise ValueError(f"Unknown tag {tag}")
    return pd.DataFrame(token_rows)


def get_frac_lines_that_are_comments(text: str):
    lines = text.split("\n")
    num_comment_lines = 0
    num_lines = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            num_comment_lines += 1
        num_lines += 1
    if num_lines == 0:
        return 0, 0
    return num_comment_lines / num_lines, num_lines


def was_fix_closer_than_gt(
    was_fixed_list: list[SolutionEvaluation],
    dataset,
):
    dfs = []
    rows_diff_agg = []
    skipped_funcs = 0
    for fix in was_fixed_list:
        problem_id = fix.solution.problem.problem_id
        old_solve: LmCodeSolution = fix.solution.problem.past_solve_context[0]
        old_text = old_solve.apply().get_only_file().content_str
        old_text_func = solve_to_text(old_solve, dataset)
        if old_text_func is None:
            skipped_funcs += 1
            continue
        old_tokens = tokenize_llama(old_text_func)
        if is_func_body_junk_solve(old_text_func, old_tokens, dataset):
            skipped_funcs += 1
            continue
        old_logprobs = [None] * len(old_tokens)
        #old_tokens = old_solve.lm_prediction.completion_tokens
        #old_logprobs = old_solve.lm_prediction.completion_logprobs

        gt = old_solve.problem.known_solutions[0]
        gt_text = solve_to_text(gt, dataset)
        gt_tokens = tokenize_llama(gt_text)

        new_solve = fix.solution
        new_text = new_solve.apply().get_only_file().content_str
        new_text_func = solve_to_text(new_solve, dataset)
        new_tokens = tokenize_llama(new_text_func)

        print("problem", old_solve.problem.problem_id)
        print("---- old to new")
        print("-- Old")
        print("".join(old_tokens))
        print("-- new")
        print('text func')
        print(new_text_func)
        df_diff_old_new = diff_tokens(old_tokens, new_tokens, old_logprobs)
        print("---- old to gt")
        print("-- gt")
        print(gt_text)
        df_diff_old_gt = diff_tokens(old_tokens, gt_tokens, old_logprobs)
        df_diff_old_gt['problem_id'] = str(old_solve.problem.problem_id)
        df_diff_old_gt['is_gt'] = True
        df_diff_old_new['problem_id'] = str(old_solve.problem.problem_id)
        df_diff_old_new['is_gt'] = False
        dfs.append(df_diff_old_gt)
        dfs.append(df_diff_old_new)

        num_changed_tokens_old_gt = df_diff_old_gt[~df_diff_old_gt['equal_in_diff']].shape[0]
        num_changed_tokens_old_new = df_diff_old_new[~df_diff_old_new['equal_in_diff']].shape[0]
        print("num_changed_tokens_old_gt", num_changed_tokens_old_gt)
        print("num_changed_tokens_old_new", num_changed_tokens_old_new)
        num_equal_tokens_old_gt = df_diff_old_gt[df_diff_old_gt['equal_in_diff']].shape[0]
        num_equal_tokens_old_new = df_diff_old_new[df_diff_old_new['equal_in_diff']].shape[0]
        frac_equal_tokens_old_gt = num_equal_tokens_old_gt / len(old_tokens)
        frac_equal_tokens_old_new = num_equal_tokens_old_new / len(old_tokens)
        print("num_equal_tokens_old_gt", df_diff_old_gt[df_diff_old_gt['equal_in_diff']].shape[0])
        print("num_equal_tokens_old_new", df_diff_old_new[df_diff_old_new['equal_in_diff']].shape[0])
        rows_diff_agg.append({
            'problem_id': str(old_solve.problem.problem_id),
            'num_changed_tokens_old_gt': num_changed_tokens_old_gt,
            'num_changed_tokens_old_new': num_changed_tokens_old_new,
            'less_changed_new_than_gt': num_changed_tokens_old_new < num_changed_tokens_old_gt,
            'num_equal_tokens_old_gt': num_equal_tokens_old_gt,
            'num_equal_tokens_old_new': num_equal_tokens_old_new,
            'frac_equal_tokens_old_gt': frac_equal_tokens_old_gt,
            'frac_equal_tokens_old_new': frac_equal_tokens_old_new,
            'more_equal_new_than_gt': num_equal_tokens_old_new > num_equal_tokens_old_gt,
            'equal_equal_new_than_gt': num_equal_tokens_old_new == num_equal_tokens_old_gt,
            'more_or_equal_equal_new_than_gt': num_equal_tokens_old_new >= num_equal_tokens_old_gt,
            'num_tokens_old': len(old_tokens),
        })

        is_good_print_ex = (
            num_equal_tokens_old_new * 0.5 > num_equal_tokens_old_gt
            and len(old_tokens) < 40
        )
        if is_good_print_ex:
            print("---- old to new")
            print("-- Old")
            print(old_text)
            print("-- new")
            print('text func')
            print(new_text)
            print("---- old to gt")
            print("-- gt")
            print(gt_text)
            #exit()

    #df = pd.concat(dfs)
    ## Sum up the number of same tokens per problem (equal_in_diff == True) for gt and not gt
    #print(df.groupby(['problem_id', 'is_gt'])['equal_in_diff'].sum())

    print("Skipped funcs", skipped_funcs)

    df = pd.DataFrame(rows_diff_agg)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        print(df)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        print(df.mean())
    # Make a plot of num_changed_tokens_old_new vs num_tokens_old and num_changed_tokens_old_gt vs num_tokens_old
    plt.figure(figsize=(7, 5))

    # Create the scatter plot
    sns.scatterplot(data=df, x='num_tokens_old', y='num_changed_tokens_old_new',
                    label='Old->GPT-4', color='blue', alpha=0.6)
    sns.scatterplot(data=df, x='num_tokens_old', y='num_changed_tokens_old_gt',
                    label='Old->GT', color='red', alpha=0.6)

    # Set labels and title
    plt.xlabel('Number of Tokens in Old Solution')
    plt.ylabel('Number of Changed Tokens')
    plt.title('Comparison of Token Changes: Old vs New and Old vs Ground Truth')

    median_old_new = df['num_changed_tokens_old_new'].median()
    median_old_gt = df['num_changed_tokens_old_gt'].median()
    median_tokens_old = df['num_tokens_old'].median()

    plt.scatter([median_tokens_old], [median_old_new], color='blue', marker='X', s=100, label='Median Old->GPT-4')
    plt.scatter([median_tokens_old], [median_old_gt], color='red', marker='X', s=100, label='Median Old->GT')

    plt.legend()  # Update the legend to include the new median points

    # Show the plot
    plt.tight_layout()
    plt.show()

    # New plot for num_equal_tokens
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x='num_tokens_old', y='num_equal_tokens_old_new',
                    label='Old->GPT-4', color='blue', alpha=0.6)
    sns.scatterplot(data=df, x='num_tokens_old', y='num_equal_tokens_old_gt',
                    label='Old->GT', color='red', alpha=0.6)
    plt.xlabel('Number of Tokens in Old Solution')
    plt.ylabel('Number of Equal Tokens')
    plt.title('Comparison of Equal Tokens: Old vs New and Old vs Ground Truth')
    median_equal_old_new = df['num_equal_tokens_old_new'].median()
    median_equal_old_gt = df['num_equal_tokens_old_gt'].median()
    plt.scatter([median_tokens_old], [median_equal_old_new],
                color='blue', marker='X', s=100, label='Median Old->GPT-4')
    plt.scatter([median_tokens_old], [median_equal_old_gt],
                color='red', marker='X', s=100, label='Median Old->GT')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_reliability_curves(
    filtered_analysis: FilteredAnalysis,
    results: MultiExploreResults,
    output_dir: Optional[Path] = None
):
    """Generate enhanced reliability curve plots for the results"""
    if output_dir is None:
        output_dir = Path(__file__).parent / "multi_plots"
    output_dir.mkdir(exist_ok=True)

    # Get the metrics
    metrics = compute_metrics(filtered_analysis)
    all_ests = metrics['estimates']
    all_actuals = metrics['actuals']
    all_old_probs = metrics['old_probs']

    # Plot both logprobs and multisample curves
    for use_est, method in (
        (all_old_probs, "lp"),
        (all_ests, "Multisample"),
    ):
        use_est = np.array(use_est, dtype=np.float64)
        assert not np.isnan(use_est).any()

        # Plot unscaled version
        fig, axs = plt.subplots(1, 1, figsize=(5, 5))
        res = ExperimentResults(use_est, all_actuals)
        res.reliability_plot(
            ax=axs,
            show_quantiles="Unscaled" if method != "lp" else False,
            annotate="Unscaled",
            show_counts="Unscaled",
        )

        if method == "Multisample":
            title_text = f"{method} {results.dataset} t={results.temperature} {gen_model_name.split('/')[-1]} ms={results.multi_samples} Unscaled"
        else:
            title_text = f"{method} {results.dataset} {gen_model_name.split('/')[-1]} Unscaled"
        axs.title.set_text(title_text)
        plt.savefig(output_dir / f"{title_text}.png")
        plt.show()

        # Plot scaled version
        fig, axs = plt.subplots(1, 1, figsize=(5, 5))
        res.reliability_plot(
            ax=axs,
            show_scaled=True,
            show_unscaled=False,
            show_counts="Scaled",
            show_quantiles="Scaled",
            annotate="Scaled",
        )

        if method == "Multisample":
            title_text = f"{method} {results.dataset} t={results.temperature} {gen_model_name.split('/')[-1]} ms={results.multi_samples} Scaled"
        else:
            title_text = f"{method} {results.dataset} {gen_model_name.split('/')[-1]} Scaled"
        axs.title.set_text(title_text)
        plt.savefig(output_dir / f"{title_text}.png")
        plt.show()

        # Fit and apply rescaler
        rescaler = PlattCalibrator(
            log_odds=True,
            fit_intercept=True,
        )
        rescaler.fit(use_est, all_actuals)

        # Visualize individual token probabilities
        for analysis in filtered_analysis.token_analyses:
            print("vis")
            print(analysis.old_tokens)
            visualize_probs(
                analysis.old_tokens,
                rescaler.predict(analysis.token_keep_tally),
            )
            print("--")
            visualize_probs(
                analysis.old_tokens,
                rescaler.predict(analysis.token_keep_tally),
                analysis.old_keeps,
            )
            print("new")
            print("".join(analysis.new_tokens))




def temperature_sweep():
    datas = []
    for dataset in (
        DatasetName.humaneval,
        DatasetName.mbpp,
        DatasetName.dypy_line_completion,
    ):
        for multi_samples in (10, 50):#(10, 50):
            for temperature in (1.0, 0.7, 0.3):#(1.0, 0.7):
                try:
                    (
                        res, fix_rate, orig_solve_rate,
                        old_junk_rate, multi_bust_rate, final_multi_used
                    ) = multi_explore(
                        use_eval_plus=dataset == DatasetName.humaneval,
                        dataset=dataset,
                        multi_temperature=temperature,
                        fix_reference="gpt4" if dataset in (DatasetName.humaneval, DatasetName.mbpp) else "gt",
                        max_problems=None if dataset == DatasetName.humaneval else 500,
                        multi_samples=multi_samples,
                    )
                except Exception as e:
                    print("Exception")
                    print(e)
                    raise
                res_scaled = res.to_platt_scaled()
                datas.append({
                    "dataset": dataset,
                    "fix_rate": fix_rate, # Fraction
                    "orig_solve_rate": orig_solve_rate,
                    "old_junk_rate": old_junk_rate,
                    "multi_bust_rate": multi_bust_rate,
                    "final_multi_used": final_multi_used,
                    "temperature": temperature,
                    "multi_samples": multi_samples,
                    "ece": res.ece,
                    "brier": res.brier_score,
                    "ss": res.skill_score,
                    "base_rate": res.base_rate,
                    "ece_scaled": res_scaled.ece,
                    "brier_scaled": res_scaled.brier_score,
                    "ss_scaled": res_scaled.skill_score,
                    "gen_model_name": gen_model_name,
                })
                pprint(datas[-1])
                unix_time = int(pd.Timestamp.now().timestamp())
                pd.DataFrame(datas).to_csv(cur_path / f"temperature_sweep_{unix_time}.csv")
    df = pd.DataFrame(datas)
    print(df.to_csv())


def longest_common_substring(a: list[str], b: list[str]) -> tuple[int, int, int]:
    """
    Find the longest common substring between two lists of strings.

    Args:
        a (List[str]): First list of strings.
        b (List[str]): Second list of strings.

    Returns:
        Tuple[int, int, int]: (a_start, b_start, size) where
                               a[a_start:a_start + size] == b[b_start:b_start + size]
                               If no common substring is found, returns (-1, -1, 0).
    """
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    match = matcher.find_longest_match(0, len(a), 0, len(b))

    if match.size == 0:
        return (-1, -1, 0)  # No common substring found

    a_start, b_start, size = match.a, match.b, match.size
    return (a_start, b_start, size)


def visualize_probs(
    tokens: list[str],
    probs: list[float],
    gts: list[bool] = None,
):
    """Given a list of tokens and their probabilities, visualize them by
    printing them with color based on the probability."""
    colorama.init()

    if gts is None:
        gts = [None] * len(tokens)

    for token, prob, gt in zip(tokens, probs, gts):
        if gts is not None:
            # Set the background color based on whether the token is a ground truthk
            if gt:
                print(colorama.Back.GREEN, end="")
            else:
                print(colorama.Back.RESET, end="")
        if prob > 0.8:
            print(token, end="")
        else:
            # Map probability to color intensity (0-255)
            # 0.8 -> 0 (white), 0 -> 255 (red)
            color_intensity = int((1 - min(prob / 0.8, 1)) * 255)

            # Create an ANSI color code for the token
            color_code = f"\033[38;2;255;{255 - color_intensity};{255 - color_intensity}m"

            # Print the token with its color
            print(f"{color_code}{token}{colorama.Style.RESET_ALL}", end="")

    # New line at the end
    print(colorama.Style.RESET_ALL)


if __name__ == "__main__":
    visualize_probs(["hello", "world"], [0.1, 0.9])
    unix_time = pd.Timestamp.now().timestamp()
    print(int(unix_time))
    print("sdf")
    main()