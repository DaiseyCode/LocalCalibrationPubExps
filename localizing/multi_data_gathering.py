from abc import ABC, abstractmethod
from pprint import pprint
from typing import Optional, Literal, cast
from functools import partial
from tqdm import tqdm
import traceback
import numpy as np
import os
import difflib
from lmwrapper.openai_wrapper import get_open_ai_lm, OpenAiModelNames
from synthegrator.code_solver import DummyCodeSolverAutoRegressive
from synthegrator.execution_threading import solve_and_evaluate_problems, solve_all_problems, \
    evaluate_all_solutions
from synthegrator.solution_eval import SolutionEvaluation

from debug_utils import inspect_object
from fixsolver import RewriteFixSolver, solution_to_repair_problem
from localizing.filter_helpers import debug_str_filterables
from localizing.fix_adders import LocalizationsFixAdder, RewriteFixAdder
from localizing.localizing_structs import LocalizationList, TokenEqualsLocalization, \
    MultiSamplingLocalization, MultisampleTokenEqualsLocalization, MultiTokenizedLocalization, \
    TokenizedLocalization, MultiSamplingConfig, BaseLocalization, MultiSampleMode
from localizing.problem_processing import tokenize_text, solve_to_text
from multi_compare import has_odd_indentation, is_func_body_junk_solve
from protogrator import LmCodeSolutionSet, make_solver_partial, CodeSolutionSet
from solve_helpers import clean_weird_problem_id, get_some_solves, get_model_in_mem, get_solutions_problem
from synthegrator.synthdatasets import DatasetName
from typing import Iterable, Callable


def get_base_solves(
    dataset: DatasetName,
    gen_model_name: str,
    filter_to_original_fails: bool,
    max_problems: Optional[int] = None,
    max_gen_tokens: Optional[int] = 1000,
) -> LocalizationList[BaseLocalization]:
    """Starting point for all with just the base solves."""
    base_evals: list[SolutionEvaluation] = get_some_solves(
        model_name=gen_model_name,
        internals=False,
        max_problems=max_problems,
        solves_per_problem=1,
        temperature=0.0,
        max_gen_tokens=max_gen_tokens,
        dataset=dataset,
        run_eval=True,
    )
    empty_localizations = LocalizationList(
        items=[
            BaseLocalization(
                base_solve=eval.solution,
                base_eval=eval,
                dataset_name=dataset,
                gen_model_name=gen_model_name,
                gen_model_properties={
                    "max_gen_tokens": max_gen_tokens,
                }
            )
            for eval in base_evals
        ],
    )
    for loc in empty_localizations.iter_passed_filtered():
        loc.annotate_filter(
            "main_metric_is_none",
            loc.base_eval.main_metric is not None,
        )
    if filter_to_original_fails:
        for loc in empty_localizations.iter_passed_filtered():
            loc.annotate_filter(
                "main_metric_is_already_success",
                not loc.base_eval.main_metric.is_success,
            )
    return empty_localizations


def _get_tokens_from_loc_and_check_valid(
    loc: TokenEqualsLocalization,
    text_property: Literal["base_text", "gt_fix_text"],
    tokenizer_key: str
) -> list[str] | None:
    text = getattr(loc, f"get_{text_property}")()
    if text is None:
        loc.annotate_failed_filter(f"{text_property}_is_none")
        return None
    tokens = tokenize_text(tokenizer_key, text)
    if is_func_body_junk_solve(
        text_func=text,
        tokens=tokens,
        dataset=loc.dataset_name,
    ):
        loc.annotate_failed_filter(
            f"is_{text_property}_func_body_junk")
        return None
    if has_odd_indentation(text, loc.dataset_name):
        loc.annotate_failed_filter(f"is_{text_property}_odd_indentation")
        return None
    return tokens


def create_multis_version(
    localizations: LocalizationList[TokenizedLocalization],
    config: MultiSamplingConfig,
    hack_backup_localizations: LocalizationList[MultiTokenizedLocalization] | None = None,
) -> LocalizationList[MultiSamplingLocalization]:
    if config.mode == MultiSampleMode.from_prompt:
        func = create_multis_version_from_prompt
    elif config.mode == MultiSampleMode.repair:
        func = create_multis_version_repair
    else:
        raise ValueError(f"Unknown mode {config.mode}")
    return func(localizations, config, hack_backup_localizations=hack_backup_localizations)


def create_multis_version_from_prompt(
    localizations: LocalizationList[BaseLocalization],
    config: MultiSamplingConfig,
    max_problems_hint: int = 3000,
    hack_backup_localizations: LocalizationList[MultiTokenizedLocalization] | None = None,
) -> LocalizationList[MultiSamplingLocalization]:
    print("Create multis from prompt")
    for loc in localizations.iter_passed_filtered():
        print("--")
        print(loc.dataset_name)
        print(loc.base_solve.problem.problem_id)
    print("datasets")
    dataset=list(localizations.get_dataset_name_set())
    print(dataset)
    print("problem ids")
    problem_id_subset = [
        loc.base_solve.problem.problem_id
        for loc in localizations.iter_passed_filtered()
    ]
    if hack_backup_localizations is not None:
        problem_id_subset_avail_backup = [
            loc.base_solve.problem.problem_id
            for loc in hack_backup_localizations.iter_passed_filtered()
        ]
        print("available hack backup localizations")
        print(problem_id_subset) 
        problem_id_subset = [
            problem_id
            for problem_id in problem_id_subset
            if problem_id not in problem_id_subset_avail_backup
        ]
        print("Filtered problem ids")
    print(problem_id_subset)
    print("num problem ids", len(problem_id_subset))
    multi_evals: list[LmCodeSolutionSet] = get_some_solves(
        model_name=localizations.get_only_gen_model_name(),
        internals=False,
        max_problems=max_problems_hint,
        solves_per_problem=config.target_num_samples,
        temperature=config.multi_temperature,
        dataset=dataset,
        problem_id_subset=problem_id_subset,
        run_eval=False,
    )
    print("num multi evals", len(multi_evals))
    assert len(multi_evals) >= (len(problem_id_subset) - 5)
    problem_id_to_multi_eval = {
        solve_set.problem.problem_id: solve_set
        for solve_set in multi_evals
    }
    problem_id_to_multi_eval.update({
        clean_weird_problem_id(problem_id): solve_set 
        for problem_id, solve_set in problem_id_to_multi_eval.items()
    })
    if hack_backup_localizations is not None:
        for loc in hack_backup_localizations.iter_passed_filtered():
            if loc.base_solve.problem.problem_id not in problem_id_to_multi_eval:
                problem_id_to_multi_eval[loc.base_solve.problem.problem_id] = loc.samples
                problem_id_to_multi_eval[clean_weird_problem_id(loc.base_solve.problem.problem_id)] = loc.samples
    out = LocalizationList()
    missing_problem_ids = []
    num_assigned_values = 0
    for loc in localizations.iter_all():
        if isinstance(loc, TokenizedLocalization):
            new = MultiTokenizedLocalization.copy_from(loc)
        elif isinstance(loc, BaseLocalization):
            new = MultiSamplingLocalization.copy_from(loc)
        else:
            raise ValueError()
        out.append(new)
        problem_id = loc.base_solve.problem.problem_id
        clean_problem_id = clean_weird_problem_id(problem_id)
        if clean_problem_id not in problem_id_to_multi_eval:
            is_problem_id_in_subset_we_want = (
                clean_problem_id in problem_id_subset
                or problem_id in problem_id_subset
            )
            if not is_problem_id_in_subset_we_want:
                continue
            #if (
            #    loc.check_passed_all_filters()
            #):
            if len(missing_problem_ids) == 0:
                print("full loc")
                pprint(sorted(list(problem_id_to_multi_eval.keys())))
            print("missing_multi_problem_id")
            print(problem_id)
            missing_problem_ids.append(problem_id)
            if len(missing_problem_ids) > 10:
                #raise ValueError("missing_multi_problem_id")
                pass
            new.annotate_failed_filter("missing_multi_problem_id")
            continue
        new.samples = problem_id_to_multi_eval[clean_problem_id]
        new.config = config
        num_assigned_values += 1
    print("missing_problem_ids", missing_problem_ids)
    print("num missing problem ids", len(missing_problem_ids))
    print("num assigned values", num_assigned_values)
    print("num subset problem ids", len(problem_id_subset))
    print(debug_str_filterables(out.iter_all()))
    if not (len(out) == len(localizations)):
        print("WARNING: len(out) != len(localizations)")
        print(f"len(out) {len(out)} != len(localizations) {len(localizations)}")
    return out


def filter_localization_by_min_lines(
    localizations: LocalizationList[BaseLocalization],
    min_lines: int,
) -> LocalizationList[BaseLocalization]:
    new_locs = localizations.copy()
    for loc in new_locs.iter_passed_filtered():
        if len(loc.get_base_text().split("\n")) < min_lines:
            loc.annotate_failed_filter("min_lines")
    return new_locs

def create_multis_version_repair(
    localizations: LocalizationList[BaseLocalization],
    config: MultiSamplingConfig,
) -> LocalizationList[MultiSamplingLocalization]:
    print("Create multis repair")
    lm = get_model_in_mem(localizations.get_only_gen_model_name())
    solver = RewriteFixSolver(lm, assume_to_be_buggy=False)
    solver = make_solver_partial(
        solver,
        num_solutions=config.target_num_samples,
        temperature=config.multi_temperature,
    )
    sols = solve_all_problems(
        solver=solver,
        problems=[
            solution_to_repair_problem(loc.base_eval)
            for loc in localizations.iter_passed_filtered()
        ],
    )
    first_instance = next(localizations.iter_passed_filtered(), None)
    if isinstance(first_instance, TokenizedLocalization):
        new = localizations.copy_with_type_change(MultiTokenizedLocalization)
    elif isinstance(first_instance, BaseLocalization):
        new = localizations.copy_with_type_change(MultiSamplingLocalization)
    else:
        raise ValueError()
    for loc, sol in zip(new.iter_passed_filtered(), sols):
        assert isinstance(sol, CodeSolutionSet), type(sol)
        loc.samples = sol
        loc.config = config
    return new


def create_multis_version_pretokenized(
    localizations: LocalizationList[TokenizedLocalization],
    config: MultiSamplingConfig,
    hacky_backup_localizations: LocalizationList[MultiTokenizedLocalization] | None = None,
) -> LocalizationList[MultiTokenizedLocalization]:
    return create_multis_version(
        localizations=localizations,
        config=config,
        hack_backup_localizations=hacky_backup_localizations,
    )


def _calculate_token_keeps(
    base_tokens: list[str], 
    fix_tokens: list[str],
) -> list[int]:
    """Calculate which tokens from base_tokens are kept in fix_tokens.
    
    Returns a list of 0s and 1s where 1 indicates the token was kept.
    """
    keeps = [0] * len(base_tokens)
    matcher = difflib.SequenceMatcher(None, base_tokens, fix_tokens)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for i in range(i1, i2):
                keeps[i] = 1
    return keeps


def _calculate_token_keep_or_insert_before(
    base_tokens: list[str], 
    fix_tokens: list[str],
) -> list[int]:
    """Calculate which tokens from base_tokens are kept in fix_tokens,
    but mark as 0 if there's an insert before the token.
    
    Returns a list of 0s and 1s where 1 indicates the token was kept
    and there was no insert before it.
    """
    keeps = [0] * len(base_tokens)
    matcher = difflib.SequenceMatcher(None, base_tokens, fix_tokens)
    
    # Track positions that have inserts immediately before them
    has_insert_before = set()
    
    # First pass: identify positions with inserts before them
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "insert":
            # Mark that position i1 has an insert before it
            has_insert_before.add(i1)
    
    # Second pass: mark tokens as kept only if they're equal AND no insert before
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for i in range(i1, i2):
                if i not in has_insert_before:
                    keeps[i] = 1
    
    return keeps


def create_tokenized_localizations(
    localizations: LocalizationList[BaseLocalization],
    tokenizer_key: str | None = None,
) -> LocalizationList[TokenizedLocalization]:
    out = LocalizationList()
    
    for loc in tqdm(
        localizations.iter_all(), 
        desc="Created tokenized localizations", 
        total=len(localizations),
    ):
        new = TokenizedLocalization.copy_from(loc)
        out.append(new)
        if not loc.check_passed_all_filters():
            continue
        if tokenizer_key is None:
            tokenizer_key = loc.gen_model_name
            if tokenizer_key is None:
                raise ValueError("No tokenizer key provided and no gen model name")
        base_tokens = _get_tokens_from_loc_and_check_valid(
            new, "base_text", tokenizer_key)
        if base_tokens is None:
            continue
        gt_fix_tokens = _get_tokens_from_loc_and_check_valid(
            new, "gt_fix_text", tokenizer_key)
        if gt_fix_tokens is None:
            continue
        new.base_tokens = base_tokens
        new.gt_fix_tokens = gt_fix_tokens
        new.tokenizer_key = tokenizer_key
        
        # Calculate which tokens kept in fix
        #new.gt_base_token_keeps = _calculate_token_keeps(new.base_tokens, new.gt_fix_tokens)
        new.gt_base_token_keeps = _calculate_token_keep_or_insert_before(new.base_tokens, new.gt_fix_tokens)
    assert len(out) == len(localizations)
    return out


def create_token_equals_localizations(
    localizations: LocalizationList[MultiTokenizedLocalization],
) -> LocalizationList[MultisampleTokenEqualsLocalization]:
    assert isinstance(localizations, LocalizationList)
    new = localizations.copy_with_type_change(
        new_type=MultisampleTokenEqualsLocalization
    )
    for loc in new.iter_passed_filtered():
        loc.keep_tallys = []
        dataset = loc.dataset_name
        for solution in loc.samples.solutions:
            multi_text = solve_to_text(solution, dataset)
            if multi_text is None:
                continue
            multi_tokens = loc.tokenize_with_same_tokenizer(multi_text)
            if is_func_body_junk_solve(multi_text, multi_tokens, dataset):
                continue
            this_keeps = _calculate_token_keep_or_insert_before(
                loc.base_tokens, multi_tokens)
            loc.keep_tallys.append(this_keeps)
        if len(loc.keep_tallys) == 0:
            loc.annotate_failed_filter("no_valid_samples")
            continue
    # Calculate token probs
    for loc in new.iter_passed_filtered():
        keeps = np.array(loc.keep_tallys)
        loc.estimated_keeps = np.mean(keeps, axis=0)
    return new


def create_tokenized_localizations_from_scratch(
    dataset: DatasetName,
    gen_model_name: str = OpenAiModelNames.gpt_4_1_mini,
    tokenizer_key: str | None = None,
    filter_to_original_fails: bool = True,
    max_problems: int = 1000,
    max_gen_tokens: int = 1000,
    fix_adder: LocalizationsFixAdder = RewriteFixAdder(
        fix_reference=OpenAiModelNames.o4_mini,
    ),
) -> LocalizationList[TokenizedLocalization]:
    localizations = get_base_solves(
        dataset=dataset,
        gen_model_name=gen_model_name,
        filter_to_original_fails=filter_to_original_fails,
        max_problems=max_problems,
        max_gen_tokens=max_gen_tokens,
    )
    print(f"base solves {localizations}")
    localizations = fix_adder.add_fix_data(localizations)
    print(f"after add fix {localizations}")
    localizations = create_tokenized_localizations(
        localizations,
        tokenizer_key=tokenizer_key,
    )
    return localizations


def multis_equals_from_scratch(
    dataset: DatasetName | list[DatasetName] = DatasetName.humaneval,
    gen_model_name="mistralai/Mistral-7B-v0.1",
    fix_reference: str = OpenAiModelNames.o3_mini,
    max_problems: int = 1000,
    max_gen_tokens: int = 1000,
    multi_config: MultiSamplingConfig = MultiSamplingConfig(
        multi_temperature=1.0,
        target_num_samples=20,
        mode=MultiSampleMode.from_prompt,
    ),
    filter_to_original_fails: bool = True,
    tokenizer_key: str | None = None,
) -> LocalizationList[MultisampleTokenEqualsLocalization]:
    all_localizations = None
    if isinstance(dataset, DatasetName):
        datasets = [dataset]
    else:
        datasets = dataset
    for dataset in datasets:
        localizations = create_tokenized_localizations_from_scratch(
            dataset=dataset,
            gen_model_name=gen_model_name,
            filter_to_original_fails=filter_to_original_fails,
            max_problems=max_problems,
            max_gen_tokens=max_gen_tokens,
            fix_reference=fix_reference,
            tokenizer_key=tokenizer_key,
        )
        print(f"after tokenized {localizations}")
        localizations = create_multis_version_pretokenized(
            localizations,
            multi_config,
        )
        print(f"after add multis {localizations}")
        assert isinstance(localizations, LocalizationList)
        localizations = create_token_equals_localizations(
            localizations,
        )
        print(f"final {localizations}")
        if all_localizations is None:
            all_localizations = localizations
        else:
            all_localizations.extend(localizations)
    return all_localizations


if __name__ == "__main__":
    localizations = multis_equals_from_scratch(
        gen_model_name=OpenAiModelNames.gpt_4o_mini,
        fix_reference=OpenAiModelNames.o3_mini,
        multi_config=MultiSamplingConfig(
            multi_temperature=1.0,
            target_num_samples=10,
            mode=MultiSampleMode.repair,
        ),
    )
    #localizations = get_base_solves(
    #    use_eval_plus=True,
    #    dataset=DatasetNames.humaneval,
    #    gen_model_name=OpenAiModelNames.gpt_4o_mini,
    #    filter_to_original_fails=True,
    #)
    print(debug_str_filterables(localizations.iter_all()))
    for loc in localizations.iter_passed_filtered():
        print("Base:")
        print(loc.get_base_text())
        for i, sample in enumerate(loc.samples.solutions):
            print(f"Sample {i}")
            print(solve_to_text(sample, loc.dataset_name))
        print("gt")
        print(loc.get_gt_fix_text())
        print("---")

    for loc in localizations.iter_all():
        if loc.did_fail_filter("main_metric_is_none"):
            print("Fail Base:")
            print(loc.get_base_text())
            print(loc.base_eval.exception)
            print(type(loc.base_eval.exception))
            print(loc.base_eval.exception.__traceback__)
            print(traceback.format_tb(loc.base_eval.exception.__traceback__))
            exit()
            print(loc.base_solve.problem)
            eval = loc.base_solve.problem.preferred_solution_evaluator.evaluate(loc.base_solve)
            print(eval)
            print(eval.main_metric)
            exit()
            solve_and_evaluate_problems()
