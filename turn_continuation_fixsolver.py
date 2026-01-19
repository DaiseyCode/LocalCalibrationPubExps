"""
A solver for repair that just continues where the last turn of the conversation.
"""
import dataclasses

import math
from collections import Counter
import transformers
import torch
import numpy as np
import difflib

from lmwrapper.abstract_predictor import LmPredictor
import pandas as pd
from lmwrapper.openai_wrapper import get_open_ai_lm, OpenAiModelNames
from lmwrapper.structs import LmPrompt, LmPrediction
from synthegrator.code_problems import CodeProblem, CodeSolution, LmCodeSolution
from synthegrator.code_solver import LmBasedSolver, LmCodeSolverAutoRegressive
from synthegrator.environments import ProjectWorkingDirectory
from synthegrator.execution_threading import solve_and_evaluate_problems
from synthegrator.few_shotting import FewShotConfig
from synthegrator.lm_few_shotting_tools import FewShotRendererWrapper
from synthegrator.problem_rendering_insertion_tags import TaggedEditRenderer, TaggedEditResponseParser
from synthegrator.transformation_spec import TransformationSpec, StsSpecStatement, StsEditable, SolveStep, \
    markup_path, iterate_parser_states_for_node
from synthegrator.response_parser import _pull_first_md_block_from_completion

from calipy.experiment_results import ExperimentResults
from calipy.plotting import postage_bar
from protogrator import CodeSolutionSet, LmCodeSolutionSet
from localizing.multi_data_gathering import get_base_solves
from synthegrator.synthdatasets import DatasetName
from fixsolver import solution_to_repair_problem
from debug_utils import inspect_lm_prediction


class TurnContinuationFixSolver(LmBasedSolver):
    def __init__(
        self,
        model: LmPredictor,
        include_lm_response: bool = False,
        few_shot_config: FewShotConfig = None,
    ):
        super().__init__(model, include_lm_response)
    

    def solve(
        self,
        problem: CodeProblem,
        randomizing_seed: int | None = None,
        temperature: float = 0,
        num_solutions: int = 1,
    ) -> LmCodeSolution | LmCodeSolutionSet:
        prompt = render_continuation_fix(problem)
        if num_solutions > 1:
            prompt = dataclasses.replace(
                prompt,
                num_completions=num_solutions,
            )


def render_continuation_fix(problem: CodeProblem) -> LmPrompt:
    assert len(problem.past_solve_context) == 1
    old_solve = problem.past_solve_context[0]
    assert isinstance(old_solve, LmCodeSolution)
    old_pred = old_solve.lm_prediction[0]
    assert isinstance(old_pred, LmPrediction)
    pred = old_pred.make_reply_prompt()


def _render_continuation_str(
    problem: CodeProblem
) -> str:
    text = ["There is an error in the given solution."]
    text.append(
        "Please try again to make a fix. "
        "Try to make the new fixed version a minimal change to the old version."
    )


def main():
    localizations = get_base_solves(
        use_eval_plus=True,
        dataset=DatasetName.livecodebench,
        gen_model_name=OpenAiModelNames.gpt_4o_mini,
        filter_to_original_fails=True,
        max_problems=20,
        max_gen_tokens=1000,
    )
    print(f"base solves {localizations}")
    fix_solver = TurnContinuationFixSolver(
        model=get_open_ai_lm(OpenAiModelNames.gpt_4o),
    )
    for localization in localizations.iter_passed_filtered():
        print(localization)
        fix_problem = solution_to_repair_problem(localization.base_eval)
        sol = fix_solver.solve(fix_problem)
        print(sol)
        exit()


if __name__ == "__main__":
    main()