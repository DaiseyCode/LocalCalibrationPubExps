import dataclasses
from synthegrator.solution_eval import SolutionEvaluation, evaluate_code_problem_execution
from synthegrator.transformation_spec import StsSelectorCharSpan
import math
from collections import Counter
import transformers
import torch
import numpy as np
import difflib

from lmwrapper.abstract_predictor import LmPredictor, LmReasoningStyle
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


class RewriteFixSolver(LmBasedSolver):
    def __init__(
        self,
        model: LmPredictor,
        include_lm_response: bool = False,
        few_shot_config: FewShotConfig = None,
        assume_to_be_buggy: bool = True,
    ):
        super().__init__(model, include_lm_response)
        self.assume_to_be_buggy = assume_to_be_buggy


    def solve(
        self,
        problem: CodeProblem,
        randomizing_seed: int | None = None,
        temperature: float = 0,
        num_solutions: int = 1,
    ) -> LmCodeSolution | LmCodeSolutionSet:
        if self.model.reasoning_style == LmReasoningStyle.ALWAYS_THINK:
            temperature = 1.0
        prompt = render_rewrite_prompt(
            problem,
            temperature,
            any(
                sub in self.model.model_name()
                for sub in ("o1", "o3", "r1", "o4")
            ),
            assume_to_be_buggy=self.assume_to_be_buggy,
        )
        if num_solutions > 1:
            prompt = dataclasses.replace(
                prompt,
                num_completions=num_solutions,
            )
        responses = self.model.predict(prompt)

        def resp_to_solve(resp):
            return LmCodeSolution(
                problem=problem,
                solve_steps=parse_rewrite(resp, problem),
                lm_prediction=resp,
                prompt=prompt,
            )

        if num_solutions == 1:
            return resp_to_solve(responses)
        else:
            return LmCodeSolutionSet(
                solutions=[
                    resp_to_solve(resp)
                    for resp in responses
                ],
                preferred_solution=None,
            )


def render_rewrite_str(problem: CodeProblem, assume_to_be_buggy: bool) -> str:
    assert len(problem.past_solve_context) == 1
    old_solve = problem.past_solve_context[0]
    files = old_solve.apply()
    if not assume_to_be_buggy:
        text = ["There is a problem or bugs with the following code."]
    else:
        text = ["There is some code that we are not sure if it is correct or not."]
    only_file = files.get_only_file()
    for path, file in files.walk():
        text.append("```python")
        #text.append("# " + path)
        text.append(file.content_str)
        text.append("```")
    if assume_to_be_buggy:
        text.append(
            "Please fix any bugs with this program and output the complete solution. "
            "Please give just the solution (don't include test cases). "
            "Try to make the fix minimal, changing where needed. "
            "Still return the full solution keeping as much the same possible to fix the bug. "
            "Place your final fixed solution in a markdown code block.")
    else:
        text.append(
            "Please carefully examine the code for bugs. "
            "If there are bugs fix the bugs and output the complete solution. "
            "Please give just the solution (don't include test cases). "
            "Try to make the fix minimal, changing only where needed. "
            "Still return the full solution keeping as much the same possible to fix the bug. "
            "If there are no bugs (which is very possible), just output the original code exactly as it is. "
            "You may reason through whether it is correct and possible bugs, but please place your final fixed solution in a markdown code block with no other markdown blocks except your final answer."
        )
    text = "\n".join(text)
    return text


def render_rewrite_prompt(
    problem: CodeProblem,
    temperature: float,
    is_reasoning_model: bool,
    assume_to_be_buggy: bool,
) -> LmPrompt:
    text = render_rewrite_str(problem, assume_to_be_buggy)
    if assume_to_be_buggy:
        max_tokens = 2000 if not is_reasoning_model else 20_000
    else:
        max_tokens = 5000 if not is_reasoning_model else 20_000
    return LmPrompt(
        text=text,
        max_tokens=max_tokens,
        temperature=temperature if not is_reasoning_model else temperature,
        cache=True,
        logprobs=False,
    )


def parse_rewrite(resp: LmPrediction, problem: CodeProblem) -> list[SolveStep]:
    text = _pull_first_md_block_from_completion(resp)
    path = problem.working_directory.files.get_only_file().name
    markup = markup_path(problem.working_directory, path, spec=problem.transformation_spec)
    mark_id = None
    for state in iterate_parser_states_for_node(markup):
        if state.is_editable:
            mark_id = state.node.mark_id
            break
    else:
        raise ValueError("No editable node found")
    return [
        SolveStep(
            path,
            mark_id,
            text
        )
    ]


def solution_to_repair_problem(
    old_eval: SolutionEvaluation,
    narrow_edit: bool = False,
) -> CodeProblem:
    if not isinstance(old_eval, SolutionEvaluation):
        raise ValueError("old_eval must be a SolutionEvaluation")
    old_solution = old_eval.solution
    wd = ProjectWorkingDirectory(old_solution.apply())
    old_spec = old_solution.problem.transformation_spec
    old_set_base_indent = None
    old_edit_statements = list(old_spec.editing_statements())
    if len(old_edit_statements) == 1:
        old_set_base_indent = old_edit_statements[0].verb.set_base_space_indent

    if not narrow_edit:
        # Make everything editable
        spec = TransformationSpec((
            StsSpecStatement(
                path_selector=None,
                text_selector=None,
                verb=StsEditable()
            ),
        ))
    else:
        stmts = []
        for step in old_solution.solve_steps:
            content = wd.files[step.path].content_str
            if step.value in content:
                stmt = StsSpecStatement(
                    path_selector=step.path,
                    text_selector=StsSelectorCharSpan(
                        start_char=content.index(step.value),
                        end_char=content.index(step.value) + len(step.value),
                    ),
                    verb=StsEditable(
                        set_base_space_indent=old_set_base_indent,
                    ),
                )
                stmts.append(stmt)
        assert len(stmts) > 0, "No editable parts found"
        # TODO: loop through for any editable parts of orig solution
        spec = TransformationSpec(stmts)
    old_instructable_metadata = old_solution.problem.instructable_metadata
    prob = CodeProblem(
        working_directory=wd,
        transformation_spec=spec,
        past_solve_context=[old_solution],
        # Copy over old stuff
        known_solutions=old_solution.problem.known_solutions,
        environment=old_solution.problem.environment,
        dataset_name=old_solution.problem.dataset_name,
        problem_id=old_solution.problem.problem_id,
        noninstructable_metadata=old_solution.problem.noninstructable_metadata,
        test_cases=old_solution.problem.test_cases,
        extension_to_lang_specs=old_solution.problem.extension_to_lang_specs,
        instructable_metadata={
            **(old_solution.problem.instructable_metadata or {}),
            "test_failures": [
                {
                    "name": tr.name,
                    "fail_text": tr.fail_text,
                    "fail_message": tr.fail_message,
                }
                for tr in old_eval.test_results 
                if not tr.success
            ]
        },
    )
    #new_known_solutions = [
    #    dataclasses.replace(sol, problem=prob)
    #    for sol in prob.known_solutions
    #]
    #prob.known_solutions = new_known_solutions
    return prob


top_closed_toks = {
 '\n': 1169,
 ' ': 260,
 '   ': 509,
 '    ': 10,
 '       ': 182,
 '           ': 108,
 '              ': 29,
 '                ': 10,
 #' !=': 7,
 ' "': 24,
 ' "")': 8,
 ' ",': 8,
 ' "__': 6,
 #' #': 127,
 ' %': 14,
 " '": 67,
 " ')": 18,
 ' (': 7,
 ' )': 12,
 #' *': 9,
 #' +': 33,
 #' +=': 19,
 #' -': 29,
 #' -=': 7,
 #' /': 6,
 #' <': 20,
 #' =': 120,
 #' ==': 135,
 #' >': 12,
 #' >=': 15,
 #' False': 22,
 #' True': 17,
 ' [': 39,
 " ['": 11,
 ' []': 40,
 ' __': 15,
 #' and': 12,
 #' del': 9,
 #' elif': 34,
 #' else': 37,
 #' for': 71,
 #' if': 126,
 #' import': 14,
 #' len': 61,
 #' max': 30,
 #' not': 18,
 #' or': 12,
 #' pass': 6,
 #' raise': 8,
 #' range': 23,
 #' str': 16,
 #' sum': 15,
 '")': 19,
 '",': 17,
 '":': 7,
 "'": 11,
 "')": 26,
 "',": 16,
 "':": 27,
 "']": 8,
 '(': 300,
 '("': 32,
 "('": 54,
 '()': 72,
 ')': 131,
 '))': 29,
 ')))': 13,
 '),': 7,
 '):': 19,
 ',': 157,
 '-': 8,
 '--': 23,
 '----------------': 166,
 '.': 204,
 ':': 219,
 #'</s>': 32,
 #'=': 13,
 '[': 113,
 '[-': 18,
 ']': 107,
 '])': 24,
 '],': 10,
 ']:': 13,
 '_': 246,
 '__': 39,
 #'if': 15,
 #'int': 14,
 #'lambda': 14,
 #'len': 19,
 #'list': 10,
 #'open': 10,
 #'reverse': 19,
 #'sort': 21,
 #'str': 10,
}


def main():
    gen_model_name: str = "mistralai/Mistral-7B-v0.1"
    evals_for_problems = get_some_solves(
        model_name=gen_model_name,
        internals=False,
        max_problems=100,
        solves_per_problem=1,
    )
    # Make a tokenizer
    gen_tokenizer = transformers.AutoTokenizer.from_pretrained(gen_model_name)

    #renderer = TaggedEditRenderer()
    #response_parser = TaggedEditResponseParser()
    lm = get_open_ai_lm(OpenAiModelNames.gpt_4o)
    #solver = LmCodeSolverAutoRegressive(
    #    lm,
    #    prompt_renderer=renderer,
    #    response_parser=response_parser,
    #    # snap_indent=4,
    #    include_lm_response=True,
    #    cache_lm=True,
    #)
    solver = RewriteFixSolver(lm)
    existing_successes = []
    new_problems = []
    old_evals_needing_fix = []
    for evals_for_problem in evals_for_problems:
        for eval in evals_for_problem:
            #print(eval.main_metric.is_success)
            if eval.main_metric.is_success:
                existing_successes.append(eval)
                continue
            solution = eval.solution
            #print(render_rewrite_str(solution_to_repair_problem(solution)))
            #print("New")
            new_problems.append(solution_to_repair_problem(solution))
            old_evals_needing_fix.append(eval)
    new_evals = list(solve_and_evaluate_problems(
        solver, new_problems,
        max_threads_solve=4,
        max_threads_eval=4,
    ))
    was_fixed = []
    for eval in new_evals:
        if eval.main_metric.is_success:
            was_fixed.append(eval)
    print("Fixed", len(was_fixed), "out of", len(new_evals), f" (originally solved {len(existing_successes)}")
    token_rows = []
    for eval in new_evals:
        if not eval.main_metric.is_success:
            continue
        old_solve: LmCodeSolution = eval.solution.problem.past_solve_context[0]
        old_text = old_solve.apply().get_only_file().content_str
        old_tokens = old_solve.lm_prediction.completion_tokens
        old_logprobs = old_solve.lm_prediction.completion_logprobs
        new_solve = eval.solution
        new_text = new_solve.apply().get_only_file().content_str
        # hacky way to get the function body
        new_text = '"""'.join(new_text.split('"""')[2:])
        new_text_tokens = [
            token.replace("▁", " ").replace("<0x0A>", "\n")
            for token in gen_tokenizer.tokenize(
                new_text,
                add_special_tokens=False,
            )
        ]
        matcher = difflib.SequenceMatcher(None, old_tokens, new_text_tokens)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "insert":
                continue
            equal_in_diff = tag == "equal"
            for token, logprob in zip(old_tokens[i1:i2], old_logprobs[i1:i2]):
                token_rows.append({
                    "token": token,
                    "logprob": logprob,
                    "prob": math.exp(logprob),
                    "is_in_new_text_str": token in new_text,
                    "equal_in_diff": equal_in_diff,
                    "is_closed_token": token in top_closed_toks,
                })

    df = pd.DataFrame(token_rows)

    # print the most common tokens
    #top_tokens = df["token"].value_counts().head(200)
    #print(top_tokens)
    ## Convert to a dict and pretty print
    #tok_to_count = top_tokens.to_dict()
    #from pprint import pprint
    #pprint(tok_to_count)


    # Make a boxplot of logprobs by is_in_new_text_str
    import seaborn as sns
    import matplotlib.pyplot as plt
    #sns.boxplot(
    #    data=df, x="equal_in_diff", y="prob")
    # Make a boxplot faceted by is_closed_token
    fig = sns.catplot(
        data=df, x="equal_in_diff", y="prob",
        col="is_closed_token", kind="box",
    )

    plt.show()

    for is_scaled in (True, False):
        fig, axs = plt.subplots(1, 1, figsize=(5, 5))
        use_df = df[~df['is_closed_token']]
        fig = postage_bar(
            exp_results=ExperimentResults(
                predicted_probabilities=np.array(use_df["prob"].values),
                true_labels=np.array(use_df["equal_in_diff"].values),
            ),
            ax=axs,
            show_scaled=is_scaled,
            show_unscaled=not is_scaled,
            show_axis=True,
        )
        plt.show()


if __name__ == "__main__":
    main()