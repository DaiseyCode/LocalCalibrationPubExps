from itertools import islice
import gc
import random
from synthegrator.problem_rendering import LmPromptRender, PromptRenderer, PromptRendererSingleEditGeneric
from pathlib import Path
from pprint import pprint
import os

from lmwrapper.claude_wrapper.wrapper import ClaudeModelInfo, get_claude_lm
from lmwrapper.openai_wrapper import OpenAiModelNames, OpenAiModelInfo, get_open_ai_lm
from synthegrator.execution_threading import evaluate_all_solutions
from synthegrator.solution_eval import SolutionEvaluation
from synthegrator.synthdatasets.dypybench import yield_dypybench
from synthegrator.synthdatasets.mbpp import yield_mbpp, yield_mbpp_plus
from tqdm import tqdm
import joblib.memory
import torch
from lmwrapper.huggingface_wrapper import get_huggingface_lm
from synthegrator.code_problems import CodeProblem, LmCodeSolution
from synthegrator.synthdatasets.human_eval import yield_human_eval
from synthegrator.util import pretty_print_python_code
from synthegrator.synthdatasets.livecodebench import yield_livecode_problems
from fixsolver import RewriteFixSolver
from prompt_shortening import ShorteningPromptRenderer
from protogrator import LmCodeSolverTemp, LmCodeSolutionSet
import diskcache
from synthegrator.synthdatasets import DatasetName
from synthegrator.synthdatasets import yield_problems_from_name

cache = diskcache.Cache("cache")


def get_solutions_problem(
    solver: LmCodeSolverTemp | RewriteFixSolver,
    problem: CodeProblem,
    model_name_key: str,
    num_solves: int = 10,
    temperature: float = 0.7,
    use_cache: bool = None,
) -> LmCodeSolution | LmCodeSolutionSet:
    is_api_model = (
        OpenAiModelNames.name_to_info(model_name_key) is not None
    )
    cache_key = (
        str(type(solver)) + solver.model.model_name() + model_name_key + problem.dataset_name + problem.problem_id + "t10" + str(float(temperature)) + ";" + str(int(num_solves))
        + str(solver.prompt_renderer)
    )
    use_cache = use_cache and not is_api_model  # rely on model cache
    if use_cache:
        if cache_key in cache:
            print("Cache hit", cache_key)
            return cache[cache_key]
    if OpenAiModelNames.name_to_info(model_name_key) is not None:
        return solver.solve(
            problem,
            num_solutions=num_solves,
            temperature=temperature,
            cache=True,
        )
    else:
        solutions = []
        tries = 0
        num_torch_runtime_errors = 0
        while tries < max(num_solves * 1.5, 5) and len(solutions) < num_solves:
            tries += 1
            if tries > 0 and temperature == 0:
                temperature += 0.1
            # Manually do this so I can cut degenerates
            try:
                solve = solver.solve(
                    problem,
                    temperature=temperature,
                    cache=False,
                )
                solutions.append(solve)
            except ValueError as e:
                if "degenerate" in str(e):
                    print("Degenerate")
                    continue
                if "Cannot fix solutions when there are more new tokens than output tokens" in str(e):
                    print("Token mismatch")
                    continue
                raise e
            except RuntimeError as e:
                if "probability tensor contains either `inf`, `nan` or element < 0" in str(e):
                    print("Probability tensor contains `inf`, `nan` or element < 0")
                    num_torch_runtime_errors += 1
                    continue
                raise e
        if num_torch_runtime_errors > tries - 3 and num_torch_runtime_errors > 0:
            raise ValueError(f"Too many torch runtime errors {num_torch_runtime_errors} in {tries} tries")
    #solutions = solver.solve(
    #    problem,
    #    num_solutions=num_solves,
    #    temperature=temperature,
    #    cache=True,
    #)
    if num_solves == 1:
        out = solutions[0]
    else:
        out = LmCodeSolutionSet(
            solutions,
            preferred_solution=solutions[0],
        )
    cache[cache_key] = out
    return out

cur_path = Path(__file__).parent
memo = joblib.memory.Memory(cur_path / "solves_full_cache")


_lm_save, _lm_save_name = None, None
def get_model_in_mem(model_name):
    global _lm_save, _lm_save_name
    if model_name != _lm_save_name:
        _lm_save = None
        if "OpenAi" in str(type(model_name)):
            _lm_save = get_open_ai_lm(model_name)
        elif "Claude" in str(type(model_name)):
            _lm_save = get_claude_lm(model_name)
        elif isinstance(model_name, str):
            _lm_save = get_huggingface_lm(model_name, precision=torch.float16, device="cuda:0")
            assert next(_lm_save._model.parameters()).device.type == "cuda", f"Model device is {next(_lm_save._model.parameters()).device}"
        else:
            raise ValueError(f"Unknown model name type {model_name}")
        _lm_save_name = model_name
    return _lm_save


def clear_model_in_mem():
    global _lm_save, _lm_save_name
    _lm_save = None
    _lm_save_name = None
    gc.collect()
    torch.cuda.empty_cache()


@memo.cache
def faster_get_problems(
    dataset: DatasetName,
    max_problems: int | None = None,
) -> list[CodeProblem]:
    val = list(tqdm(yield_problems_from_name(dataset, max_problems=max_problems), desc=f"Getting problems from {dataset}"))
    return val


def clean_weird_problem_id(problem_id: str) -> str:
    # Repocod changed the way problem ids are formatted in their version update.
    #   Now this doesn't align and has to be fixed.
    problem_id = problem_id.replace("_File-level", "")
    problem_id = problem_id.replace("_file-level", "")
    problem_id = problem_id.replace("_Repository-level", "")
    problem_id = problem_id.replace("_repository-level", "")
    problem_id = problem_id.replace("_Self-contained", "")
    problem_id = problem_id.replace("_self-contained", "")
    return problem_id


@memo.cache
def get_some_solves(
    #model_name: str = "Salesforce/codegen-350M-mono",
    model_name: str = "mistralai/Mistral-7B-v0.1",
    internals: bool = False,
    max_problems: int | None = 10,
    solves_per_problem: int = 10,
    temperature: float = 0.0,
    max_gen_tokens: int = 400,
    dataset: DatasetName | list[DatasetName] = DatasetName.humaneval,
    problem_id_subset: list[str] | None = None,
    run_eval: bool = True,
    include_logprobs: bool = True,
) -> list[
    SolutionEvaluation
    | list[SolutionEvaluation]
    | LmCodeSolutionSet
    | LmCodeSolution
]:
    print("Getting model....")
    lm = get_model_in_mem(model_name)
    print(f"Getting problems {dataset}...")
    all_problems = []
    problem_id_to_problem = {}

    if not isinstance(dataset, list):
        dataset = [dataset]
    for ds in dataset:
        print(f"Getting problems from {ds}")
        problems = faster_get_problems(ds, max_problems=max_problems)
        #if dataset != DatasetName.repocod:
        #    problems = list(yield_problems_from_name(ds, max_problems=max_problems))
        #else:
        #    # Repocod problems are ordered by dataset I think, so
        #    # sample randomly from the full set
        #    problems = list(yield_problems_from_name(ds))
        #    if len(problems) > max_problems:
        #        rng = random.Random(42)
        #        problems = list(rng.sample(problems, max_problems))
        if problem_id_subset is not None:
            problem_id_to_problem.update({problem.problem_id: problem for problem in problems})
            problem_id_to_problem.update({clean_weird_problem_id(problem.problem_id): problem for problem in problems})
        print(f"Got {len(problems)} problems from {ds}")
        all_problems.extend(problems)
    print(f"Got {len(all_problems)} problems")
    print(f"Num keys in problem_id_to_problem: {len(problem_id_to_problem)}")
    if problem_id_subset is not None:
        print(f"Select {len(problem_id_subset)} problems")
        all_problems = []
        for problem_id in problem_id_subset:
            if problem_id in problem_id_to_problem:
                all_problems.append(problem_id_to_problem[problem_id])
            elif clean_weird_problem_id(problem_id) in problem_id_to_problem:
                all_problems.append(problem_id_to_problem[clean_weird_problem_id(problem_id)])
            else:
                pprint(list(sorted(problem_id_to_problem.keys())))
                print(f"Problem {problem_id} not found")
                all_problems.append(None)
                exit()
        #all_problems = [
        #    problem_id_to_problem.get(problem_id, None)
        #    for problem_id in problem_id_subset
        #]
        none_count = sum(1 for problem in all_problems if problem is None)
        print(f"None count: {none_count}")
        all_problems = [problem for problem in all_problems if problem is not None]
    problems = all_problems
    print(f"Get solves {max_problems=} {temperature=}...")
    solver = LmCodeSolverTemp(
        lm, internals,
        max_gen_tokens=max_gen_tokens,
        include_logprobs=include_logprobs,
        prompt_renderer=ShorteningPromptRenderer(
            wrapper_renderer=PromptRendererSingleEditGeneric(),
            only_datasets=[DatasetName.repocod],
        ),
    )
    solver.cache_lm = True
    if max_problems is None:
        max_problems = len(problems)
    solutions = []
    print("start loop")
    for problem in tqdm(problems[:min(max_problems, len(problems))], desc=f"Getting solves"):
        try:
            sol = get_solutions_problem(
                solver, problem,
                model_name_key=model_name,
                num_solves=solves_per_problem,
                temperature=temperature,
                #use_cache=True,
            )
        except ValueError as e:
            if "degenerate" in str(e):
                print("Degenerate")
                continue
            if "Cannot fix solutions when there are more new tokens than output tokens" in str(e):
                print("Token mismatch")
                continue
            raise e
        except torch.cuda.OutOfMemoryError:
            print("!!!!")
            print("Torch OOM in get_some_solves")
            print("!!!!")
            continue
            raise e
        solutions.append(sol)
    if not run_eval:
        return solutions
    if solves_per_problem > 1:
        raise ValueError
    num_cpus = os.cpu_count()
    if dataset == DatasetName.repocod:
        target = min(int(num_cpus * 0.5), num_cpus - 1)
    else:
        target = min(int(num_cpus * 0.75), num_cpus - 1)
    target = max(1, target)
    print(f"Evaluating {len(solutions)} of {dataset} with {target} cpus")
    return list(evaluate_all_solutions(solutions, max_threads=target))


def main():
    evals = get_some_solves()[4]
    for eval in evals:
        solution = eval.solution
        assert isinstance(solution, LmCodeSolution)
        pretty_print_python_code(solution.lm_prediction.prompt.text)
        pretty_print_python_code(solution.solve_steps[0].value)
        if eval.main_metric.is_success:
            print("✅ Pass")
        else:
            for test_result in eval.test_results:
                pprint(test_result)
        print("-"*10)


if __name__ == "__main__":
    main()