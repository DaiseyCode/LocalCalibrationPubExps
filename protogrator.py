"""
Some extensions to synthegrator that I don't want to necessarily
worry about merging right now.
"""
from dataclasses import dataclass
import dataclasses
import enum
from timeit import default_timer as timer
from lmwrapper.internals import ModelInternalsRequest
from synthegrator.code_problems import CodeProblem, LmCodeSolution, CodeSolution
from synthegrator.code_solver import LmCodeSolverAutoRegressive, BaseCodeSolver
from synthegrator.lang_specs.lang_spec_python import PythonLangSpec
from lmwrapper.openai_wrapper import OpenAIPredictor
from typing import TypeVar, Any
import dataclasses
import warnings
from io import StringIO

from lmwrapper.abstract_predictor import LmPredictor, LmReasoningStyle
from lmwrapper.structs import LmPrediction, LmPrompt
from lxml import etree

from synthegrator.code_problems import CodeProblem
from synthegrator.environments import ProjectWorkingDirectory
from synthegrator.few_shotting import FewShotConfig
from synthegrator.problem_rendering import LmPromptRender, PromptRenderer, PromptRendererSingleEditGeneric
from synthegrator.prompting_test_case_selection import (
    PromptingTestCaseSelectionStrategy,
)
from synthegrator.response_parser import (
    ResponseParser,
    format_return_val_for_node,
)
from synthegrator.transformation_spec import (
    MarkElement,
    MarkText,
    SolveStep,
    StsEditable,
    StsInsert,
    StsPlaceTransforming,
    TransformationSpec,
    get_mark_element,
    get_verbs_per_path,
    map_paths_to_path_ids,
    markup_path,
)
from synthegrator.code_solver import snap_line_indent, LmPromptRenderSingleEdit, LmBasedSolver


T = TypeVar('T', bound=BaseCodeSolver)


@dataclass
class CodeSolutionSet(CodeSolution):
    def __init__(
        self,
        solutions: list[CodeSolution],
        preferred_solution: CodeSolution | None,
        time: float | None = None
    ):
        if not isinstance(solutions, list) or len(solutions) == 0:
            raise ValueError(
                "solutions must be a list with at least one element")
        if preferred_solution is None:
            preferred_solution = solutions[0]
        self.solutions = solutions
        self.preferred_solution = preferred_solution
        self.solve_steps = preferred_solution.solve_steps
        self.problem = preferred_solution.problem
        self.lang_spec = preferred_solution.lang_spec
        if time is not None:
            self.time = time
        else:
            max_time = None
            for sol in solutions:
                if sol.time is not None:
                    if max_time is None or sol.time > max_time:
                        max_time = sol.time
            self.time = max_time
        self.preferred_solution_index = None
        for i, sol in enumerate(solutions):
            if sol == preferred_solution:
                self.preferred_solution_index = i
                break

    def __len__(self):
        return len(self.solutions)

    def iterate_solutions(self):
        yield self.preferred_solution
        for i, sol in enumerate(self.solutions):
            if i != self.preferred_solution_index:
                yield sol

    def __iter__(self):
        return self.iterate_solutions()

    def dict_serialize(self) -> dict:
        out = super().dict_serialize()
        out["preferred_solution_index"] = self.preferred_solution_index
        out["solutions"] = [sol.dict_serialize() for sol in self.solutions]
        return out


@dataclass
class LmCodeSolutionSet(CodeSolutionSet, LmCodeSolution):
    def __init__(
        self,
        solutions: list[LmCodeSolution],
        preferred_solution: LmCodeSolution | None,
        time: float | None = None
    ):
        super().__init__(
            solutions, preferred_solution, time)
        self.uncertainty_estimates = self.preferred_solution.uncertainty_estimates
        self.prompt = self.preferred_solution.prompt
        self.lm_prediction = self.preferred_solution.lm_prediction


class LmCodeSolverTemp(LmCodeSolverAutoRegressive):
    def __init__(
        self,
        lm,
        internals: bool,
        include_logprobs: bool = None,
        max_gen_tokens: int = 400,
        prompt_renderer: PromptRenderer = PromptRendererSingleEditGeneric(),
    ):
        super().__init__(
            lm, 
            include_lm_response=True,
            prompt_renderer=prompt_renderer,
        )
        self._internals = internals
        self.include_logprobs = include_logprobs
        self._max_gen_tokens = max_gen_tokens


    def solve(
        self,
        problem: CodeProblem,
        randomizing_seed: int | None = None,
        temperature: float = 0,
        num_solutions: int = 1,
        cache: bool = True,
    ) -> LmCodeSolution | LmCodeSolutionSet:
        if temperature == 0 and num_solutions > 1:
            raise ValueError()
        start = timer()
        render = self.prompt_renderer.render(
            problem=problem,
            lm=self.model,
            prompt_seed=None,
        )
        logprobs = render.prompt.logprobs
        if (
            self.include_logprobs
            or (
                self.include_logprobs is None
                and (self.include_lm_response and (logprobs is None or logprobs == 0))
            )
        ):
            if isinstance(self.model, OpenAIPredictor):
                logprobs = 5
            else:
                logprobs = 1
        #if self.model.reasoning_style == LmReasoningStyle.ALWAYS_THINKING:
        #    temperature = 1.0
        render = dataclasses.replace(
            render,
            prompt=dataclasses.replace(
                render.prompt,
                temperature=temperature,
                max_tokens=self._max_gen_tokens,
                cache=cache,
                logprobs=logprobs,
                echo=False,  # Broken now with logprobs
                model_internals_request=ModelInternalsRequest(
                    return_hidden_states=False,
                    return_attentions=self._internals,
                ) if self._internals else None,
                num_completions=num_solutions,
            ),
        )

        resp = self.model.predict(
            render.prompt,
        )
        all_solve_steps = []
        for i in range(num_solutions):
            all_solve_steps.append(
                self.response_parser.parse(render, resp[i], problem)
            )
        end = timer()

        def find_lang_spec(solve_steps):
            for step in solve_steps:
                if step.path is not None:
                    return problem.get_lang_spec_for_path(step.path)
            return PythonLangSpec()

        solutions = []
        for solve_steps in all_solve_steps:
            solutions.append(LmCodeSolution(
               solve_steps,
               problem,
               lang_spec=find_lang_spec(solve_steps),
               time=end - start,
               prompt=render.prompt,
               lm_prediction=resp if self.include_lm_response else None,
            ))
        if num_solutions == 1:
            return solutions[0]
        else:
            return LmCodeSolutionSet(
                solutions,
                preferred_solution=solutions[0],
                time=end - start,
            )


def make_solver_partial(solver: T, **fixed_kwargs: Any) -> T:
    """
    Creates a new solver instance that wraps the provided solver instance,
    fixing the given keyword arguments on every call to solve.
    (This is analogous to functools.partial for functions.)

    The returned solver is a subtype of the provided solver's type and delegates
    attribute accesses to the underlying solver.

    :param solver: An instance of a solver (subclass of BaseCodeSolver).
    :param fixed_kwargs: Keyword arguments to always pass to solver.solve.
    :return: A new solver instance with a modified solve method.
    """
    class PartialSolver(type(solver)):
        def __init__(self, solver: T, fixed_kwargs: dict[str, Any]):
            # Do not call super().__init__ since we’re wrapping an existing solver.
            self._solver = solver
            self._fixed_kwargs = fixed_kwargs

        def solve(self, problem, randomizing_seed=None, **kwargs):
            # Merge fixed kwargs with call-specific kwargs.
            combined_kwargs = {**self._fixed_kwargs, **kwargs}
            return self._solver.solve(problem, randomizing_seed=randomizing_seed, **combined_kwargs)

        def __getattr__(self, attr: str) -> Any:
            # Delegate any missing attributes to the wrapped solver.
            return getattr(self._solver, attr)

        def __repr__(self) -> str:
            return f"<PartialSolver wrapping {self._solver!r} with fixed_kwargs={self._fixed_kwargs}>"

    return PartialSolver(solver, fixed_kwargs)


class LmHackSolver(LmBasedSolver):
    """
    A code solver that works by prompting a left-to-right language model
    and parsing the response.
    """

    def __init__(
        self,
        model: LmPredictor,
        prompt_renderer: PromptRenderer,
        response_parser: ResponseParser,
        prompting_test_case_strategy: PromptingTestCaseSelectionStrategy = None,
        include_lm_response: bool = False,
        snap_indent: int | None = None,
        cache_lm: bool = True,
    ) -> None:
        super().__init__(model, include_lm_response)
        self.prompt_renderer = prompt_renderer
        self.response_parser = response_parser
        self.snap_indent = snap_indent
        self.cache_lm = cache_lm
        if prompting_test_case_strategy is not None:
            msg = "Prompting test case strategy not implemented refactored"
            raise NotImplementedError(
                msg,
            )

    def solve(
        self,
        problem: CodeProblem,
        randomizing_seed: int | None = None,
    ) -> LmCodeSolution:
        start = timer()
        render = self.prompt_renderer.render(
            problem=problem,
            lm=self.model,
            prompt_seed=None,
        )
        if isinstance(render, LmPromptRenderSingleEdit):
            if render.changed_element is None:
                msg = "Insert element cannot be None"
                raise ValueError(msg)
            if not isinstance(render.changed_element, MarkElement):
                msg = (
                    "Render error. Expected changed element to be a MarkElement. Got: "
                    + str(type(render.changed_element))
                )
                raise ValueError(msg)
            if not isinstance(render.changed_element.verb, StsPlaceTransforming):
                msg = (
                    "Render error. Expected changed element verb to be "
                    "a verb StsPlaceTransforming. Got: "
                    + str(type(render.changed_element.verb))
                )
                raise ValueError(msg)
        logprobs = render.prompt.logprobs
        if self.include_lm_response and (logprobs is None or logprobs == 0):
            logprobs = 1
        render = dataclasses.replace(
            render,
            prompt=dataclasses.replace(
                render.prompt,
                temperature=1.0,
                max_tokens=20_000,
                cache=self.cache_lm,
                logprobs=logprobs,
                echo=False,  # Broken now with logprobs
            ),
        )

        resp = self.model.predict(render.prompt)
        solve_steps = self.response_parser.parse(render, resp, problem)
        if self.snap_indent:
            # TODO: move this into response parser?
            solve_steps = (
                dataclasses.replace(
                    solve_steps[0],
                    value=snap_line_indent(solve_steps[0].value, self.snap_indent),
                ),
            )
        end = timer()

        lang_spec = PythonLangSpec()
        for step in solve_steps:
            if step.path is not None:
                lang_spec = problem.get_lang_spec_for_path(step.path)
                break

        return LmCodeSolution(
            solve_steps,
            problem,
            lang_spec=lang_spec,
            time=end - start,
            prompt=render.prompt,
            lm_prediction=resp if self.include_lm_response else None,
        )


