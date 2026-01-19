from abc import ABC, abstractmethod
import dis
from synthegrator.problem_rendering_insertion_tags import TaggedEditRenderer, TaggedEditResponseParser, LmTaggedEditPrompt
from tqdm import tqdm
from lmwrapper.claude_wrapper.wrapper import ClaudeModelNames, get_claude_lm
import copy
from synthegrator.code_problems import CodeProblem, CodeSolution, LmCodeSolution, LmPrompt, SolveStep
from typing import Callable, Iterable, Optional
import os
import difflib
from synthegrator.code_solver import LmCodeSolverAutoRegressive
from lmwrapper.structs import LmPrediction

from lmwrapper.abstract_predictor import LmPredictor
from synthegrator.code_solver import LmBasedSolver
from lmwrapper.openai_wrapper import get_open_ai_lm, OpenAiModelNames
from lmwrapper.batch_config import CompletionWindow
from synthegrator.execution_threading import solve_and_evaluate_problems, evaluate_all_solutions
from synthegrator.solution_eval import SolutionEvaluation, _group_steps_by_path, apply_steps_to_markup
from synthegrator.transformation_spec import markup_path
from synthegrator.response_parser import _pull_first_md_block_from_completion
from debug_utils import inspect_object
from localizing.filter_helpers import debug_str_filterables
from fixsolver import RewriteFixSolver, solution_to_repair_problem
from localizing.localizing_structs import BaseLocalization, LocalizationList
from localizing.problem_processing import solve_to_text
from protogrator import LmHackSolver


def get_lm_constructor(model_reference: str):
    """
    Get a language model constructor function for the given model reference.
    
    Args:
        model_reference: Model name (OpenAI or Claude model)
        
    Returns:
        Constructor function that returns an LmPredictor instance
        
    Raises:
        ValueError: If model_reference is not a known OpenAI or Claude model
    """
    is_oai = model_reference in [str(info) for info in OpenAiModelNames]
    is_claude = model_reference in [str(info) for info in ClaudeModelNames]
    
    if is_oai:
        return lambda: get_open_ai_lm(model_reference)
    elif is_claude:
        return lambda: get_claude_lm(model_reference)
    else:
        raise ValueError(f"Unknown model_reference {model_reference}")


class LocalizationsFixAdder(ABC):
    """Abstract base class for adding fix data to localizations."""
    def add_fix_data(
        self,
        localizations: LocalizationList[BaseLocalization],
    ) -> LocalizationList[BaseLocalization]:
        output_localizations = localizations.copy()
        if not all(
            isinstance(loc, BaseLocalization)
            for loc in localizations.iter_passed_filtered()
        ):
            raise ValueError("All localizations must be BaseLocalizations")
        # Expect the fix properties to not be set
        if not all(
            loc.base_solve is not None
            or loc.gt_fix_solve is None
            or loc.gt_fix_eval is None
            for loc in localizations.iter_passed_filtered()
        ):
            raise ValueError("All localizations must have gt_fix properties unset")

        unfixed_locs = []
        for loc in output_localizations.iter_passed_filtered():
            if loc.is_base_success:
                # Copy over the fixes for already solved problems
                loc.gt_fix_solve = loc.base_solve
                loc.gt_fix_eval = loc.base_eval
            else:
                unfixed_locs.append(loc)
        print(f"ADD_FIX_DATA: Unfixed locs len {len(unfixed_locs)}")
        new_evals = self._attempt_fixes_for_unfixed(unfixed_locs)

        for new_eval, loc in zip(
            new_evals,
            unfixed_locs,
            strict=True,
        ):
            if not loc.annotate_filter(
                "fix_eval_is_not_none",
                new_eval is not None,
            ): continue
            if new_eval.solution is None:
                loc.annotate_failed_filter("fix_eval_solution_is_none")
                continue
            if not loc.annotate_filter(
                "fix_main_metric_is_not_none",
                new_eval.main_metric is not None,
            ): continue
            if not loc.annotate_filter(
                "fix_main_metric_is_success",
                new_eval.main_metric.is_success,
            ): continue
            loc.gt_fix_solve = new_eval.solution
            assert new_eval.solution is not None, "Solution is None"
            loc.gt_fix_eval = new_eval

        for loc in output_localizations.iter_passed_filtered():
            loc.annotate_tag("passed_fix_adder", self.name)

        return output_localizations
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return a human-readable name for this fix adder."""
        pass
    
    def _attempt_fixes_for_unfixed(self, unfixed_locs: list[BaseLocalization]) -> Iterable[SolutionEvaluation]:
        """
        Generate fixed solutions for the given unfixed localizations.
        
        Args:
            unfixed_locs: List of BaseLocalization objects that need fixing
            
        Returns:
            Iterable of SolutionEvaluation objects, one for each unfixed_loc.
            May yield None for localizations that couldn't be fixed.
        """
        raise NotImplementedError(
            "_attempt_fixes_for_unfixed is not implemented for this fix adder. "
            "Add or use a non-default add_fix_data method."
        )


class GroundTruthFixAdder(LocalizationsFixAdder):
    """Fix adder that uses ground truth solutions from the problem's known_solutions."""
    
    @property
    def name(self) -> str:
        return "ground_truth"
    
    def _attempt_fixes_for_unfixed(self, unfixed_locs: list[BaseLocalization]) -> Iterable[SolutionEvaluation]:
        has_a_gt = []
        
        for loc in unfixed_locs:
            has_a_gt.append(
                loc.base_solve.problem.known_solutions is not None and 
                len(loc.base_solve.problem.known_solutions) > 0
            )

        num_valid = sum(has_a_gt)
        
        # Create iterator over valid evaluations
        valid_evals_iter = iter(evaluate_all_solutions(
            (
                loc.base_solve.problem.known_solutions[0]
                for loc, has_valid in zip(unfixed_locs, has_a_gt)
                if has_valid
            ),
            max_threads=min(int(os.cpu_count() // 2), num_valid),
        ))
        
        # Second pass: yield evaluations or None
        for loc, has_valid in zip(unfixed_locs, has_a_gt):
            if not has_valid:
                yield None
            else:
                yield next(valid_evals_iter)



class SolverFixAdder(LocalizationsFixAdder):
    def __init__(
        self,
        solver_constructor: Callable[[], LmBasedSolver],
        max_threads_solve: Optional[int] = None,
        max_threads_eval: Optional[int] = None,
        solver_name: Optional[str] = None,
        narrow_edit: bool = False,
    ):
        self.solver_constructor = solver_constructor
        self.max_threads_solve = max_threads_solve
        self.max_threads_eval = max_threads_eval or min(os.cpu_count() // 2, 32)
        self._solver_name = solver_name
        self._solver = None
        self.narrow_edit = narrow_edit

    @property
    def name(self) -> str:
        if self._solver_name is None:
            solver = self._make_solver()
            self._solver_name = f"{solver.__class__.__name__}_{solver.model.model_name()}"
        return f"fix_add_with_{self._solver_name}"

    def _make_solver(self) -> LmBasedSolver:
        return self.solver_constructor()
    
    def _attempt_fixes_for_unfixed(
        self, 
        unfixed_locs: list[BaseLocalization]
    ) -> Iterable[SolutionEvaluation]:
        new_problems = [
            solution_to_repair_problem(
                loc.base_eval,
                narrow_edit=self.narrow_edit,
            )
            for loc in unfixed_locs
        ]
        
        print(f"NEW PROBLEMS ({self.name})")
        print(len(new_problems))
        #if new_problems:
        #    inspect_object(new_problems[0])
        
        solver = self._make_solver()
        if self.max_threads_solve is None:
            if solver.allows_multithreading:
                self.max_threads_solve = max(min(os.cpu_count() // 4, 16), 1)
            else:
                self.max_threads_solve = 1
        new_evals = solve_and_evaluate_problems(
            solver, 
            new_problems,
            max_threads_solve=self.max_threads_solve,
            max_threads_eval=min(self.max_threads_eval, len(new_problems)),
        )
        
        print(f"NEW EVALS ({self.name})")
        new_evals = list(new_evals)
        ev_with_no_sol = [ev for ev in new_evals if ev.solution is None]
        if ev_with_no_sol:
            print(f"No solutions for {len(ev_with_no_sol)} problems")
            inspect_object(ev_with_no_sol[0])
            print(ev_with_no_sol[0].exception_traceback)
            print(self.name)
            print(solver.solve(new_problems[0]))
            # If more than 5% runtime error
            if len(ev_with_no_sol) / len(new_evals) > 0.05:
                raise RuntimeError(f"No solutions for {len(ev_with_no_sol)} problems")
        print(len(new_evals))
        if new_evals and False:
            for i, ev in enumerate(new_evals):
                #inspect_object(new_evals[0].solution)
                print(f"-- {i} new_eval Main Metric")
                print(ev.main_metric)
                if ev.main_metric.is_success:
                    continue
                print(ev.test_results)
                for test_result in ev.test_results:
                    print(test_result.fail_message)
                print(new_evals[0].solution.prompt.text)
                print("--\nResponse:\n--\n")
                print(ev.solution.lm_prediction.completion_text)
                print("-- value")
                print(repr(ev.solution.solve_steps[0].value))
                exit()
        else:
            print(f"No new evals. New probs len {len(new_problems)}. Unfixed locs len {len(unfixed_locs)}")
            #exit()
        
        yield from new_evals


class RewriteFixAdder(SolverFixAdder):
    """Fix adder that uses a language model to rewrite and fix solutions."""
    
    def __init__(
        self, 
        fix_reference: str = OpenAiModelNames.o4_mini,
        max_threads_solve: int = None,
        max_threads_eval: Optional[int] = None,
    ):
        """
        Initialize the rewrite fix adder.
        
        Args:
            fix_reference: Model name to use for fixing (must be an OpenAI model)
            max_threads_solve: Maximum threads for solving
            max_threads_eval: Maximum threads for evaluation (defaults to half CPU count)
        """
        self.fix_reference = fix_reference
        lm_constructor = get_lm_constructor(fix_reference)
        
        def solver_constructor() -> LmBasedSolver:
            return RewriteFixSolver(
                lm_constructor(),
                include_lm_response=True,
            )
        
        super().__init__(
            solver_constructor=solver_constructor,
            max_threads_solve=max_threads_solve,
            max_threads_eval=max_threads_eval,
            solver_name=f"rewrite_with_{fix_reference}",
        )
    
    @property
    def name(self) -> str:
        return f"rewrite_with_{self.fix_reference}"


class TaggedEditFixAdder(SolverFixAdder):
    def __init__(
        self,
        fix_reference: str = OpenAiModelNames.o4_mini,
        max_threads_solve: int = None,
        max_threads_eval: Optional[int] = None,
    ):
        renderer = TaggedEditRendererWithMinimalRef(
            tag_name_edit="buggy",
            tag_name_solve="minimal_fix",
            include_first_tag_at_end=None,
            #custom_closing_lines=(
            #    "Task: There is a bug in the above code snippet somewhere in the code tagged as <buggy> and </buggy>. "
            #    "We are attempting to find a fix to the code. We want the fix to be as minimal as possible, that is touch "

            #    "the fewest characters as possible while still fixing any bugs. "
            #    "Please generate a fixed version inside of tags <minimal_fix> and </minimal_fix>.\n"
            #),
        )
        self.fix_reference = fix_reference
        
        parser = HackTaggedEditResponseParser()
        def solver_constructor() -> LmBasedSolver:
            return LmHackSolver(
                model=get_lm_constructor(fix_reference)(),
                prompt_renderer=renderer,
                response_parser=parser,
                include_lm_response=True,
            )

        super().__init__(
            solver_constructor=solver_constructor,
            max_threads_solve=max_threads_solve,
            max_threads_eval=max_threads_eval,
            solver_name=f"tagged_edit_fix_with_{str(fix_reference)}",
            narrow_edit=True,
        )

    @property
    def name(self) -> str:
        return f"tagged_edit_fix_with_{str(self.fix_reference)}"
    

class LocalizationScorer(ABC):
    """Abstract base class for scoring BaseLocalization objects."""
    
    @abstractmethod
    def score(self, loc: BaseLocalization) -> float:
        """
        Score a BaseLocalization where higher scores = better.
        Will always be a float, where failing values might be
        heavily penalized. Use as_distance to get a distance metric.
        
        Args:
            loc: BaseLocalization to score
            
        Returns:
            Score as float, where higher values indicate better localizations
        """
        pass
    
    @property
    def name(self) -> str:
        """Return a human-readable name for this scorer."""
        return self.__class__.__name__

    def __call__(self, loc: BaseLocalization) -> float:
        return self.score(loc)

    def as_distance(
        self, 
        loc: BaseLocalization, 
        none_if_fail: bool = True,
    ) -> float | None:
        """A version of the score as a distance metric. 
        Intended to be possibly more human interpretable. Might return None
        for failure cases where score is not coherent."""
        raise NotImplementedError("as_distance is not implemented for this scorer")


def calculate_failure_penalty(loc: BaseLocalization) -> float:
    if loc.check_passed_all_filters():
        return 0.0
    return -100_000 + len(loc.get_all_passed_filters()) * 1000


class CharEditDistanceBasedScorer(LocalizationScorer):
    """Scorer that primarily considers filter pass/fail status with penalties for failures.
    Uses character-level edit distance between base and ground truth texts as a scoring metric.
    Lower edit distance = better score."""
    
    def __init__(self):
        pass
    
    def score(self, loc: BaseLocalization) -> float:
        score = calculate_failure_penalty(loc)
        
        if score < 0:
            return score
            
        similarity = char_edit_distance(loc)
        distance_score = -100 * (1 - similarity)
        
        return score + distance_score

    def as_distance(
        self, 
        loc: BaseLocalization, 
        none_if_fail: bool = True,
    ) -> float | None:
        if none_if_fail and not loc.check_passed_all_filters():
            return None
        return char_edit_distance(loc, normalize_as_ratio=False)


def char_edit_distance_text(
    base_text: str,
    gt_text: str,
    normalize_as_ratio: bool = True,
) -> float:
    if base_text is None:
        raise ValueError("Base text is None")
    if gt_text is None:
        raise ValueError("GT text is None")
    matcher = difflib.SequenceMatcher(None, base_text, gt_text)
    
    if normalize_as_ratio:
        return matcher.ratio()
    else:
        # Calculate actual edit distance by counting operations from opcodes
        edit_distance = 0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                # Replace operations: count as max of the two ranges
                edit_distance += max(i2 - i1, j2 - j1)
            elif tag == 'delete':
                # Delete operations: count deleted characters
                edit_distance += i2 - i1
            elif tag == 'insert':
                # Insert operations: count inserted characters
                edit_distance += j2 - j1
            # 'equal' operations don't contribute to edit distance
        return float(edit_distance)

def char_edit_distance(
    loc: BaseLocalization,
    normalize_as_ratio: bool = True,
) -> float:
    base_text = loc.get_base_text()
    gt_text = loc.get_gt_fix_text()
    return char_edit_distance_text(base_text, gt_text, normalize_as_ratio)


class PickBestFixAdder(LocalizationsFixAdder):
    """Fix adder that composes multiple fix adders and picks the best result based on scoring."""

    def __init__(
        self,
        fix_adders: list[LocalizationsFixAdder],
        scorer: LocalizationScorer,
    ):
        """
        Initialize the pick best fix adder.
        
        Args:
            fix_adders: List of fix adders to try for each localization
            scorer: Scorer to determine which fix is best (higher score = better)
        """
        if not fix_adders:
            raise ValueError("Must provide at least one fix adder")
        self.fix_adders = fix_adders
        self.scorer = scorer
    
    @property
    def name(self) -> str:
        adder_names = [adder.name for adder in self.fix_adders]
        return f"pick_best({'+'.join(adder_names)})_by_{self.scorer.name}"
    
    def add_fix_data(
        self,
        localizations: LocalizationList[BaseLocalization],
    ) -> LocalizationList[BaseLocalization]:
        candidate_results = [
            fix_adder.add_fix_data(localizations).iter_all()
            for fix_adder in self.fix_adders
        ]

        # Score all candidates and pick the best
        best_candidates = []
        for og_loc, candidates in zip(
            localizations.iter_all(),
            zip(*candidate_results, strict=True),
            strict=True,
        ):
            if og_loc.base_eval.main_metric is None or og_loc.base_eval.main_metric.is_success:
                best_candidates.append(og_loc)
                continue
            best_candidate: BaseLocalization = None
            best_score = -float("inf")
            best_adder_name = ""
            passed_adders = []
            failed_adders = []
            scores = []
            distances = []
            for candidate, fix_adder in zip(candidates, self.fix_adders, strict=True):
                if candidate.check_passed_all_filters():
                    passed_adders.append(fix_adder.name)
                else:
                    failed_adders.append(fix_adder.name)
                if candidate.get_base_text() is None:
                    print(f"Base text is None for {candidate}")
                    inspect_object(candidate)
                    print("Solve to text:")
                    print(solve_to_text(candidate.base_solve, candidate.dataset_name))
                    exit()
                score = self.scorer.score(candidate)
                print(f"SCORE ({fix_adder.name}): {score}")
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
                    best_adder_name = fix_adder.name
                scores.append(score)
                distance = self.scorer.as_distance(candidate, none_if_fail=True)
                distances.append(distance)
            best_candidate.annotate_tag(f"chosen_best_adder", best_adder_name)
            for passed_adder in passed_adders:
                best_candidate.annotate_tag(f"pick_best_branched_passed", passed_adder)
            for failed_adder in failed_adders:
                best_candidate.annotate_tag(f"pick_best_branched_failed", failed_adder)
            for score, fix_adder in zip(scores, self.fix_adders, strict=True):
                best_candidate.annotate_tag(f"pick_best_branched_score", (fix_adder.name, score))
            for distance, fix_adder in zip(distances, self.fix_adders, strict=True):
                best_candidate.annotate_tag(
                    f"pick_best_branched_distance", (fix_adder.name, distance))
            best_candidates.append(best_candidate)
        
        assert len(best_candidates) == len(localizations)
        return LocalizationList(best_candidates)


class MakeMoreMinimalFixAdder(LocalizationsFixAdder):
    def __init__(
        self, 
        first_fix_adder: LocalizationsFixAdder,
        lm: LmPredictor,
    ):
        self.first_fix_adder = first_fix_adder
        self.lm = lm

    @property
    def name(self) -> str:
        return f"make_more_minimal_with_{self.lm.name}"

    def add_fix_data(
        self,
        localizations: LocalizationList[BaseLocalization],
    ) -> LocalizationList[BaseLocalization]:
        localizations = self.first_fix_adder.add_fix_data(localizations)
        print(debug_str_filterables(localizations.iter_all()))
        #exit()
        new_locs = []
        locs_to_process_indices = []
        locs_to_process = []
        for loc in localizations.iter_all():
            loc_copy = loc.copy()
            new_locs.append(loc_copy)
            if loc.base_eval.main_metric.is_success:
                # Was already origionally correct. No need to repair
                continue
            if loc.gt_fix_eval is None:
                # Didn't repair it successfully. Won't minimize
                continue
            if not loc.gt_fix_eval.main_metric.is_success:
                # Still not repaired, so can't make it more minimal
                continue
            locs_to_process.append(loc_copy)
        
        self._make_minimal(locs_to_process)
        return LocalizationList(new_locs)


    def _make_minimal(self, locs: list[BaseLocalization]) -> None:
        prompts = [
            _make_minimal_solution_prompt(loc, self.lm) 
            for loc in locs
        ]
        prompts = [p for p in prompts if p is not None]
        completion_window = CompletionWindow.ASAP
        completions = self.lm.predict_many(
            prompts,
            completion_window=completion_window,
        )
        solves_to_eval = []
        locs_for_evals = []
        if completion_window == CompletionWindow.ASAP:
            completions = tqdm(completions, total=len(prompts), desc="Running minimal fixes prompts")
        for resp in completions:
            new_solution, distance_change = _parse_response(resp)
            if distance_change < 0:
                continue
            solves_to_eval.append(new_solution)
            locs_for_evals.append(resp.prompt.metadata)
        for loc, ev in zip(
            locs_for_evals, 
            evaluate_all_solutions(
                solves_to_eval, 
                max_threads=min(os.cpu_count() // 2, 16),
            ),
            strict=True,
        ):
            if ev.main_metric.is_success:
                loc.gt_fix_eval = ev
                loc.gt_fix_solve = new_solution
                loc.annotate_passed_filter("minimize_distance", distance_change)
        

def _parse_response(resp: LmPrediction[BaseLocalization]) -> tuple[CodeSolution, float]:
    loc = resp.prompt.metadata
    distance = char_edit_distance(loc, normalize_as_ratio=False)
    base_solve_step = loc.base_solve.solve_steps[0]
    fix_solve_step = loc.gt_fix_solve.solve_steps[0]
    completion_text = _pull_first_md_block_from_completion(resp)
    if completion_text is None:
        completion_text = resp.completion_text.lstrip("\n")
    if completion_text.startswith("def ") and not base_solve_step.value.startswith("def "):
        # Strip off the first line if it's a function definition
        completion_text = "\n".join(completion_text.split("\n")[1:])
    print("Base solve Value:")
    print(base_solve_step.value)
    print("Fix solve Value:")
    print(fix_solve_step.value)
    print("Completion text:")
    print(resp.completion_text)
    print("-pull")
    print(completion_text)
    print("---")
    new_dist = char_edit_distance_text(
        loc.get_base_text(), 
        completion_text, 
        normalize_as_ratio=False
    )
    #print(completion_text)
    #print("Tags:")
    #print(hack_loc.get_tags())
    #if new_dist < distance:
    #    print(f"✅ New distance: {new_dist} (was {distance})")
    #else:
    #    print(f"❌ New distance: {new_dist} (was {distance})")
    #    print("🚫 No improvement")
    new_solution = copy.copy(loc.base_solve)
    completion_starting_left_content = completion_text[: len(completion_text) - len(completion_text.lstrip())]
    base_starting_left_content = base_solve_step.value[: len(base_solve_step.value) - len(base_solve_step.value.lstrip())]
    if completion_starting_left_content != base_starting_left_content:
        completion_text = base_starting_left_content + completion_text[len(base_starting_left_content) :]
    new_solution.solve_steps = [SolveStep(
        mark_id=base_solve_step.mark_id,
        value=completion_text,
        path=base_solve_step.path,
    )]
    return new_solution, new_dist - distance


def _make_minimal_solution_prompt(
    loc: BaseLocalization,
    lm: LmPredictor,
) -> LmPrompt[BaseLocalization]:
    if not loc.gt_fix_eval.main_metric.is_success:
        raise ValueError("GT fix must be successful")
    distance = char_edit_distance(loc, normalize_as_ratio=False)
    if distance <= 0.0:
        return None
    fix_solve_steps = loc.gt_fix_solve.solve_steps
    if len(fix_solve_steps) > 1:
        raise NotImplementedError("Only one solve step is supported")
    base_solve_steps = loc.base_solve.solve_steps
    if len(base_solve_steps) > 1:
        raise NotImplementedError("Only one solve step is supported")
    if len(loc.get_base_text().strip()) == 0:
        return None
    base_solve_step = base_solve_steps[0]
    fix_solve_step = fix_solve_steps[0]
    prompt_str = []
    prompt_str.append(f"We have the following context for a code problem:")
    prompt_str.append("```")
    prompt_str.append(loc.base_solve.problem.working_directory.files.get_only_file().content_str)
    prompt_str.append("```")
    prompt_str.append(f"We have the following base solution which is buggy:")
    prompt_str.append("```")
    prompt_str.append(loc.get_base_text())
    prompt_str.append("```")
    prompt_str.append(f"We then have the following proposed corrected solution which is correct:")
    prompt_str.append("```")
    prompt_str.append(loc.get_gt_fix_text())
    prompt_str.append("```")
    prompt_str.append(f"We are working to the mimimal fix of the base solution. "
                      f"Currently, the fix is {distance} characters different from the base solution.")
    prompt_str.append("If possible, please propose a new solution which is as close as possible to base solution (a minimal fix) while still being correct.")
    prompt_str.append("Give your answer with just the new code (not diff format). Place your solution in a markdown block. Start where the previous solutions begin (do not add a signature if the origional solution just had the body. Only give the body).")
    prompt_str = "\n".join(prompt_str)
    print(prompt_str)
    print("\n---\n")
    prompt = LmPrompt(
        prompt_str,
        logprobs=0,
        max_tokens=10_000,
        cache=True,
        metadata=loc,
    )
    return prompt



from synthegrator.problem_rendering_insertion_tags import TaggedEditRenderer, TaggedEditResponseParser, LmTaggedEditPrompt, PromptRendererMultiEdit, _marked_up_text_to_tagged_prompt_text, _make_closing_lines
import dataclasses
import warnings
from io import StringIO

from lmwrapper.abstract_predictor import LmPredictor
from lmwrapper.structs import LmPrediction, LmPrompt
from lxml import etree

from synthegrator.code_problems import CodeProblem
from synthegrator.environments import ProjectWorkingDirectory
from synthegrator.few_shotting import FewShotConfig
from synthegrator.problem_rendering import LmPromptRender, PromptRenderer
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


class TaggedEditRendererWithMinimalRef(PromptRendererMultiEdit):
    """Renders a problem as a sequence of edits in xml-like tags"""

    def __init__(
        self,
        few_shot_config: FewShotConfig = None,
        prompt_test_case_selection_strategy: PromptingTestCaseSelectionStrategy = None,
        tag_name_edit: str = "buggy",
        tag_name_solve: str = "minimal_fix",
        include_first_tag_at_end: bool | None = None,
        custom_closing_lines: str | None = None,
        add_stop: bool | None = None,
    ):
        super().__init__()
        if custom_closing_lines is not None:
            raise ValueError("custom_closing_lines not supported")
        self._few_shot_config = few_shot_config
        self._prompt_test_case_selection_strategy = prompt_test_case_selection_strategy
        self._tag_name_edit = tag_name_edit
        self._tag_name_solve = tag_name_solve
        self._include_first_tag_at_end = include_first_tag_at_end
        self._custom_closing_lines = custom_closing_lines
        self._add_stop = add_stop

    def render(
        self,
        problem: CodeProblem,
        lm: LmPredictor,
        prompt_seed: int | None = None,
    ) -> LmTaggedEditPrompt:
        if self._few_shot_config is not None:
            warnings.warn(
                "Few shot not supported with this non-chat version. "
                "No few shot will be used. TODO",
            )
        spec = problem.transformation_spec
        if spec.count_editing_statements() != 1:
            msg = "Currently simple and assumes a exactly one edit statement"
            raise NotImplementedError(
                msg,
            )
        path_to_verbs = get_verbs_per_path(problem.working_directory, spec)
        prompt_lines = []
        if problem.instructions:
            prompt_lines.append(problem.instructions)
        path_to_path_id = map_paths_to_path_ids(problem.working_directory, spec)
        all_tag_ids = []
        prompt_lines.append("We are going to try to make a minimal fix to some buggy code. Please pay particular attention to the code in the <buggy></buggy> tag. Long code files might be truncated.")
        for path in path_to_verbs:
            markedup_text = markup_path(
                problem.working_directory,
                path,
                problem.transformation_spec,
            )

            prompt_lines.append("@@ " + str(path) + " @@")
            tagged_prompt, tag_ids = _marked_up_text_to_tagged_prompt_text(
                markedup_text,
                path_id=path_to_path_id[path],
                tag_name=self._tag_name_edit,
            )
            if not "<buggy id=" in tagged_prompt:
                print("NO buggy at all??")
                raise ValueError("No buggy tag found in tagged prompt")
                exit()
            
            # Find buggy tag positions
            buggy_start = tagged_prompt.index("<buggy id=")
            buggy_end = tagged_prompt.index("</buggy>") + len("</buggy>")
            
            limit = 3000
            forward_limit = limit * 4
            
            # Truncate before buggy tag if needed
            if buggy_start > forward_limit:
                tagged_prompt = "..." + tagged_prompt[buggy_start - forward_limit:]
                # Update positions after truncation
                buggy_start = tagged_prompt.index("<buggy id=")
                buggy_end = tagged_prompt.index("</buggy>") + len("</buggy>")
            
            # Truncate after buggy tag if needed
            distance_after_buggy = len(tagged_prompt) - buggy_end
            if distance_after_buggy > limit:
                tagged_prompt = tagged_prompt[:buggy_end + limit] + "...\nTRUNCATED"
            if "<buggy id=" not in tagged_prompt:
                print("NO buggy start??")
                raise ValueError("No buggy start found in tagged prompt")
            if "</buggy>" not in tagged_prompt:
                print("NO buggy end??")
                raise ValueError("No buggy end found in tagged prompt")
            prompt_lines.append("```\n" + tagged_prompt + "\n```")
            all_tag_ids.extend(tag_ids)
            prompt_lines.append("")
        #closing, prepend = _make_closing_lines(
        #    all_tag_ids=all_tag_ids,
        #    tag_name_edit=self._tag_name_edit,
        #    tag_name_solve=self._tag_name_solve,
        #    include_first_tag_at_end=(
        #        not lm.is_chat_model
        #        if self._include_first_tag_at_end is None
        #        else self._include_first_tag_at_end
        #    ),
        #    custom_closing_lines=self._custom_closing_lines,
        #)
        text_so_far = "\n".join(prompt_lines)
        buggy_tag_loc = text_so_far.index("</buggy>")
        distance_back_buggy = len(text_so_far) - buggy_tag_loc
        prompt_lines.append("---")
        closing = (
            "There is a bug in the above code snippet somewhere in the code tagged as <buggy> and </buggy>.\n"
        )
        failures = problem.instructable_metadata.get('test_failures', [])
        if failures:
            closing += "\n--- Sample of Test Failures ---\n"
            for i, tc in enumerate(failures):
                closing += "-- Test {i}"
                message = tc['fail_message']  # TODO: check if should be fail_text ??
                if len(message) > 400:
                    message = "..." + message[-400:]
                closing += f"```\n{message}\n```"
                if i > 3:
                    break
            closing += f"\n--- End of Sample of Test Failures ({len(failures) - i - 1} more not shown) ---\n"
        else:
            closing += "\nNo test failure message failed, which means there was some failing issue that caused the tests to not run (eg, import error, syntax error, etc).\n"
        closing += (
            "\nTASK: We are attempting to find a fix to the buggy code. We want the fix to be as minimal as possible, that is touch "
            "the fewest characters as possible while still fixing any bugs.\n"
        )

        if problem.known_solutions:
            closing += "For reference, here is the reference solution:\n\n"
            sol = problem.known_solutions[0]
            if len(sol.solve_steps) > 1:
                raise ValueError("Expected only one solve step for known solution")
            base_value = sol.solve_steps[0].value
            closing += "<reference>" + base_value + "</reference>"
            closing += "\n\n"
            past_solve = problem.past_solve_context[0].solve_steps[0].value
            dist = char_edit_distance_text(past_solve, base_value, normalize_as_ratio=False)
            closing += f"The reference solution passes test cases, but it is {dist} characters away from the buggy solution."
            closing += "\n"
            closing += f"We are looking for a correct fix which is more minimal than {dist} characters. Considerations might be to use the same variable names, style, and approach as the buggy code, while fixing any bugs. It might be helpful to start by understanding how the buggy code differs from reference code, and then trying to figure out how integrate in the fixes in a minimal way, or if the bug in the buggy code is clear, you can fix it directly."
            if distance_back_buggy > 500:
                closing += "As a reminder, the buggy code is:\n"
                closing += "<buggy>" + past_solve + "</buggy>"
        closing += (
            "\nDo as much thinking as you want, but then please place your fixed version of the buggy inside of tags <minimal_fix> and </minimal_fix>. Include only exactly one fixed version in order for us to be able to parse your solution. Your solution will completely replace the code in the buggy tag, so please make sure you do not miss any code (for the given function body or lines) and give a complete solution.\n"
        )
        prepend = ""
        prompt_lines.append(closing)
        text = "\n".join(prompt_lines)
        stop = None
        add_stop = self._add_stop
        if add_stop is None:
            add_stop = not lm.is_chat_model
        if len(all_tag_ids) == 1 and add_stop:
            stop = [f"</{self._tag_name_solve}>"]
        return LmTaggedEditPrompt(
            prompt=LmPrompt(
                text=text,
                stop=stop,
            ),
            tag_name_edit=self._tag_name_edit,
            tag_name_solve=self._tag_name_solve,
            preprompted_tag_start=prepend,
        )

    def __call__(
        self,
        problem: CodeProblem,
        lm: LmPredictor,
        prompt_seed: int | None = None,
    ) -> "LmTaggedEditPrompt":
        return self.render(problem, lm, prompt_seed)


from synthegrator.problem_rendering_insertion_tags import _find_ids_and_values_in_tagged_answer, _ids_to_path_id_and_mark_id, _get_mark_default_id

class HackTaggedEditResponseParser(ResponseParser):
    def __init__(self):
        super().__init__()

    def parse(
        self,
        render: LmTaggedEditPrompt,
        resp: LmPrediction,
        problem: CodeProblem,
    ) -> list[SolveStep]:
        ids_to_values = _find_ids_and_values_in_tagged_answer(
            render.preprompted_tag_start + resp.completion_text,
            tag_name_edit=render.tag_name_edit,
            tag_name_solve=render.tag_name_solve,
            default_id=_get_mark_default_id(problem),
        )
        spec = problem.transformation_spec
        wd = problem.working_directory
        # print(resp.completion_text)
        ids_to_path_and_mark_id = _ids_to_path_id_and_mark_id(ids_to_values, spec, wd)
        solve_steps = []
        for tag_id, (path, mark_id) in ids_to_path_and_mark_id.items():
            node = get_mark_element(spec, path, wd, mark_id)
            if node is None:
                continue  # We are permissive for responses with nonexisted nodes
            lang_spec = problem.get_lang_spec_for_path(path)
            value = ids_to_values[tag_id]
            if value.startswith("```") and value.endswith("```"):
                first_new_line = value.find("\n")
                if first_new_line != -1:
                    value = value[first_new_line + 1 :-3]

            solution_text = format_return_val_for_node(
                value,
                node,
                lang_spec,
            )
            solve_steps.append(
                SolveStep(
                    path,
                    mark_id,
                    solution_text,
                ),
            )
        return solve_steps