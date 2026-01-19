from functools import partial
from difflib import SequenceMatcher
from typing import Literal, TypeVar, Callable
import numpy as np
from lmwrapper.openai_wrapper import OpenAiModelNames
from lmwrapper.utils import StrEnum

from localizing.evaling import token_equals_to_exp_token_agg, plot_exp
from localizing.localizing_structs import LocalizationList, BaseLocalization, AllCorrectEstimate, \
    TokenEqualsLocalization, MultiSamplingConfig, MultiSampleMode, MultiSamplingLocalization
import numpy as np
from typing import Callable
from numpy.typing import ArrayLike
from scipy.stats import gmean

from localizing.multi_data_gathering import multis_equals_from_scratch, FIX_REFERENCE_GT
from synthegrator.synthdatasets import DatasetName

class AggFuncsNames(StrEnum):
    min = "min"
    geomean = "geomean"
    mean = "mean"

    def __call__(self, x: ArrayLike) -> float:
        match self:
            case AggFuncsNames.min:
                return np.min(x)
            case AggFuncsNames.geomean:
                return float(gmean(x))
            case AggFuncsNames.mean:
                return np.mean(x)


def token_equals_all_correct_estimate(
    loc: TokenEqualsLocalization,
    agg_func: AggFuncsNames,
) -> AllCorrectEstimate:
    prob = agg_func(loc.estimated_keeps)
    return AllCorrectEstimate(
        prob_all_correct=prob,
        prob_all_correct_name=f"{agg_func.name}_estimated_keeps",
        actual=loc.is_base_success,
    )


def diff_sim_estimate_all_correct(
    loc: MultiSamplingLocalization,
    agg_func: AggFuncsNames = AggFuncsNames.mean,
) -> AllCorrectEstimate:
    vals = [
        SequenceMatcher(
            None,
            loc.base_solve.solve_steps[0].value,
            sample.solve_steps[0].value,
        ).ratio()
        for sample in loc.samples
    ]
    prob = agg_func(vals)
    return AllCorrectEstimate(
        prob_all_correct=prob,
        prob_all_correct_name=f"{agg_func.name}_diff_sim",
        actual=loc.is_base_success,
    )


T = TypeVar("T", bound=BaseLocalization)

def assign_all_correct(
    localizations: LocalizationList[T],
    est_func: Callable[[T], AllCorrectEstimate] ,
) -> LocalizationList[T]:
    new = localizations.copy()
    for loc in new.iter_passed_filtered():
        loc.all_correct_estimate = est_func(loc)
        if loc.all_correct_estimate.prob_all_correct is None:
            loc.annotate_failed_filter(
                f"failed_all_correct_estimate_{loc.all_correct_estimate.prob_all_correct_name}"
            )
    return new


if __name__ == "__main__":
    dataset = DatasetName.humaneval
    localizations = multis_equals_from_scratch(
        dataset=dataset,
        gen_model_name=OpenAiModelNames.gpt_3_5_turbo_instruct,
        filter_to_original_fails=False,
        # fix_reference=OpenAiModelNames.o3_mini,
        fix_reference=FIX_REFERENCE_GT,
        max_problems=200,
        multi_config=MultiSamplingConfig(
            multi_temperature=1.0,
            target_num_samples=10,
            # mode=MultiSampleMode.repair,
            mode=MultiSampleMode.from_prompt,
        ),
    )
    for agg in AggFuncsNames:
        a_locs = assign_all_correct(
            localizations,
            lambda loc: token_equals_all_correct_estimate(loc, agg_func=agg),
        )
        exp = token_equals_to_exp_token_agg(
            a_locs,
            pred_accessor=lambda loc: [loc.all_correct_estimate.prob_all_correct],
            gt_accessor=lambda loc: [loc.all_correct_estimate.actual],
        )
        plot_exp(exp, title=f"AllCorrect Token {dataset.value} {agg}")
    a_locs = assign_all_correct(
        localizations,
        lambda loc: diff_sim_estimate_all_correct(loc),
    )
    exp = token_equals_to_exp_token_agg(
        a_locs,
        pred_accessor=lambda loc: [loc.all_correct_estimate.prob_all_correct],
        gt_accessor=lambda loc: [loc.all_correct_estimate.actual],
    )
    plot_exp(exp, title=f"AllCorrect DiffSim {dataset.value}")
