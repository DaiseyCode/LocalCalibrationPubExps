import dataclasses

from lmwrapper.openai_wrapper import OpenAiModelNames
from localizing.filter_helpers import debug_str_filterables
from calipy.calibrate import PlattCalibrator
from localizing.localizing_structs import BaseLocalization, MultiSamplingConfig, LocalizationList, \
    MultiSamplingLocalization, MultiSampleMode
from dataclasses import dataclass
import difflib
import numpy as np

from localizing.multi_data_gathering import get_base_solves, add_fix_data, create_multis_version
from localizing.problem_processing import solve_to_text
from multi_compare import is_func_body_junk_solve
from synthegrator.synthdatasets import DatasetName

@dataclass()
class LineEditInfo:
    prob_deletes: np.ndarray
    prob_insert_before: np.ndarray
    prob_rewrites: np.ndarray
    line_hunks: list[str]

    def __post_init__(self):
        if not (
            len(self.prob_deletes)
            == len(self.prob_insert_before) - 1
            == len(self.prob_rewrites)
            == len(self.line_hunks)
        ):
            raise ValueError("not matching lenghts")

    def is_shape_compatible_with(self, other: 'LineEditInfo') -> bool:
        assert self.prob_deletes.shape == other.prob_deletes.shape
        assert self.prob_insert_before.shape == other.prob_insert_before.shape
        assert self.prob_rewrites.shape == other.prob_rewrites.shape


@dataclass(kw_only=True)
class LineLocalization(BaseLocalization):
    _precomputed_info: LineEditInfo = None
    _gold_info: LineEditInfo = None

    def get_line_probs(self) -> LineEditInfo:
        return self._precomputed_info

    def get_gold_line_info(self) -> LineEditInfo:
        if self._gold_info is None:
            self._gold_info = compute_gold_line_info(
                self.get_base_text(),
                self.get_gt_fix_text(),
            )
        return self._gold_info


@dataclass(kw_only=True)
class LineLocalizationMultis(MultiSamplingLocalization, LineLocalization):
    edit_info_samples: list[LineEditInfo] = dataclasses.field(default_factory=list)


def compute_gold_line_info(
    base_text: str,
    gt_fix_text: str,
    clean_whitespace: bool = True,
) -> LineEditInfo:
    base_text = base_text.lstrip("\n")
    gt_fix_text = gt_fix_text.lstrip("\n")
    base_lines = base_text.splitlines()
    base_lines_orig = base_lines.copy()
    fix_lines = gt_fix_text.splitlines()
    def run_clean_whitespace(lines):
        return [line.strip().replace(" ", "") for line in lines]
    if clean_whitespace:
        base_lines = run_clean_whitespace(base_lines)
        fix_lines = run_clean_whitespace(fix_lines)
    n = len(base_lines)
    rewrites = [0.0] * n
    deletes = [0.0] * n
    insert_before = [0.0] * (n + 1)
    
    matcher = difflib.SequenceMatcher(None, base_lines, fix_lines)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue  # No need to assign line_hunks, it will be base_lines
        elif tag == "replace":
            # For lines that have corresponding replacements, mark rewrite = 1.0
            common = min(i2 - i1, j2 - j1)
            for k in range(i1, i1 + common):
                rewrites[k] = 1.0
            # Extra base lines (if any) are considered deletions.
            for k in range(i1 + common, i2):
                deletes[k] = 1.0
            # If there are extra fix lines, flag an insertion at gap i2.
            if (j2 - j1) > (i2 - i1):
                insert_before[i2] = 1.0
        elif tag == "delete":
            is_only_comments_deleted = all(
                base_lines[k].strip().startswith("#")
                for k in range(i1, i2)
            )
            if is_only_comments_deleted:
                continue
            for k in range(i1, i2):
                deletes[k] = 1.0
        elif tag == "insert":
            is_only_comments_inserted = all(
                fix_lines[k].strip().startswith("#")
                for k in range(j1, j2)
            )
            if is_only_comments_inserted:
                continue
            is_only_whitespace_inserted = all(
                fix_lines[k].strip() == ""
                for k in range(j1, j2)
            )
            if is_only_whitespace_inserted:
                continue
            # Mark an insertion flag at the corresponding gap.
            insert_before[i1] = 1.0

    return LineEditInfo(
        prob_deletes=np.array(deletes),
        prob_insert_before=np.array(insert_before),
        prob_rewrites=np.array(rewrites),
        line_hunks=base_lines_orig,  # Set line_hunks to base_lines directly
    )


def _round_prob(prob):
    return min(int(prob * 10), 9)


default_rescalers = {
    "prob_insert_before": lambda x: x,
    "prob_rewrites": lambda x: x,
    "prob_deletes": lambda x: x,
}

def vis_line_localization_str(
    line_localization,
    use_gold: bool = False,
    use_color: bool = True,
    insert_show_threshold: float = 0.3,
    rescalers = default_rescalers,
) -> str:
    if use_gold:
        info = line_localization.get_gold_line_info()
    else:
        info = line_localization.get_line_probs()
    out = []

    def format_prob(v: float):
        v = _round_prob(v)
        if use_color:
            if v > 7:
                return f"\033[91m{v}\033[0m"
            if v > 4:
                return f"\033[93m{v}\033[0m"
            return str(v)

    for i in range(len(info.line_hunks) + 1):
        prob_insert_before = rescalers["prob_insert_before"](
            info.prob_insert_before[i])
        if prob_insert_before >= insert_show_threshold:
            out.append(f"   ▶{format_prob(prob_insert_before)}")
        if i >= len(info.line_hunks):
            continue
        prob_rewrites = rescalers["prob_rewrites"](info.prob_rewrites[i])
        prob_deletes = rescalers["prob_deletes"](info.prob_deletes[i])
        line = f"✗{format_prob(prob_deletes)} ↻{format_prob(prob_rewrites)} {info.line_hunks[i].rstrip()}"
        out.append(line)
    return "\n".join(out)


def print_vis_line_localization(
    line_localization: LineLocalization,
    use_gold: bool = False,
    rescalers = default_rescalers
):
    print(vis_line_localization_str(
        line_localization, use_gold, use_color=True, rescalers=default_rescalers))


def create_multis_line_version(
    localizations: LocalizationList[MultiSamplingLocalization],
) -> LocalizationList[LineLocalizationMultis]:
    new = localizations.copy_with_type_change(
        new_type=LineLocalizationMultis
    )
    dataset = new.get_only_datset_name()
    for loc in new.iter_passed_filtered():
        if loc.get_gt_fix_text() is None:
            loc.annotate_failed_filter("no_gt_fix_text")
            continue
        for solution in loc.samples.solutions:
            if (
                len(solution.solve_steps) == 0 
                # Not clear why this happens, but sometimes the values are None
                or any(step.value is None for step in solution.solve_steps)
            ):
                continue
            multi_text = solve_to_text(solution, dataset)
            if multi_text is None:
                continue
            if is_func_body_junk_solve(multi_text, None, dataset):
                continue
            val = compute_gold_line_info(
                loc.get_base_text(),
                multi_text,
            )
            if val is None:
                continue
            loc.edit_info_samples.append(val)
        if len(loc.edit_info_samples) == 0:
            loc.annotate_failed_filter("no_valid_samples")
            continue
    for loc in new.iter_passed_filtered():
        loc._precomputed_info = combine_line_infos(loc.edit_info_samples)
    return new


def combine_line_infos(infos: list[LineEditInfo]) -> LineEditInfo:
    """Take an average of the probabilities"""
    n = len(infos)
    if n == 0:
        raise ValueError("No infos to combine")
    prob_deletes = np.zeros_like(infos[0].prob_deletes)
    prob_insert_before = np.zeros_like(infos[0].prob_insert_before)
    prob_rewrites = np.zeros_like(infos[0].prob_rewrites)
    for info in infos:
        prob_deletes += info.prob_deletes
        prob_insert_before += info.prob_insert_before
        prob_rewrites += info.prob_rewrites
    prob_deletes /= n
    prob_insert_before /= n
    prob_rewrites /= n
    return LineEditInfo(
        prob_deletes=prob_deletes,
        prob_insert_before=prob_insert_before,
        prob_rewrites=prob_rewrites,
        line_hunks=infos[0].line_hunks,
    )


def get_for_all_eg_line(
    dataset: DatasetName | list[DatasetName] = DatasetName.humaneval,
    max_problems: int = None,
    max_gen_tokens: int = 1000,
    fix_reference: str = OpenAiModelNames.gpt_4,
    multi_config: MultiSamplingConfig = MultiSamplingConfig(
        multi_temperature=1.0,
        target_num_samples=10,
        mode=MultiSampleMode.repair,
    ),
    filter_to_original_fails: bool = True,
) -> LocalizationList[LineLocalizationMultis]:
    localizations_all = None
    if isinstance(dataset, list):
        datasets = dataset
    else:
        datasets = [dataset]
    for dataset in datasets:
        localizations = get_base_solves(
            use_eval_plus=True,
            dataset=dataset,
            #gen_model_name="mistralai/Mistral-7B-v0.1",
            gen_model_name=OpenAiModelNames.gpt_4o_mini,
            filter_to_original_fails=filter_to_original_fails,
            max_problems=max_problems,
            max_gen_tokens=max_gen_tokens,
        )
        print(f"base solves {localizations}")
        add_fix_data(
            localizations,
            fix_reference=fix_reference,
        )
        print(f"after add fix {localizations}")
        localizations = create_multis_version(localizations, multi_config)
        print(f"after create multis {localizations}")
        localizations = create_multis_line_version(localizations)
        print(f"after create line multis {localizations}")
        if localizations_all is None:
            localizations_all = localizations
        else:
            localizations_all.extend(localizations)
    return localizations_all


def make_rescale_and_baseline_line(
    localizations: LocalizationList[LineLocalization],
    prop: str,
):
    all_ests = []
    all_gts = []
    for loc in localizations.iter_passed_filtered():
        combined_vec = np.stack([
            getattr(loc.get_line_probs(), prop),
            getattr(loc.get_gold_line_info(), prop)
        ], axis=1)
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


def rescale_line_locs(
    localizations: LocalizationList[LineLocalization]
) -> LocalizationList[LineLocalization]:
    new_localizations = localizations.copy()
    for prop in ("prob_insert_before", "prob_rewrites", "prob_deletes"):
        rescaler, baseline = make_rescale_and_baseline_line(
            localizations,
            prop=prop,
        )
        # TODO make it deep copy right the info
        for loc in new_localizations.iter_passed_filtered():
            cur = getattr(loc._precomputed_info, prop)
            setattr(loc._precomputed_info, prop, rescaler.predict(cur))
    return new_localizations


def print_all_line_locs(localizations: LocalizationList[LineLocalization]):
    for loc in localizations.iter_passed_filtered():
        #print("- reference content")
        print(loc.base_solve.problem.working_directory.files.get_only_file().content_str)
        print("- get_base_text")
        print(loc.get_base_text())
        #print("   -- completion text")
        #print(loc.base_solve.lm_prediction[0].completion_text)
        #exit()
        print("- gold")
        print_vis_line_localization(loc, use_gold=True)
        print("- actual gold text")
        print(loc.get_gt_fix_text())
        print("- calc")
        print_vis_line_localization(loc, use_gold=False)


def main():
    localizations = get_for_all_eg_line(
        dataset=DatasetName.livecodebench,
        fix_reference=OpenAiModelNames.o3_mini,
    )
    localizations = rescale_line_locs(localizations)
    print(debug_str_filterables(localizations.iter_all()))
    exit()
    print_all_line_locs(localizations)


if __name__ == "__main__":
    main()