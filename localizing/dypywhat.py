import types
import dataclasses
from synthegrator.synthdatasets.dypybench import yield_dypybench
from lmwrapper.abstract_predictor import LmPredictor
from lmwrapper.structs import LmPrediction, LmPrompt
from synthegrator.code_problems import CodeProblem
from lmwrapper.openai_wrapper import get_open_ai_lm, OpenAiModelNames
from synthegrator.code_solver import LmCodeSolverAutoRegressive, LmBasedSolver
from synthegrator.synthdatasets import DatasetName
from localizing.filter_helpers import debug_str_filterables
from lmwrapper.claude_wrapper.wrapper import ClaudeModelNames

from localizing.fix_adders import CharEditDistanceBasedScorer, GroundTruthFixAdder, MakeMoreMinimalFixAdder, PickBestFixAdder, RewriteFixAdder, SolverFixAdder, TaggedEditFixAdder
from localizing.localizing_structs import LocalizationList
from localizing.multi_data_gathering import get_base_solves
from localizing.probe.probe_models_agg import train_agg_line_model_on_folds
from synthegrator.problem_rendering_insertion_tags import TaggedEditRenderer, TaggedEditResponseParser, LmTaggedEditPrompt

from protogrator import LmHackSolver


def explore_adder_thing():
    #localizations = create_tokenized_localizations_from_scratch(
    #    #dataset=DatasetName.dypy_line_completion,
    #    dataset=DatasetName.livecodebench,
    #    gen_model_name=OpenAiModelNames.gpt_4_1_mini,
    #    filter_to_original_fails=True,
    #    max_problems=30,
    #    max_gen_tokens=1000,
    #    fix_reference=OpenAiModelNames.o4_mini,
    #    tokenizer_key="Qwen/Qwen2.5-Coder-0.5B",
    #)
    combined_localizations = LocalizationList()

    for dataset in [DatasetName.mbpp_plus, DatasetName.humaneval_plus, DatasetName.livecodebench]:
        localizations = get_base_solves(
                dataset=dataset,
            #dataset=DatasetName.dypy_line_completion,
            gen_model_name=OpenAiModelNames.gpt_4_1_mini,
            filter_to_original_fails=False,
            max_problems=200,
            max_gen_tokens=1000,
        )
        print("LOCALIZATIONS")
        print(localizations)
        print(debug_str_filterables(localizations.iter_all()))
        pick_best_solver = MakeMoreMinimalFixAdder(
            PickBestFixAdder(
                fix_adders=[
                    RewriteFixAdder(
                        fix_reference=OpenAiModelNames.o4_mini,
                    ),
                    RewriteFixAdder(
                        fix_reference=OpenAiModelNames.o3,
                    ),
                    RewriteFixAdder(
                        fix_reference=ClaudeModelNames.claude_4_sonnet,
                    ),
                    GroundTruthFixAdder(),
                ],
                scorer=CharEditDistanceBasedScorer(),
            ),
            lm=get_open_ai_lm(OpenAiModelNames.o4_mini),
        )
        pick_best_solver = RewriteFixAdder(
            fix_reference=OpenAiModelNames.o4_mini,
        )

        localizations = pick_best_solver.add_fix_data(localizations)

        print("Localizations after add_fix_data")
        print(localizations)
        print(debug_str_filterables(localizations.iter_all()))
        combined_localizations.extend(localizations)
    
    print(combined_localizations)
    print("Combined localizations")
    print(debug_str_filterables(combined_localizations.iter_all()))


def explore_fix_solver():
    localizations = get_base_solves(
        #dataset=DatasetName.humaneval_plus,
        dataset=DatasetName.repocod,
        #dataset=DatasetName.livecodebench,
        gen_model_name=OpenAiModelNames.gpt_4o,
        filter_to_original_fails=True,
        max_problems=20,
        max_gen_tokens=1000,
    )
    print("LOCALIZATIONS")
    print(debug_str_filterables(localizations.iter_all()))
    exit()
    adder = TaggedEditFixAdder(
        fix_reference=OpenAiModelNames.o4_mini,
        #fix_reference=OpenAiModelNames.o3,
    )
    #adder = GroundTruthFixAdder()
    #adder = RewriteFixAdder(
    #    fix_reference=OpenAiModelNames.o4_mini,
    #)
    #adder = PickBestFixAdder(
    #    fix_adders=[
    #        TaggedEditFixAdder(
    #            fix_reference=OpenAiModelNames.o4_mini,
    #        ),
    #        GroundTruthFixAdder(),
    #    ],
    #    scorer=CharEditDistanceBasedScorer(),
    #)

    localizations = adder.add_fix_data(localizations)
    print("LOCALIZATIONS")
    print(debug_str_filterables(localizations.iter_all()))
    for loc in localizations.iter_passed_filtered():
        if loc.get_base_text() == loc.get_gt_fix_text():
            continue
        print("Base text")
        print(loc.get_base_text())
        print("Fix text")
        print(loc.get_gt_fix_text())
        print("--------------------------------")



if __name__ == "__main__":
    explore_fix_solver()
