from lmwrapper.openai_wrapper import OpenAiModelNames

from localizing.evaling import make_rescale_and_baseline
from localizing.filter_helpers import debug_str_filterables
from localizing.localizing_structs import MultiSamplingConfig, MultiSampleMode
from localizing.multi_data_gathering import multis_equals_from_scratch, FIX_REFERENCE_GT
from multi_compare import visualize_probs
from synthegrator.synthdatasets import DatasetName

if __name__ == "__main__":
    #localizations = multis_equals_from_scratch()
    localizations = multis_equals_from_scratch(
        dataset=DatasetName.dypy_line_completion,
        gen_model_name=OpenAiModelNames.gpt_3_5_turbo_instruct,
        #fix_reference=OpenAiModelNames.o3_mini,
        fix_reference=FIX_REFERENCE_GT,
        max_problems=200,
        multi_config=MultiSamplingConfig(
            multi_temperature=1.0,
            target_num_samples=10,
            #mode=MultiSampleMode.repair,
            mode=MultiSampleMode.from_prompt,
        ),
    )
    print(localizations)
    print(debug_str_filterables(localizations.iter_all()))
    rescaler, _ = make_rescale_and_baseline(
        localizations,
    )
    for loc in localizations.iter_passed_filtered():
        content = loc.base_solve.problem.working_directory.files.get_only_file().content_str
        if len(content.split("\n")) > 11:
            content = "\n".join(content.split("\n")[-10:])
        print(content)
        visualize_probs(
            loc.base_tokens,
            rescaler.predict(loc.estimated_keeps),
        )
        print("w/ gts")
        visualize_probs(
            loc.base_tokens,
            rescaler.predict(loc.estimated_keeps),
            loc.gt_base_token_keeps,
        )
        print("gt")
        print(loc.get_gt_fix_text())
        print("---")