from lmwrapper.structs import LmChatDialog
from lmwrapper.huggingface_wrapper import get_huggingface_lm
from dataclasses import dataclass
from synthegrator.synthdatasets import DatasetName, DatasetSpec
from localizing.localizing_structs import BaseLocalization, LocalizationList
from localizing.multi_data_gathering import create_tokenized_localizations
import numpy as np

from localizing.probe.embedding_styles import EmbeddingStyle
from localizing.probe.probe_data_gather import add_embedding_to_localization_list


# Hacky way to extend DatasetName with custom datasets
# Add a toy dataset to the DatasetName class
nonsynth_name = DatasetName.nonsynth = DatasetSpec(
    name="NonSynthDataset",
    display_name="Non-Synthegrator Dataset",
    is_a_base_collection=True,
)


def make_localization_from_fixed(
    conversation: LmChatDialog,
    dataset_name: DatasetName,
    base_text: str,
    fix_text: str = None,
) -> BaseLocalization:
    return BaseLocalization(
        dataset_name=dataset_name,
        preset_full_convo=conversation,
        preset_base_text=base_text,
        preset_gt_fix_text=fix_text,
        base_solve=None,
        gen_model_name=None,
    )


def test_simple_make_localization():
    loc = make_localization_from_fixed(
        conversation=LmChatDialog([
            "What is your favorite color?",
            "My favorite color is blue.",
        ]),
        dataset_name=nonsynth_name,
        base_text="My favorite color is blue.",
        fix_text="My favorite color is red.",
    ) 
    assert not loc.is_base_success
    loc_list = LocalizationList([loc])
    locs = create_tokenized_localizations(
        loc_list,
        tokenizer_key="Qwen/Qwen2.5-Coder-0.5B",
    )
    passed_locs = list(locs.iter_passed_filtered())
    assert len(passed_locs) == 1
    for loc in passed_locs:
        print(loc.base_tokens)
        assert loc.base_tokens == ["My", " favorite", " color", " is", " blue", "."]
        assert np.array_equal(loc.gt_base_token_keeps, [1, 1, 1, 1, 0, 1])
    embed_lm = get_huggingface_lm("Qwen/Qwen2.5-Coder-0.5B")
    embedding_style = EmbeddingStyle.LAST_LAYER
    locs = list(add_embedding_to_localization_list(
        locs, embed_lm, embedding_style).iter_passed_filtered())
    assert len(locs) == 1
    print(locs[0].base_tokens_embedding)


def main():
    test_simple_make_localization()
    print("Done")


if __name__ == "__main__":
    main()