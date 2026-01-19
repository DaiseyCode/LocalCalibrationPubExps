import datasets
from lmwrapper.structs import LmChatDialog
from pprint import pprint
from localizing.localizing_structs import BaseLocalization, LocalizationList
from synthegrator.synthdatasets import DatasetName, DatasetSpec
from localizing.multi_data_gathering import create_tokenized_localizations
from localizing.nonsynth_localization import nonsynth_name
from localizing.probe.embedding_styles import EmbeddingStyle
from localizing.probe.probe_data_gather import EmbeddedTokenization, add_embedding_to_localization_list
from lmwrapper.huggingface_wrapper import get_huggingface_lm
from diskcache import Cache
import solve_helpers


cache = Cache("halu_eval_cache")


halu_eval_name = DatasetName.halueval = DatasetSpec(
    name="HaluEval",
    display_name="HaluEval",
    base_collection=nonsynth_name,
    is_a_base_collection=False,
)

def get_halu_eval_base_locs(
    max_problems: int = 100,
) -> LocalizationList[BaseLocalization]:
    dataset = datasets.load_dataset(
        "PatronusAI/HaluBench", split="test")
    dataset = dataset.shuffle(seed=42).select(range(max_problems))
    locs = []
    for i, example in enumerate(dataset):
        id = example["id"]
        passage = example["passage"]
        answer = example["answer"]
        if isinstance(answer, list):
            answer = answer[0]
        label = example["label"]
        assert label in ["FAIL", "PASS"]
        label = label == "PASS"
        source_ds = example["source_ds"]
        question = example["question"]
        prompt_text = f"Passage:\n```text\n{passage}\n```\nQuestion: {question}"
        resp_base = f"Answer: {answer}"
        if label:
            resp_fix = f"Answer: {answer}"
        else:
            resp_fix = f"Answer: fail"

        convo = LmChatDialog([
            prompt_text,
            resp_base,
        ])
        #print("--")
        #print(resp_fix)
        #print(convo)
        #print(label)
        #print(answer)
        locs.append(BaseLocalization(
            dataset_name=halu_eval_name,
            gen_model_name=None,
            base_solve=None,
            preset_base_text=resp_base,
            preset_gt_fix_text=resp_fix,
            preset_full_convo=convo,
        ))
    return LocalizationList(locs)


@cache.memoize()
def get_embedded_halu_eval(
    max_problems: int,
    embed_lm_name: str,
    embedding_style: EmbeddingStyle,
) -> LocalizationList[EmbeddedTokenization]:
    halu = get_halu_eval_base_locs(max_problems=max_problems)
    print("tokenizing")
    halu = create_tokenized_localizations(
        halu,
        tokenizer_key=embed_lm_name,
    )
    print("embedding")
    embed_lm = solve_helpers.get_model_in_mem(embed_lm_name)
    halu = add_embedding_to_localization_list(
        halu, embed_lm, embedding_style=embedding_style)
    return halu


def main():
    dataset = datasets.load_dataset(
        "PatronusAI/HaluBench", streaming=True, split="test")
    dataset = dataset.take(10)
    
    for i, example in enumerate(dataset):
        print(f"Example {i+1}:")
        # Print just the keys to see the structure
        print(f"Keys: {list(example.keys())}")
        # Print a sample of the data (first few characters if it's text)
        for key, value in example.items():
            if isinstance(value, str):
                print(f"  {key}: {value}")
            else:
                pprint(value)
        print()


if __name__ == "__main__":
    #main()
    get_halu_eval_base_locs()