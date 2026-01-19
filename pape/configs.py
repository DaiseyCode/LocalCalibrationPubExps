from synthegrator.synthdatasets import DatasetName
import dataclasses
from lmwrapper.openai_wrapper import OpenAiModelNames
from localizing.localizing_structs import MultiSampleMode, MultiSamplingConfig
from localizing.probe.agg_models.agg_config import ProbeConfig


_datasets = [
    DatasetName.humaneval_plus,
    DatasetName.livecodebench,
    DatasetName.mbpp_plus,
    DatasetName.repocod,
]
_gen_model_name = OpenAiModelNames.gpt_4o
gen_model_name_qwen = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_LINE_AGGREGATOR_INTRINSIC = "gmean"
DEFAULT_LINE_AGGREGATOR_MULTIS = "gmean"


BASE_PAPER_CONFIG = ProbeConfig(
    datasets=_datasets,
    gen_model_name=_gen_model_name,
    max_problems=1200,
    repo_cod_max_problems=200,
)

BASE_PAPER_CONFIG_QWEN = dataclasses.replace(BASE_PAPER_CONFIG,
    gen_model_name=gen_model_name_qwen,
)

BASE_MULTIS_CONFIG = MultiSamplingConfig(
    mode=MultiSampleMode.from_prompt,
    multi_temperature=0.8,
    target_num_samples=5,
)