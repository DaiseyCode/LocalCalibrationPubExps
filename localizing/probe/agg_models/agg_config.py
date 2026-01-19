"""Configuration constants and settings for probe training."""

from dataclasses import dataclass
from typing import List
from enum import Enum, auto
from lmwrapper.utils import StrEnum
from lmwrapper.claude_wrapper import ClaudeModelNames
from localizing.fix_adders import CharEditDistanceBasedScorer, GroundTruthFixAdder, LocalizationsFixAdder, PickBestFixAdder, RewriteFixAdder, TaggedEditFixAdder
from localizing.multi_data_gathering import DatasetName
from lmwrapper.openai_wrapper import OpenAiModelNames
from localizing.probe.embedding_styles import EmbeddingStyle


class FixReferenceMode(StrEnum):
    """Mode for how to handle fix reference."""
    STANDARD_FIX_ADDER = "standard_fix_adder_2"

    def create_fix_adder(self, dataset: DatasetName = None) -> LocalizationsFixAdder:
        if self == FixReferenceMode.STANDARD_FIX_ADDER:
            adders = [
                TaggedEditFixAdder(
                    fix_reference=OpenAiModelNames.o4_mini_2025_04_16,
                ),
                TaggedEditFixAdder(
                    fix_reference=ClaudeModelNames.claude_4_sonnet,
                ),
            ]
            if dataset not in (DatasetName.livecodebench,):
                adders.append(GroundTruthFixAdder())
            if dataset in (DatasetName.livecodebench, DatasetName.humaneval_plus, DatasetName.mbpp_plus):
                adders.append(RewriteFixAdder(
                    fix_reference=OpenAiModelNames.o4_mini_2025_04_16,
                ))
                #adders.append(RewriteFixAdder(
                #    fix_reference=ClaudeModelNames.claude_4_sonnet,
                #))
            return PickBestFixAdder(
                fix_adders=adders,
                scorer=CharEditDistanceBasedScorer(),
            )
        else:
            raise ValueError(f"Invalid fix reference mode: {mode}")


class ProbeAggregator(StrEnum):
    """Aggregator for probe training."""
    MAX_POOL = "max_pool"
    AVG_POOL = "avg_pool"
    MHA_POOL_4_HEADS = "mha_pool_4_heads"
    

class ProbeLoss(StrEnum):
    """Loss for probe training."""
    BCE = "bce"
    FL2 = "fl2"
    """Focal loss with gamma=2 (the default)"""
    BRIER = "brier"


class FoldMode(StrEnum):
    cross_fold = "crossfold"
    dataset_fold = "datasetfold"
    

@dataclass(frozen=True, kw_only=True)
class ProbeConfig:
    """Configuration for probe training with sensible defaults."""

    # Eval params
    n_folds: int = 5
    fold_mode: FoldMode = FoldMode.cross_fold
    
    # Training hyperparameters
    learning_rate: float = 0.001
    num_epochs: int = 30
    weight_decay: float = 0.01
    batch_size: int = 16
    hidden_dim: int = 32
    dropout_rate: float = 0.2
    
    # Loss weights
    token_weight: float = 0.33
    line_weight: float = 0.34
    problem_weight: float = 0.33
    
    # Early stopping
    early_stopping_patience: int = 10
    
    # Data settings
    max_problems: int = 1200
    repo_cod_max_problems: int = 200

    max_gen_tokens: int = 1000
    filter_to_original_fails: bool = False
    embedding_style: EmbeddingStyle = EmbeddingStyle.MIDDLE_LAYER
    #embedding_style: EmbeddingStyle = EmbeddingStyle.THREE_QUARTERS_LAYER
    agg_style: ProbeAggregator = ProbeAggregator.MAX_POOL
    #loss_style: ProbeLoss = ProbeLoss.BCE
    loss_style: ProbeLoss = ProbeLoss.BRIER
    
    # Default datasets
    datasets: List[DatasetName] = None
    
    # Default models
    gen_model_name: str = OpenAiModelNames.gpt_4o
    embed_lm_name: str = "Qwen/Qwen2.5-Coder-0.5B"
    fix_style: FixReferenceMode = FixReferenceMode.STANDARD_FIX_ADDER
    
    # Visualization
    save_plots: bool = True

    dev_size_frac: float = 0.2

    @property
    def tidy_gen_name_code(self) -> str:
        return {
            OpenAiModelNames.gpt_4o: "gpt4o",
            "Qwen/Qwen2.5-Coder-7B-Instruct": "qwen7b",
        }.get(self.gen_model_name, self.gen_model_name)
    

    @property
    def tidy_gen_name_human(self) -> str:
        return {
            OpenAiModelNames.gpt_4o: "GPT-4o",
            "Qwen/Qwen2.5-Coder-7B-Instruct": "Qwen2.5Code-7B-Inst",
        }.get(self.gen_model_name, self.gen_model_name)
    
    def __post_init__(self):
        if self.datasets is None:
            object.__setattr__(self, 'datasets', [
                DatasetName.humaneval_plus,
                DatasetName.livecodebench,
                DatasetName.mbpp_plus,
                #DatasetName.repocod,
            ])