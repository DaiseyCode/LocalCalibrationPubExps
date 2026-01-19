from lmwrapper.huggingface_wrapper import get_huggingface_lm
from tqdm.auto import tqdm
from lmwrapper.structs import LmPrompt, LmChatTurn, ChatGptRoles
from lmwrapper.internals import ModelInternalsRequest
from localizing.fix_adders import GroundTruthFixAdder, LocalizationsFixAdder, RewriteFixAdder
from localizing.probe.agg_models.agg_config import ProbeAggregator
from localizing.problem_processing import tokenize_text
from localizing.localizing_structs import BaseLocalization, TokenizedLocalization, LocalizationList
from lmwrapper.huggingface_wrapper import HuggingFacePredictor
import torch
from dataclasses import dataclass
from localizing.multi_data_gathering import create_tokenized_localizations_from_scratch, DatasetName
from lmwrapper.openai_wrapper import OpenAiModelNames
from pathlib import Path
import pickle
import random
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from enum import Enum, auto
from typing import List, Tuple, TYPE_CHECKING
from localizing.probe.embedding_styles import EmbeddingStyle
from mem_util import check_resources_and_exit
from solve_helpers import clear_model_in_mem

if TYPE_CHECKING:
    from localizing.probe.agg_models.agg_config import ProbeConfig


@dataclass(kw_only=True)
class EmbeddedTokenization(TokenizedLocalization):
    base_tokens_embedding: torch.Tensor | None = None  # (num_tokens, embedding_dim) float16
    
    def __post_init__(self):
        super().__post_init__() if hasattr(super(), "__post_init__") else None
        if self.base_tokens_embedding is not None and hasattr(self, "gt_base_token_keeps"):
            assert self.base_tokens_embedding.shape[0] == len(self.gt_base_token_keeps), \
                f"Number of embeddings ({self.base_tokens_embedding.shape[0]}) " \
                f"must match number of tokens in gt_base_token_keeps ({len(self.gt_base_token_keeps)})"


def _add_embedding_to_localization(
    localization: EmbeddedTokenization,
    embed_lm: HuggingFacePredictor,
    embedding_style: EmbeddingStyle,
) -> None:
    if not isinstance(localization, TokenizedLocalization):
        raise ValueError(f"Localization must be a TokenizedLocalization, got {type(localization)}")

    base_text = localization.get_base_text()
    base_tokens = localization.base_tokens

    #assert localization.base_solve.prompt.is_text_a_chat
    new_prompt = localization.get_base_convo(include_last_answer=False).copy()
    # TODO use the actual convo
    new_prompt.append(
        LmChatTurn(role=ChatGptRoles.assistant, content=base_text.lstrip("\n"))
    )

    # Get the layer fractions needed for this embedding style
    hidden_layer_fractions = embedding_style.hidden_layer_fractions
    # Do a pred just to get the tokenizations
    pred = embed_lm.predict(LmPrompt(
        text=new_prompt,
        echo=True,
        max_tokens=0,
    ))
    if len(pred.prompt_tokens) > 8192 - 192:
        print(f"WARNING: lots of tokens")
        localization.annotate_failed_filter(f"too_many_tokens_in_prompt")
        return
    try:
        pred = embed_lm.predict(LmPrompt(
            text=new_prompt,
            echo=True,
            max_tokens=0,
            model_internals_request=ModelInternalsRequest(
                return_hidden_states=True,
                hidden_layer_fractions=hidden_layer_fractions,
            ),
        ))
    except torch.cuda.OutOfMemoryError as e:
        print(f"WARNING: Out of memory error when adding embedding to localization")
        localization.annotate_failed_filter(f"out_of_memory_on_token_embedding")
        torch.cuda.empty_cache()
        return
    pred_tokens = pred.prompt_tokens
    hidden_states = pred.internals.hidden_states
    # Convert numpy arrays to PyTorch tensors
    hidden_states = [torch.from_numpy(h).to(torch.float16) for h in hidden_states]
    
    # Find the start index of base_tokens in pred_tokens
    start_idx = None
    stripped_pred_tokens = [t.strip() for t in pred_tokens]
    stripped_base_tokens = [t.strip() for t in base_tokens]
    for i in range(len(pred_tokens) - len(base_tokens) + 1):
        if stripped_pred_tokens[i:i+len(base_tokens)] == stripped_base_tokens:
            start_idx = i
            break
    
    if start_idx is None:
        msg = (f"Could not find base_tokens as a subsequence in pred_tokens.\n"
                         f"dataset: {localization.dataset_name}\n"
                         f"base_text: {repr(base_text)}\n"
                         f"base_tokens: {base_tokens}\n"
                         f"pred_text: {repr(pred.completion_text)}\n"
                         f"pred_tokens: {pred_tokens}\n"
                         f"new tokenize base: {embed_lm._tokenizer.tokenize(base_text)}\n"
                         f"another tokenize base: {tokenize_text('Qwen/Qwen2.5-Coder-0.5B', base_text)}")
        print(msg)
        localization.annotate_failed_filter(f"base_tokens_not_found_in_pred_tokens")
        return localization
    
    # Prepare embeddings according to the layer-token combinations
    all_embeddings = []
    
    # Process each layer-token combination
    for layer_idx, layer_fraction in enumerate(hidden_layer_fractions):
        layer_hidden = hidden_states[layer_idx]
        
        # Find which token offsets apply to this layer
        layer_shifts = [offset for offset, frac in embedding_style.layer_token_combinations 
                       if abs(frac - layer_fraction) < 1e-6]  # Compare floats with tolerance
        
        # If no specific shifts for this layer, default to current token
        if not layer_shifts:
            layer_shifts = [0]
        
        for shift in layer_shifts:
            # Calculate start and end indices for this shift
            slice_start = start_idx + shift
            slice_end = slice_start + len(base_tokens)
            
            # Handle potential out-of-bounds conditions
            if slice_start < 0:
                raise ValueError(f"Negative slice start index: {slice_start}. Cannot extract tokens before the start of the sequence.")
            
            if slice_end > len(layer_hidden):
                raise ValueError(f"Slice end index ({slice_end}) exceeds hidden state length ({len(layer_hidden)})")

            # Extract embeddings for this layer-shift combination
            embeddings = layer_hidden[slice_start:slice_end]
            
            all_embeddings.append(embeddings)
    
    # Concatenate all embeddings along the embedding dimension
    concatenated_hidden = torch.cat(all_embeddings, dim=1)
    
    # Check if there's a mismatch between embedding tokens and ground truth tokens
    if hasattr(localization, "gt_base_token_keeps") and len(localization.gt_base_token_keeps) != len(base_tokens):
        print(f"WARNING: Token count mismatch detected!")
        print(f"  Embedding tokens: {len(base_tokens)}")
        print(f"  Ground truth tokens: {len(localization.gt_base_token_keeps)}")
        print(f"  These should match for correct model training.")
        print(f"  Prediction prompt tokens: {pred.prompt_tokens}")
        print(f"  Base tokens: {base_tokens}")
        raise ValueError("Token count mismatch detected!")
    
    # Assign the concatenated embeddings to the localization
    localization.base_tokens_embedding = concatenated_hidden
    return localization


def add_embedding_to_localization_list(
    localizations: LocalizationList[TokenizedLocalization],
    embed_lm: HuggingFacePredictor,
    embedding_style: EmbeddingStyle,
) -> LocalizationList[EmbeddedTokenization]:
    localizations = localizations.copy_with_type_change(EmbeddedTokenization)
    for i, localization in enumerate(tqdm(
        localizations.iter_passed_filtered(), 
        desc="Adding embeddings to localizations", 
        total=len(list(localizations.iter_passed_filtered())),
    )):
        if i % 100 == 0:
            check_resources_and_exit()
        _add_embedding_to_localization(localization, embed_lm, embedding_style)
    return localizations


def get_embedded_localizations(
    dataset: DatasetName,
    gen_model_name: str,
    embed_lm_name: str,
    embedding_style: EmbeddingStyle,
    fix_adder: LocalizationsFixAdder,
    filter_to_original_fails: bool = True,
    max_problems: int = 1000,
    max_gen_tokens: int = 1000,
) -> LocalizationList[EmbeddedTokenization]:
    localizations = create_tokenized_localizations_from_scratch(
        dataset=dataset,
        gen_model_name=gen_model_name,
        filter_to_original_fails=filter_to_original_fails,
        max_problems=max_problems,
        max_gen_tokens=max_gen_tokens,
        tokenizer_key=embed_lm_name,
        fix_adder=fix_adder,
    )
    clear_model_in_mem()
    embed_lm = get_huggingface_lm(embed_lm_name)
    localizations = add_embedding_to_localization_list(localizations, embed_lm, embedding_style)
    return localizations


cur_path = Path(__file__).parent
default_save_path = cur_path / "serialized_localizations"


def serialize_localizations(
    localizations: LocalizationList[EmbeddedTokenization],
    filename: Path,
) -> Path:
    filename.parent.mkdir(parents=True, exist_ok=True)
    if str(filename).endswith('.lz4'):
        import lz4.frame
        with lz4.frame.open(filename, "wb") as f:
            pickle.dump(localizations, f, protocol=5)
    else:
        with open(filename, "wb") as f:
            pickle.dump(localizations, f, protocol=5)
    return filename


def deserialize_localizations(
    filename: Path,
) -> LocalizationList[BaseLocalization] | None:
    if not filename.exists():
        return None
    if str(filename).endswith('.lz4'):
        import lz4.frame
        with lz4.frame.open(filename, "rb") as f:
            return pickle.load(f)
    else:
        with open(filename, "rb") as f:
            return pickle.load(f)


def make_basic_serialize_key_args(
    config: 'ProbeConfig',
    dev_mode: bool = False,
) -> tuple[str, ...]:
    from localizing.probe.agg_models.agg_config import ProbeConfig
    assert isinstance(config, ProbeConfig)
    assert isinstance(dev_mode, bool)
    # Convert single dataset to list for uniform processing
    datasets = config.datasets
    
    # Sort datasets for consistent filename generation
    datasets = sorted(datasets, key=lambda d: str(d))
    
    # Create a filename based on all datasets
    datasets_str = "AND".join(str(d) for d in datasets)
    fix_key = config.fix_style.value
    import hashlib
    name_hash = hashlib.md5(config.fix_style.create_fix_adder().name.encode()).hexdigest()[:5]
    fix_key += f"{name_hash}"
    default_save_path.mkdir(parents=True, exist_ok=True)
    args = (
        datasets_str,
        config.gen_model_name,
        config.max_problems,
        config.max_gen_tokens,
        fix_key,
        config.filter_to_original_fails,
        dev_mode,
    )
    return tuple(str(a) for a in args)


def get_or_serialize_tokenized_localizations(
    config: 'ProbeConfig',
    dev_mode: bool = False,
) -> LocalizationList[TokenizedLocalization]:
    args = make_basic_serialize_key_args(config, dev_mode)
    fn = default_save_path / "tokenized" / ("_".join(args) + ".pkl.lz4")

    combined_localizations = deserialize_localizations(fn)
    if combined_localizations is not None:
        return combined_localizations

    # See if the nondev one exissts
    args = make_basic_serialize_key_args(config, dev_mode=False)
    full_fn = default_save_path / "tokenized" / ("_".join(args) + ".pkl.lz4")
    if full_fn.exists():
        locs = deserialize_localizations(full_fn)
        if locs is not None:
            combined_localizations = locs
    
    
    if combined_localizations is None: # Otherwise we are dev mode and grabbed it
        for dataset_name in config.datasets:
            print(f"Processing dataset: {dataset_name}")
            if dataset_name == DatasetName.repocod:
                max_problems = min(config.repo_cod_max_problems, config.max_problems)
            else:
                max_problems = config.max_problems

            # Do it here
            current_localizations = create_tokenized_localizations_from_scratch(
                dataset=dataset_name,
                gen_model_name=config.gen_model_name,
                filter_to_original_fails=config.filter_to_original_fails,
                max_problems=max_problems,
                max_gen_tokens=config.max_gen_tokens,
                tokenizer_key=config.embed_lm_name,
                fix_adder=config.fix_style.create_fix_adder(dataset_name),
            )
            # Initialize or extend the combined localizations
            if combined_localizations is None:
                combined_localizations = current_localizations
            else:
                combined_localizations.extend(current_localizations)
    if dev_mode:
        rng = random.Random(42)
        combined_localizations = LocalizationList(rng.sample(
            list(combined_localizations.iter_all()), 
            int(len(combined_localizations) * config.dev_size_frac)
        ))
    serialize_localizations(combined_localizations, fn)
    return combined_localizations


def make_basic_serialize_key_args_embedded(
    config: 'ProbeConfig',
    dev_mode: bool = False,
) -> tuple[str, ...]:
    from localizing.probe.agg_models.agg_config import ProbeConfig
    assert isinstance(config, ProbeConfig)
    assert isinstance(dev_mode, bool)
    args = make_basic_serialize_key_args(config, dev_mode)
    args = (*args, config.embed_lm_name, config.embedding_style.value)

    return tuple(str(a) for a in args)

def get_or_serialize_localizations_embedded(
    config: 'ProbeConfig',
    dev_mode: bool = False,
    save_serialize_embeddings: bool = False,
) -> LocalizationList[EmbeddedTokenization]:
    """
    Get or serialize localizations for one or more datasets using ProbeConfig.
    
    Args:
        config: ProbeConfig instance containing all necessary parameters
        dev_mode: Whether to use dev mode which is a random sample of the localizations
        save_serialize_embeddings: Whether to save the serialized embeddings
    Returns:
        LocalizationList containing localizations for all requested datasets
    """
    assert isinstance(save_serialize_embeddings, bool)
    args = make_basic_serialize_key_args_embedded(config, dev_mode)
    fn = default_save_path / "embedded" / ("_".join(args) + ".pkl.lz4")
    combined_localizations = deserialize_localizations(fn)
    if combined_localizations is not None:
        return combined_localizations
    print("Getting tokenized localizations")
    combined_localizations = get_or_serialize_tokenized_localizations(
        config, dev_mode)
    print("Adding embeddings to localizations")
    print("load models")
    clear_model_in_mem()
    embed_lm = get_huggingface_lm(
        config.embed_lm_name,
        precision=torch.float16,
    )
    print("add embeddings")
    combined_localizations = add_embedding_to_localization_list(
        combined_localizations, 
        embed_lm, 
        config.embedding_style,
    )
    if save_serialize_embeddings:
        serialize_localizations(combined_localizations, fn)
    
    return combined_localizations


@dataclass(kw_only=True)
class SingleVecToLabelDataset(Dataset):
    embeddings: torch.Tensor
    labels: torch.Tensor
    
    def __post_init__(self):
        assert self.embeddings.ndim == 2, f"Embeddings must be 2-dimensional, got shape {self.embeddings.shape}"
        assert self.embeddings.shape[0] == self.labels.shape[0], f"Number of embeddings ({self.embeddings.shape[0]}) must match number of labels ({self.labels.shape[0]})"
        assert self.labels.ndim == 1, f"Labels must be a vector (1-dimensional), got shape {self.labels.shape}"
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


@dataclass(kw_only=True)
class SingleVecToLabelDatasetMultiFold:
    folds_train_test: tuple[tuple[SingleVecToLabelDataset, SingleVecToLabelDataset], ...]
    
    def __str__(self) -> str:
        num_folds = len(self.folds_train_test)
        if num_folds == 0:
            return "SingleVecToLabelDatasetMultiFold: No folds available"
        
        # Get info from first fold to represent the dataset
        first_train, first_test = self.folds_train_test[0]
        embedding_dim = first_train.embeddings.shape[1]
        
        total_train_samples = sum(train.embeddings.shape[0] for train, _ in self.folds_train_test)
        total_test_samples = sum(test.embeddings.shape[0] for _, test in self.folds_train_test)
        avg_train_samples = total_train_samples // num_folds
        avg_test_samples = total_test_samples // num_folds
        
        return (f"SingleVecToLabelDatasetMultiFold: {num_folds} folds\n"
                f"  Embedding dimension: {embedding_dim}\n"
                f"  Average samples per fold: {avg_train_samples} train, {avg_test_samples} test")
    
    def __repr__(self) -> str:
        num_folds = len(self.folds_train_test)
        fold_details = []
        
        for i, (train, test) in enumerate(self.folds_train_test):
            fold_details.append(
                f"Fold {i+1}: {train.embeddings.shape[0]} train samples, "
                f"{test.embeddings.shape[0]} test samples"
            )
        
        fold_info = "\n  ".join(fold_details)
        return (f"SingleVecToLabelDatasetMultiFold(folds={num_folds}, "
                f"details=\n  {fold_info})")


@dataclass(kw_only=True)
class GroupedVecLabelDataset(Dataset):
    localizations: list[EmbeddedTokenization]
    
    def __len__(self):
        return len(self.localizations)
    
    def __getitem__(self, idx):
        return self.localizations[idx]


@dataclass(kw_only=True)
class GroupedVecLabelDatasetMultiFold:
    folds_train_test: tuple[tuple[GroupedVecLabelDataset, GroupedVecLabelDataset], ...]
    
    def __str__(self) -> str:
        num_folds = len(self.folds_train_test)
        if num_folds == 0:
            return "GroupedVecLabelDatasetMultiFold: No folds available"
        
        # Get info from first fold to represent the dataset
        first_train, first_test = self.folds_train_test[0]
        
        total_train_samples = sum(len(train.localizations) for train, _ in self.folds_train_test)
        total_test_samples = sum(len(test.localizations) for _, test in self.folds_train_test)
        avg_train_samples = total_train_samples // num_folds
        avg_test_samples = total_test_samples // num_folds
        
        return (f"GroupedVecLabelDatasetMultiFold: {num_folds} folds\n"
                f"  Average samples per fold: {avg_train_samples} train, {avg_test_samples} test")
    
    def __repr__(self) -> str:
        num_folds = len(self.folds_train_test)
        fold_details = []
        
        for i, (train, test) in enumerate(self.folds_train_test):
            fold_details.append(
                f"Fold {i+1}: {len(train.localizations)} problem samples, "
                f"{len(test.localizations)} problem samples"
            )
        
        fold_info = "\n  ".join(fold_details)
        return (f"GroupedVecLabelDatasetMultiFold(folds={num_folds}, "
                f"details=\n  {fold_info})")


def localizations_to_grouped_vec_label_dataset_custom_split(
    train_localizations: LocalizationList[EmbeddedTokenization],
    test_localizations: LocalizationList[EmbeddedTokenization],
) -> tuple[GroupedVecLabelDataset, GroupedVecLabelDataset]:
    """
    Create train and test datasets from specific train and test localizations.
    
    Args:
        train_localizations: LocalizationList to use for training data
        test_localizations: LocalizationList to use for test data
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    # Get all localizations that passed filters
    train_locs = list(train_localizations.iter_passed_filtered())
    test_locs = list(test_localizations.iter_passed_filtered())
    
    # Create train and test datasets
    train_dataset = GroupedVecLabelDataset(localizations=train_locs)
    test_dataset = GroupedVecLabelDataset(localizations=test_locs)
    
    return train_dataset, test_dataset


def localizations_to_grouped_vec_label_dataset(
    localizations: LocalizationList[EmbeddedTokenization],
    n_folds: int,
) -> GroupedVecLabelDatasetMultiFold:
    # Get all localizations that passed filters
    passed_locs = list(localizations.iter_passed_filtered())
    
    # Create KFold splitter
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Create train/test splits for each fold
    train_test_folds = []
    for train_idx, test_idx in kf.split(passed_locs):
        # Get train and test localizations
        train_locs = [passed_locs[i] for i in train_idx]
        test_locs = [passed_locs[i] for i in test_idx]
        
        # Create train and test datasets
        train_dataset = GroupedVecLabelDataset(localizations=train_locs)
        test_dataset = GroupedVecLabelDataset(localizations=test_locs)
        
        train_test_folds.append((train_dataset, test_dataset))
    
    return GroupedVecLabelDatasetMultiFold(
        folds_train_test=tuple(train_test_folds)
    )


def localizations_to_single_vec_to_label_dataset_custom_split(
    train_localizations: LocalizationList[EmbeddedTokenization],
    test_localizations: LocalizationList[EmbeddedTokenization],
) -> tuple[SingleVecToLabelDataset, SingleVecToLabelDataset]:
    """
    Create train and test datasets from specific train and test localizations.
    
    Args:
        train_localizations: LocalizationList to use for training data
        test_localizations: LocalizationList to use for test data
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    # Helper function to create dataset from list of localizations
    def create_dataset(localizations: LocalizationList[EmbeddedTokenization]):
        locs = list(localizations.iter_passed_filtered())
        embeddings = []
        labels = []
        for i, loc in enumerate(locs):
            if loc.base_tokens_embedding is not None:
                x, y = loc.base_tokens_embedding, torch.tensor(loc.gt_base_token_keeps, dtype=torch.float16)
                try:
                    assert x.shape[0] == y.shape[0], \
                        f"Number of embeddings ({x.shape[0]}) must match number of labels ({y.shape[0]})"
                    embeddings.append(x)
                    labels.append(y)
                except AssertionError as e:
                    print(f"Error at localization {i}:")
                    print(f"base_tokens length: {len(loc.base_tokens) if hasattr(loc, 'base_tokens') else 'N/A'}")
                    print(f"gt_base_token_keeps length: {len(loc.gt_base_token_keeps)}")
                    print(f"base_tokens_embedding shape: {loc.base_tokens_embedding.shape}")
                    raise e
        
        if not embeddings:  # Handle case where no valid embeddings found
            return None
            
        return SingleVecToLabelDataset(
            embeddings=torch.cat(embeddings, dim=0),
            labels=torch.cat(labels, dim=0)
        )
    
    # Create train and test datasets
    train_dataset = create_dataset(train_localizations)
    test_dataset = create_dataset(test_localizations)
    
    if train_dataset is None or test_dataset is None:
        raise ValueError("Unable to create valid datasets - no valid embeddings found")
    
    return train_dataset, test_dataset


def localizations_to_single_vec_to_label_dataset(
    localizations: LocalizationList[EmbeddedTokenization],
    n_folds: int,
) -> SingleVecToLabelDatasetMultiFold:
    # Get all localizations that passed filters
    passed_locs = list(localizations.iter_passed_filtered())
    
    # Create KFold splitter
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Create train/test splits for each fold
    train_test_folds = []
    for train_idx, test_idx in kf.split(passed_locs):
        # Get train and test localizations
        train_locs = [passed_locs[i] for i in train_idx]
        test_locs = [passed_locs[i] for i in test_idx]
        
        # Helper function to create dataset from list of localizations
        def create_dataset(locs):
            embeddings = []
            labels = []
            for i, loc in enumerate(locs):
                if loc.base_tokens_embedding is not None:
                    x, y = loc.base_tokens_embedding, torch.tensor(loc.gt_base_token_keeps, dtype=torch.float16)
                    try:
                        assert x.shape[0] == y.shape[0], \
                            f"Number of embeddings ({x.shape[0]}) must match number of labels ({y.shape[0]})"
                        embeddings.append(x)
                        labels.append(y)
                    except AssertionError as e:
                        print(f"Error at localization {i}:")
                        print(f"base_tokens length: {len(loc.base_tokens) if hasattr(loc, 'base_tokens') else 'N/A'}")
                        print(f"gt_base_token_keeps length: {len(loc.gt_base_token_keeps)}")
                        print(f"base_tokens_embedding shape: {loc.base_tokens_embedding.shape}")
                        raise e
            
            if not embeddings:  # Handle case where no valid embeddings found
                return None
                
            return SingleVecToLabelDataset(
                embeddings=torch.cat(embeddings, dim=0),
                labels=torch.cat(labels, dim=0)
            )
        
        # Create train and test datasets
        train_dataset = create_dataset(train_locs)
        test_dataset = create_dataset(test_locs)
        
        if train_dataset is not None and test_dataset is not None:
            train_test_folds.append((train_dataset, test_dataset))
    
    return SingleVecToLabelDatasetMultiFold(
        folds_train_test=tuple(train_test_folds)
    )


if __name__ == "__main__":
    # Import ProbeConfig for the examples
    from localizing.probe.agg_models.agg_config import ProbeConfig
    
    # Create a default config for testing
    config = ProbeConfig(embedding_style=EmbeddingStyle.LAST_LAYER)
    
    # Example using the default last_layer style
    localizations_last = get_or_serialize_localizations_embedded(config)
    print("Locs (Last Layer):")
    # Print info for the first few localizations to avoid too much output
    for i, loc in enumerate(localizations_last.iter_passed_filtered()):
        if i >= 2: break
        print("-" * 80)
        print(f"Base Tokens ({len(loc.base_tokens)}): {loc.base_tokens[:10]}...")
        print(f"Embedding Shape: {loc.base_tokens_embedding.shape}")
        print(f"GT Keeps ({len(loc.gt_base_token_keeps)}): {loc.gt_base_token_keeps[:10]}...")

    dataset_last = localizations_to_single_vec_to_label_dataset(localizations_last, 5)
    print(dataset_last)

    # Example using the three_layer style
    config_three = ProbeConfig(embedding_style=EmbeddingStyle.THREE_LAYER)
    localizations_three = get_or_serialize_localizations_embedded(config_three)
    print("Locs (Three Layer):")
    for i, loc in enumerate(localizations_three.iter_passed_filtered()):
        if i >= 2: break
        print("-" * 80)
        print(f"Base Tokens ({len(loc.base_tokens)}): {loc.base_tokens[:10]}...")
        print(f"Embedding Shape: {loc.base_tokens_embedding.shape}") # Note the embedding dim will be different
        print(f"GT Keeps ({len(loc.gt_base_token_keeps)}): {loc.gt_base_token_keeps[:10]}...")

    dataset_three = localizations_to_single_vec_to_label_dataset(localizations_three, 5)
    print(dataset_three)
    
    # Example using the shifted_three_quarters style
    config_shifted = ProbeConfig(embedding_style=EmbeddingStyle.SHIFTED_THREE_QUARTERS)
    localizations_shifted = get_or_serialize_localizations_embedded(config_shifted)
    print("Locs (Shifted Three Quarters):")
    for i, loc in enumerate(localizations_shifted.iter_passed_filtered()):
        if i >= 2: break
        print("-" * 80)
        print(f"Base Tokens ({len(loc.base_tokens)}): {loc.base_tokens[:10]}...")
        print(f"Embedding Shape: {loc.base_tokens_embedding.shape}")
        print(f"GT Keeps ({len(loc.gt_base_token_keeps)}): {loc.gt_base_token_keeps[:10]}...")
        
    # Print out different embedding style properties for comparison
    print("\nEmbedding Style Properties:")
    for style in EmbeddingStyle:
        print(f"{style.value}:")
        print(f"  Layer Fractions: {style.hidden_layer_fractions}")
        print(f"  Shift Offsets: {style.shift_offsets}")
        print(f"  Includes Shifted: {style.includes_shifted_embeddings}")
        print(f"  Layer-Token Combinations: {style.layer_token_combinations}")
        print()
        
    # Example using GroupedVecLabelDataset
    print("\nTesting GroupedVecLabelDataset...")
    grouped_dataset = localizations_to_grouped_vec_label_dataset(localizations_last, 5)
    print(grouped_dataset)
    
    # Example accessing individual localizations in the dataset
    train_dataset, test_dataset = grouped_dataset.folds_train_test[0]
    print(f"\nFirst fold train dataset size: {len(train_dataset)}")
    print(f"First fold test dataset size: {len(test_dataset)}")
    
    # Access a single localization
    if len(train_dataset) > 0:
        first_loc = train_dataset[0]
        print(f"\nSample localization from train dataset:")
        print(f"Base Tokens Count: {len(first_loc.base_tokens)}")
        print(f"Embedding Shape: {first_loc.base_tokens_embedding.shape if first_loc.base_tokens_embedding is not None else 'None'}")
        print(f"GT Keeps Count: {len(first_loc.gt_base_token_keeps) if hasattr(first_loc, 'gt_base_token_keeps') else 'N/A'}")