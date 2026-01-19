"""
Heuristic feature extraction for code localization.

This module provides a heuristic baseline for measuring how much performance 
comes from model embeddings versus simple heuristics. It mirrors the structure
of the neural embedding pipeline but uses hand-crafted features instead.
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Tuple
from collections import Counter
import keyword
import re
from pathlib import Path
import pickle
from sklearn.model_selection import KFold
from tqdm import tqdm
from torch.utils.data import Dataset

from localizing.localizing_structs import TokenizedLocalization, LocalizationList
from localizing.multi_data_gathering import create_tokenized_localizations_from_scratch, DatasetName
from lmwrapper.openai_wrapper import OpenAiModelNames
from localizing.probe.probe_data_gather import (
    SingleVecToLabelDataset, 
    SingleVecToLabelDatasetMultiFold,
    GroupedVecLabelDataset,
    GroupedVecLabelDatasetMultiFold,
    EmbeddedTokenization,
)


#@dataclass(kw_only=True)
#class HeuristicTokenization(TokenizedLocalization):
#    """A tokenization with heuristic features instead of neural embeddings."""
#    base_tokens_features: torch.Tensor | None = None  # (num_tokens, feature_dim) float32
#    
#    def __post_init__(self):
#        super().__post_init__() if hasattr(super(), "__post_init__") else None
#        if self.base_tokens_features is not None and hasattr(self, "gt_base_token_keeps"):
#            assert self.base_tokens_features.shape[0] == len(self.gt_base_token_keeps), \
#                f"Number of feature vectors ({self.base_tokens_features.shape[0]}) " \
#                f"must match number of tokens in gt_base_token_keeps ({len(self.gt_base_token_keeps)})"
#

class HeuristicFeatureExtractor:
    """Extracts heuristic features from tokenized code."""
    
    def __init__(self, top_k_tokens: int = 100):
        self.top_k_tokens = top_k_tokens
        self.token_vocabulary: Set[str] = set()
        self.token_frequencies: Dict[str, int] = {}
        self.top_tokens: List[str] = []
        self.is_fitted = False
        
        # Log quantile scale for normalization
        self.log_quantile_scale: Optional[float] = None
        
        # Python keywords and common code tokens
        self.python_keywords = set(keyword.kwlist)
        self.common_operators = {'+', '-', '*', '/', '=', '==', '!=', '<', '>', 
                               '<=', '>=', '&&', '||', '!', '&', '|', '^'}
        self.brackets = {'(', ')', '[', ']', '{', '}'}
        self.punctuation = {',', ';', ':', '.', '?'}
        
    def fit(self, localizations: LocalizationList[TokenizedLocalization]) -> None:
        """Fit the feature extractor on a collection of localizations."""
        token_counter = Counter()
        distances_from_start = []
        distances_from_end = []
        
        # Count all tokens across all localizations and collect distance stats
        for loc in localizations.iter_passed_filtered():
            if loc.base_tokens:
                total_tokens = len(loc.base_tokens)
                for i, token in enumerate(loc.base_tokens):
                    token_counter[token] += 1
                    self.token_vocabulary.add(token)
                    
                    # Collect distance statistics for fitting log quantiles
                    distances_from_start.append(i + 1)
                    distances_from_end.append(total_tokens - i)
        
        self.token_frequencies = dict(token_counter)
        self.top_tokens = [token for token, _ in token_counter.most_common(self.top_k_tokens)]
        
        # Fit log quantile parameters based on actual data distribution
        if distances_from_start:
            # Use 95th percentile as the normalization factor for log quantiles
            self.log_quantile_scale = np.log(np.percentile(distances_from_start, 95) + 1)
        else:
            # Fallback values
            self.log_quantile_scale = 10.0
            
        self.is_fitted = True
        
    def extract_features(self, tokens: List[str]) -> torch.Tensor:
        """Extract heuristic features for a list of tokens."""
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted before extracting features")
            
        features = []
        total_tokens = len(tokens)
        
        # Precompute line boundaries and positions for efficiency
        line_starts, positions_in_line = self._compute_line_info(tokens)
        
        # Precompute nesting levels for all tokens (O(n) instead of O(n²))
        nesting_levels = self._compute_nesting_levels(tokens)
        
        for i, token in enumerate(tokens):
            token_features = []
            stripped_token = token.strip()  # For cleaner matching
            
            # === SYNTACTIC FEATURES (8 features) ===
            # 1. Is Python keyword (use stripped token)
            token_features.append(1.0 if stripped_token in self.python_keywords else 0.0)
            
            # 2. Is operator (use stripped token)
            token_features.append(1.0 if stripped_token in self.common_operators else 0.0)
            
            # 3. Is bracket/parenthesis (use stripped token)
            token_features.append(1.0 if stripped_token in self.brackets else 0.0)
            
            # 4. Is punctuation (use stripped token)
            token_features.append(1.0 if stripped_token in self.punctuation else 0.0)
            
            # 5. Is numeric literal (use stripped token)
            token_features.append(1.0 if self._is_numeric(stripped_token) else 0.0)
            
            # 6. Is string literal (use stripped token)
            token_features.append(1.0 if self._is_string_literal(stripped_token) else 0.0)
            
            # 7. Token length (normalized) - use original token
            token_features.append(min(len(token) / 20.0, 1.0))  # Cap at 20 chars
            
            # 8. Contains special characters (use original token)
            token_features.append(1.0 if self._has_special_chars(token) else 0.0)
            
            # === POSITIONAL FEATURES (7 features) ===
            # 9. Position in sequence (normalized)
            token_features.append(i / max(total_tokens - 1, 1))
            
            # 10. Distance from start (log-quantiled with fitted distribution)
            token_features.append(self._log_quantile_fitted(i + 1))
            
            # 11. Distance from end (log-quantiled with fitted distribution)
            token_features.append(self._log_quantile_fitted(total_tokens - i))
            
            # 12. Is at line start (first token on line)
            is_line_start = (i == 0 or line_starts[i] != line_starts[i-1])
            token_features.append(1.0 if is_line_start else 0.0)
            
            # 13. Position in line (normalized)
            current_line_pos = positions_in_line[i]
            line_length = self._get_line_length(tokens, i, line_starts)
            token_features.append(current_line_pos / max(line_length, 1))
            
            # 14. Indentation level (count leading spaces/tabs)
            token_features.append(self._get_indentation_level(token))
            
            # 15. Is after equals sign in current line
            token_features.append(1.0 if self._is_after_equals_in_line(tokens, i, line_starts) else 0.0)
            
            # === FREQUENCY FEATURES (top_k + 2 features) ===
            # 16 to 16+top_k-1. One-hot encoding for top-K most frequent tokens (use original token)
            top_k_features = [1.0 if token == top_token else 0.0 
                             for top_token in self.top_tokens]
            token_features.extend(top_k_features)
            
            # 16+top_k. Token frequency rank (normalized) - use original token
            rank = self._get_frequency_rank(token)
            token_features.append(rank)
            
            # 17+top_k. Log frequency - use original token
            freq = self.token_frequencies.get(token, 1)
            token_features.append(np.log(freq) / 10.0)  # Normalize
            
            # === CONTEXTUAL FEATURES (7 features) ===
            # 18+top_k. Length of surrounding context (avg of prev/next token lengths)
            context_length = self._get_context_length(tokens, i)
            token_features.append(context_length)
            
            # 19+top_k. Is followed by opening bracket (use stripped tokens)
            is_followed_by_bracket = (i < total_tokens - 1 and 
                                    tokens[i + 1].strip() in {'(', '[', '{'})
            token_features.append(1.0 if is_followed_by_bracket else 0.0)
            
            # 20+top_k. Is preceded by dot (method call) - use stripped tokens
            is_after_dot = (i > 0 and tokens[i - 1].strip() == '.')
            token_features.append(1.0 if is_after_dot else 0.0)
            
            # 21+top_k. Is assignment target (followed by =) - use stripped tokens
            is_assignment_target = (i < total_tokens - 1 and tokens[i + 1].strip() == '=')
            token_features.append(1.0 if is_assignment_target else 0.0)
            
            # 22+top_k. Is in function definition context - use stripped tokens
            is_in_function_def = self._is_in_function_def_context(tokens, i)
            token_features.append(1.0 if is_in_function_def else 0.0)
            
            # 23+top_k. Is variable-like (alphanumeric, not keyword) - use stripped token
            is_variable_like = (stripped_token.replace('_', '').isalnum() and 
                              stripped_token not in self.python_keywords and
                              not self._is_numeric(stripped_token))
            token_features.append(1.0 if is_variable_like else 0.0)
            
            # 24+top_k. Nesting level (precomputed efficiently)
            token_features.append(nesting_levels[i] / 10.0)  # Normalize
            
            features.append(token_features)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _is_numeric(self, token: str) -> bool:
        """Check if token is a numeric literal."""
        stripped = token.strip()
        if not stripped:
            return False
        try:
            float(stripped)
            return True
        except ValueError:
            return False
    
    def _is_string_literal(self, token: str) -> bool:
        """Check if token is a string literal."""
        return ((token.startswith('"') and token.endswith('"')) or
                (token.startswith("'") and token.endswith("'")))
    
    def _has_special_chars(self, token: str) -> bool:
        """Check if token contains special characters."""
        special_chars = set('!@#$%^&*()_+-=[]{}|;:,.<>?')
        return any(c in special_chars for c in token)
    
    def _log_quantile_fitted(self, value: int) -> float:
        """Convert value to log-quantiled feature using fitted 95th percentile."""
        if self.log_quantile_scale is None:
            # Fallback if not fitted
            return min(np.log(value + 1) / 10.0, 1.0)
        return min(np.log(value + 1) / self.log_quantile_scale, 1.0)
    
    def _compute_line_info(self, tokens: List[str]) -> Tuple[List[int], List[int]]:
        """Compute line boundaries and position-in-line for all tokens."""
        line_starts = []  # line_starts[i] = line number for token i
        positions_in_line = []  # positions_in_line[i] = position in line for token i
        
        current_line = 0
        position_in_current_line = 0
        
        for token in tokens:
            line_starts.append(current_line)
            positions_in_line.append(position_in_current_line)
            
            # Count newlines in this token
            newline_count = token.count('\n')
            if newline_count > 0:
                current_line += newline_count
                # If token ends with newline, next position is 0
                # Otherwise, it's the length after last newline
                lines_in_token = token.split('\n')
                position_in_current_line = len(lines_in_token[-1])
            else:
                position_in_current_line += len(token)
        
        return line_starts, positions_in_line
    
    def _compute_nesting_levels(self, tokens: List[str]) -> List[int]:
        """Efficiently compute nesting levels for all tokens in O(n)."""
        nesting_levels = []
        current_nesting = 0
        
        open_brackets = {'(', '[', '{'}
        close_brackets = {')', ']', '}'}
        
        for token in tokens:
            nesting_levels.append(current_nesting)
            
            if token in open_brackets:
                current_nesting += 1
            elif token in close_brackets:
                current_nesting = max(0, current_nesting - 1)
        
        return nesting_levels
    
    def _get_line_length(self, tokens: List[str], token_index: int, line_starts: List[int]) -> int:
        """Get the length of the line containing the given token."""
        target_line = line_starts[token_index]
        
        # Find all tokens on the same line
        line_tokens = []
        for i, line_num in enumerate(line_starts):
            if line_num == target_line:
                line_tokens.append(tokens[i])
            elif line_num > target_line:
                break
        
        # Calculate total length
        return sum(len(token) for token in line_tokens)
    
    def _is_after_equals_in_line(self, tokens: List[str], token_index: int, line_starts: List[int]) -> bool:
        """Check if token comes after an '=' sign in the same line."""
        target_line = line_starts[token_index]
        
        # Look backwards on the same line for '=' sign
        for i in range(token_index - 1, -1, -1):
            if line_starts[i] != target_line:
                break  # Different line
            if tokens[i] == '=':
                return True
        
        return False
    
    def _get_indentation_level(self, token: str) -> float:
        """Get normalized indentation level."""
        if not token.startswith((' ', '\t', '\n')):
            return 0.0
        
        # Count leading whitespace
        indent_count = 0
        for char in token:
            if char == ' ':
                indent_count += 1
            elif char == '\t':
                indent_count += 4  # Treat tab as 4 spaces
            else:
                break
        
        return min(indent_count / 20.0, 1.0)  # Normalize to [0, 1]
    
    def _get_frequency_rank(self, token: str) -> float:
        """Get normalized frequency rank (1.0 for most frequent, 0.0 for least)."""
        if token not in self.token_frequencies:
            return 0.0
        
        # Find rank of this token
        sorted_tokens = sorted(self.token_frequencies.items(), 
                             key=lambda x: x[1], reverse=True)
        
        for rank, (t, _) in enumerate(sorted_tokens):
            if t == token:
                return 1.0 - (rank / len(sorted_tokens))
        
        return 0.0
    
    def _get_context_length(self, tokens: List[str], index: int) -> float:
        """Get normalized average length of surrounding tokens."""
        lengths = []
        
        # Previous token
        if index > 0:
            lengths.append(len(tokens[index - 1]))
        
        # Next token
        if index < len(tokens) - 1:
            lengths.append(len(tokens[index + 1]))
        
        if not lengths:
            return 0.0
        
        avg_length = sum(lengths) / len(lengths)
        return min(avg_length / 20.0, 1.0)  # Normalize
    
    def _is_in_function_def_context(self, tokens: List[str], index: int) -> bool:
        """Check if token is in a function definition context."""
        # Look backwards for 'def' keyword within reasonable distance
        look_back = min(index, 10)
        for i in range(index - look_back, index):
            if i >= 0 and tokens[i].strip() == 'def':
                return True
        return False


def add_heuristic_features_to_localization_list(
    localizations: LocalizationList[TokenizedLocalization],
    feature_extractor: HeuristicFeatureExtractor
) -> LocalizationList[EmbeddedTokenization]:
    """Add heuristic features to a list of tokenized localizations."""
    localizations = localizations.copy_with_type_change(EmbeddedTokenization)
    for localization in tqdm(
        localizations.iter_passed_filtered(), 
        desc="Adding embeddings to localizations", 
        total=len(list(localizations.iter_passed_filtered())),
    ):
        features = feature_extractor.extract_features(localization.base_tokens)
        localization.base_tokens_embedding = features
    return localizations


def get_heuristic_localizations(
    dataset: DatasetName,
    gen_model_name: str,
    fix_reference: str,
    filter_to_original_fails: bool = True,
    max_problems: int = 1000,
    max_gen_tokens: int = 1000,
    tokenizer_key: str = "Qwen/Qwen2.5-Coder-0.5B",
    top_k_tokens: int = 100,
) -> LocalizationList[EmbeddedTokenization]:
    """Get localizations with heuristic features."""
    
    # First get tokenized localizations
    localizations = create_tokenized_localizations_from_scratch(
        dataset=dataset,
        gen_model_name=gen_model_name,
        filter_to_original_fails=filter_to_original_fails,
        max_problems=max_problems,
        max_gen_tokens=max_gen_tokens,
        fix_reference=fix_reference,
        tokenizer_key=tokenizer_key,
    )
    
    # Create and fit feature extractor
    feature_extractor = HeuristicFeatureExtractor(top_k_tokens=top_k_tokens)
    feature_extractor.fit(localizations)
    
    # Add heuristic features
    heuristic_localizations = add_heuristic_features_to_localization_list(
        localizations, feature_extractor
    )
    
    return heuristic_localizations


def get_or_serialize_heuristic_localizations(
    datasets: List[DatasetName] = None,
    gen_model_name: str = "mistralai/Mistral-7B-v0.1",
    fix_reference: str = OpenAiModelNames.o3_mini,
    max_problems: int = 1000,
    max_gen_tokens: int = 1000,
    filter_to_original_fails: bool = True,
    tokenizer_key: str = "Qwen/Qwen2.5-Coder-0.5B",
    top_k_tokens: int = 100,
) -> LocalizationList[EmbeddedTokenization]:
    """Get or load cached heuristic localizations."""
    
    if datasets is None:
        datasets = [DatasetName.humaneval_plus, DatasetName.mbpp_plus]
    
    # Create filename for caching
    dataset_names = "_".join([str(d) for d in datasets])
    fn = (f"heuristic_localizations_{dataset_names}_{gen_model_name.replace('/', '_')}_"
          f"{tokenizer_key.replace('/', '_')}_top{top_k_tokens}_max{max_problems}.pkl")
    
    fn = Path("cache") / fn
    fn.parent.mkdir(exist_ok=True)
    
    # Try to load cached version
    if fn.exists():
        print(f"Loading cached heuristic localizations from {fn}")
        with open(fn, 'rb') as f:
            return pickle.load(f)
    
    print(f"Creating new heuristic localizations for datasets: {datasets}")
    
    # Need to process each dataset and combine
    combined_localizations = None
    
    for dataset_name in datasets:
        print(f"Processing dataset: {dataset_name}")
        
        # Get localizations for this dataset
        current_localizations = get_heuristic_localizations(
            dataset=dataset_name,
            gen_model_name=gen_model_name,
            fix_reference=fix_reference,
            max_problems=max_problems,
            max_gen_tokens=max_gen_tokens,
            tokenizer_key=tokenizer_key,
            top_k_tokens=top_k_tokens,
            filter_to_original_fails=filter_to_original_fails,
        )
        
        # Initialize or extend the combined localizations
        if combined_localizations is None:
            combined_localizations = current_localizations
        else:
            combined_localizations.extend(current_localizations)
    
    # Cache the combined localizations
    print(f"Saving heuristic localizations to {fn}")
    with open(fn, 'wb') as f:
        pickle.dump(combined_localizations, f)
    
    return combined_localizations


def heuristic_localizations_to_grouped_vec_label_dataset(
    localizations: LocalizationList[EmbeddedTokenization],
    n_folds: int,
) -> GroupedVecLabelDatasetMultiFold:
    """Convert heuristic localizations to grouped dataset."""
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


def heuristic_localizations_to_single_vec_to_label_dataset(
    localizations: LocalizationList[EmbeddedTokenization],
    n_folds: int,
) -> SingleVecToLabelDatasetMultiFold:
    """Convert heuristic localizations to single vector dataset."""
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
            features = []
            labels = []
            for i, loc in enumerate(locs):
                if loc.base_tokens_embedding is not None:
                    x, y = loc.base_tokens_embedding, torch.tensor(loc.gt_base_token_keeps, dtype=torch.float32)
                    try:
                        assert x.shape[0] == y.shape[0], \
                            f"Number of features ({x.shape[0]}) must match number of labels ({y.shape[0]})"
                        features.append(x)
                        labels.append(y)
                    except AssertionError as e:
                        print(f"Error at localization {i}:")
                        print(f"base_tokens length: {len(loc.base_tokens) if hasattr(loc, 'base_tokens') else 'N/A'}")
                        print(f"gt_base_token_keeps length: {len(loc.gt_base_token_keeps)}")
                        print(f"base_tokens_features shape: {loc.base_tokens_features.shape}")
                        raise e
            
            if not features:  # Handle case where no valid features found
                return None
                
            return SingleVecToLabelDataset(
                embeddings=torch.cat(features, dim=0),  # Use 'embeddings' not 'features'
                labels=torch.cat(labels, dim=0),
            )
        
        # Create train and test datasets
        train_dataset = create_dataset(train_locs)
        test_dataset = create_dataset(test_locs)
        
        if train_dataset is None or test_dataset is None:
            continue  # Skip this fold if no valid data
        
        train_test_folds.append((train_dataset, test_dataset))
    
    return SingleVecToLabelDatasetMultiFold(
        folds_train_test=tuple(train_test_folds)
    )


def main():
    """Example usage and testing."""
    print("Testing Heuristic Feature Extraction...")
    
    # Get heuristic localizations
    localizations = get_or_serialize_heuristic_localizations(
        datasets=[DatasetName.humaneval_plus],
        gen_model_name=OpenAiModelNames.gpt_4o_mini,
        #max_problems=10,  # Small for testing
        top_k_tokens=20,
    )
    
    print(f"Got {len(localizations)} localizations")
    print(f"Passed filters: {localizations.len_passed_filtered()}")
    
    # Show sample features
    for i, loc in enumerate(localizations.iter_passed_filtered()):
        if i >= 3:  # Show first 3
            break
        print("-" * 80)
        print(f"Localization {i}:")
        print(f"Base Tokens ({len(loc.base_tokens)}): {loc.base_tokens[:10]}...")
        if loc.base_tokens_embedding is not None:
            print(f"Feature Shape: {loc.base_tokens_embedding.shape}")
            print(f"Sample features: {loc.base_tokens_embedding[0][:10]}")
        if hasattr(loc, 'gt_base_token_keeps') and loc.gt_base_token_keeps is not None:
            print(f"GT Keeps ({len(loc.gt_base_token_keeps)}): {loc.gt_base_token_keeps[:10]}...")
    
    # Test dataset creation
    print("\nTesting dataset creation...")
    single_dataset = heuristic_localizations_to_single_vec_to_label_dataset(localizations, 3)
    print(single_dataset)
    
    grouped_dataset = heuristic_localizations_to_grouped_vec_label_dataset(localizations, 3)
    print(grouped_dataset)


if __name__ == "__main__":
    main()