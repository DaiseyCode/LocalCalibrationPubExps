import difflib
from typing import Literal
from xml.etree.ElementInclude import include
from localizing.probe.probe_data_gather import default_save_path, deserialize_localizations
import numpy as np
import random
from pprint import pprint
from tqdm import tqdm
from localizing.cross_fold import localizations_to_folds
from localizing.localizing_structs import TokenizedLocalization
from localizing.pape_multis import loc_token_level_to_agg_pred
from localizing.predictions_repr import FoldResultsPreds, deserialize_fold_results_preds, serialize_fold_results_preds
from localizing.predictions_repr import make_metrics_from_fold_results_multilevel
from localizing.probe.agg_models.agg_config import FoldMode, ProbeConfig
from localizing.probe.probe_data_gather import get_or_serialize_tokenized_localizations, make_basic_serialize_key_args
from pape.configs import BASE_PAPER_CONFIG, BASE_PAPER_CONFIG_QWEN, DEFAULT_LINE_AGGREGATOR_INTRINSIC
from numba import njit


def get_or_serialize_logprobs_fold_results(
    config: ProbeConfig, 
    dev_mode: bool = False,
    line_aggregator: Literal["mean", "gmean", "min"] = DEFAULT_LINE_AGGREGATOR_INTRINSIC,
    problem_aggregator: Literal["mean", "gmean", "min"] = DEFAULT_LINE_AGGREGATOR_INTRINSIC,
) -> FoldResultsPreds:
    args = make_basic_serialize_key_args(config, dev_mode)
    args = (*args, line_aggregator, problem_aggregator)
    if config.fold_mode != FoldMode.cross_fold:
        args = (*args, config.fold_mode.value)
    args_str = "_".join(str(arg) for arg in args)
    fn = default_save_path / "intrinsic_results" / (args_str + ".pkl.lz4")
    results = deserialize_fold_results_preds(fn)
    if results is not None:
        return results

    locs = get_or_serialize_tokenized_localizations(
        config, dev_mode=dev_mode)
    locs = list(locs.iter_passed_filtered())
    folds = localizations_to_folds(locs, config)
    all_train_preds = []
    all_test_preds = []
    fold_names = []
    for fold_num, (train_locs, test_locs, fold_name) in enumerate(folds):
        print(f"Fold {fold_num}: {fold_name}")
        pred = lambda loc_list: [
            loc_token_level_to_agg_pred(
                loc,
                loc_to_base_probs(loc),
                line_aggregator=line_aggregator,
                problem_aggregator=problem_aggregator,
            )
            for loc in tqdm(loc_list, desc=f"Fold {fold_name} preding")
        ]
        all_train_preds.append(pred(train_locs))
        all_test_preds.append(pred(test_locs))
        fold_names.append(fold_name)
    results = FoldResultsPreds(
        train_preds_each_fold=all_train_preds,
        test_preds_each_fold=all_test_preds,
        config=config,
        fold_names=fold_names,
    )
    serialize_fold_results_preds(results, fn)
    return results


def loc_to_base_probs(
    loc: TokenizedLocalization
) -> np.ndarray:
    logprobs = loc.base_solve.lm_prediction[0].completion_logprobs
    probs = np.exp(logprobs)
    mapping = align_tokens(loc.base_tokens, loc.base_solve.lm_prediction[0].completion_tokens)
    out = np.zeros(len(loc.base_tokens))
    agg_avg_prob = np.mean(probs)
    for i, mapping in enumerate(mapping):
        if len(mapping) == 0:
            out[i] = agg_avg_prob
        elif len(mapping) == 1:
            out[i] = probs[mapping[0]]
        else:
            out[i] = np.min([probs[j] for j in mapping])
    return out


def main():
    for line_aggregator in ["mean", "gmean", "min"]:
        print(f"Line aggregator: {line_aggregator}")
        pprint(
            make_metrics_from_fold_results_multilevel(
                get_or_serialize_logprobs_fold_results(
                    BASE_PAPER_CONFIG_QWEN,
                    line_aggregator=line_aggregator,
                    problem_aggregator=line_aggregator,
                ),
                include_problem_level=True,
            )
        )
    exit()
    locs = get_or_serialize_tokenized_localizations(
        BASE_PAPER_CONFIG)
    locs = list(locs.iter_passed_filtered())
    first_loc: TokenizedLocalization = locs[0]
    #num_docstr = 0
    #for loc in locs.iter_passed_filtered():
    #    if '"""' in loc.get_base_text():
    #        num_docstr += 1
    #print(f"Number of docstrings: {num_docstr}")

    print("Completion text")
    print(first_loc.base_solve.lm_prediction[0].completion_text)
    print("Completion tokens")
    print(first_loc.base_solve.lm_prediction[0].completion_tokens)
    #print("Completion logprobs")
    #print(first_loc.base_solve.lm_prediction[0].completion_logprobs)
    print("Base Text")
    print(first_loc.get_base_text())
    print("Base tokens")
    print(first_loc.base_tokens)

    print("Mapping")
    print(align_tokens(first_loc.base_tokens, first_loc.base_solve.lm_prediction[0].completion_tokens))

    success = 0
    failed = 0
    excessive_empties = 0
    has_excessively_long = 0
    sample = locs
    for loc in tqdm(sample):
        try:
            mapping = align_tokens(loc.base_tokens, loc.base_solve.lm_prediction[0].completion_tokens)
            if sum(1 for m in mapping if len(m) == 0) > 3:
                excessive_empties += 1
            if sum(1 for m in mapping if len(m) > 3):
                has_excessively_long += 1
            print(mapping)
            success += 1
        except ValueError:
            raise
            failed += 1
    print(f"Success: {success}, Failed: {failed}")
    print(f"Excessive empties: {excessive_empties}")
    print(f"Has excessively long: {has_excessively_long}")


#@njit
def map_chunks(A: list[str], B: list[str]) -> list[list[int]]:
    """Return, for every chunk in A, the B-indices that overlap it."""
    if "".join(A) != "".join(B):
        print(f"A: {A!r}")
        print(f"B: {B!r}")
        raise ValueError("A and B must represent the same string for this algorithm to work")

    out = [[] for _ in A]

    i = j = 0       # current chunk indices in A and B
    ai = bj = 0     # offsets inside current chunks

    while i < len(A) and j < len(B):
        # amount we can consume before hitting the end of the current chunk
        step = min(len(A[i]) - ai, len(B[j]) - bj)

        # record that A[i] overlaps B[j]
        if not out[i] or out[i][-1] != j:
            out[i].append(j)

        # advance
        ai += step
        bj += step
        if ai == len(A[i]):
            i += 1
            ai = 0
        if bj == len(B[j]):
            j += 1
            bj = 0
    return out


#@njit
def lcs(a, b):
    """Find the longest common substring (not subsequence) between two strings."""
    m, n = len(a), len(b)
    # dp[i][j] stores the length of common substring ending at a[i-1] and b[j-1]
    dp = [[0] * (n+1) for _ in range(m+1)]
    max_length = 0
    ending_pos_a = 0
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    ending_pos_a = i
            else:
                dp[i][j] = 0
    
    # Extract the longest common substring
    if max_length == 0:
        return ""
    
    start_pos = ending_pos_a - max_length
    return a[start_pos:ending_pos_a]



def find_token_span_for_str(tokens: list[str], substr: str, full_str: str) -> tuple[int, int]:
    """
    Find the token span (start_idx, end_idx) that corresponds to substr in full_str.
    Returns indices such that tokens[start_idx:end_idx] covers the substr.
    
    Args:
        tokens: List of tokens
        substr: The substring to find
        full_str: The full string that tokens represent when joined
    
    Returns:
        (start_token_idx, end_token_idx) - end is exclusive
    """
    import bisect
    
    # Find where substr appears in full_str
    substr_start = full_str.find(substr)
    if substr_start == -1:
        print(f"find_token_span_for_str error")
        print(f"substr: {substr!r}")
        print(f"full_str: {full_str!r}")
        print(f"tokens: {tokens!r}")
        raise ValueError(f"Substring '{substr}' not found in full string")
    
    substr_end = substr_start + len(substr)
    
    # Build cumulative character positions
    # char_positions[i] = character position where token i starts
    char_positions = [0]
    for token in tokens:
        char_positions.append(char_positions[-1] + len(token))
    
    # Use binary search to find token boundaries
    # Find the rightmost token that starts at or before substr_start
    start_token_idx = bisect.bisect_right(char_positions, substr_start) - 1
    start_token_idx = max(0, start_token_idx)
    
    # Find the leftmost token that starts at or after substr_end
    end_token_idx = bisect.bisect_left(char_positions, substr_end)
    end_token_idx = min(len(tokens), end_token_idx)
    
    # If substr_end falls exactly on a token boundary, we don't need to include that token
    # But if it falls in the middle of a token, we need to include that token
    if end_token_idx < len(tokens) and char_positions[end_token_idx] < substr_end:
        end_token_idx += 1
    
    return start_token_idx, end_token_idx


def align_tokens(
    base_tokens: list[str],
    tokens_to_map_to: list[str],
) -> list[list[int]]:
    """
    Estimates a mapping for each base_token to to
    tokens in tokens_to_map_to. If there is no matches
    then the mapping might be None.
    """
    def clear_ws(tokens: list[str]) -> list[str]:
        out = []
        for token in tokens:
            striped = token.strip()
            if striped:
                out.append(striped)
            else:
                out.append("")
        return out

    base_tokens = clear_ws(base_tokens)
    tokens_to_map_to = clear_ws(tokens_to_map_to)

    base_combo = "".join(base_tokens)
    map_to_combo = "".join(tokens_to_map_to)
    #print(f"base_combo: {base_combo!r}")
    #print(f"map_to_combo: {map_to_combo!r}")

    if base_combo == map_to_combo:
        return map_chunks(base_tokens, tokens_to_map_to)

    # Find the longest common subsequence between the two strings
    lcs_str = lcs(base_combo, map_to_combo)
    #print(f"LCS: {repr(lcs_str)} (length: {len(lcs_str)})")
    
    if not lcs_str:
        # No common subsequence, return empty mappings
        return [[] for _ in base_tokens]
    
    # Find token spans for the LCS in both tokenizations
    base_start, base_end = find_token_span_for_str(base_tokens, lcs_str, base_combo)
    map_to_start, map_to_end = find_token_span_for_str(tokens_to_map_to, lcs_str, map_to_combo)
    
    # Create the mapping for the overlapping region
    base_overlap_tokens = base_tokens[base_start:base_end]
    map_to_overlap_tokens = tokens_to_map_to[map_to_start:map_to_end]

    # Clean up start and end for nomralizing
    #base_overlap_tokens[0] = "START"
    #base_overlap_tokens[-1] = "END"
    #map_to_overlap_tokens[0] = "START"
    #map_to_overlap_tokens[-1] = "END"
    
    # Use map_chunks on the overlapping region
    try:
        overlap_mapping = map_chunks(base_overlap_tokens, map_to_overlap_tokens)
    except ValueError:
        print(f"map_chunks error")
        print(f"base_tokens: {base_tokens!r}")
        print(f"tokens_to_map_to: {tokens_to_map_to!r}")
        print(f"base_combo: {base_combo!r}")
        print(f"map_to_combo: {map_to_combo!r}")
        print(f"lcs_str: {lcs_str!r}")
        raise
    
    # Build the full mapping
    result = [[] for _ in base_tokens]
    for i, mapping in enumerate(overlap_mapping):
        # Adjust indices to account for the offset
        result[base_start + i] = [map_to_start + j for j in mapping]
    
    return result
        


def test_align_tokens_matching():
    base_tokens = ["def", "f", "()", ":", "pass"]
    tokens_to_map_to = ["def", "f", "()", ":", "pass"]
    mapping = align_tokens(base_tokens, tokens_to_map_to)
    assert mapping == [[0], [1], [2], [3], [4]]


def test_align_tokens_matching_2():
    base_tokens = ["1 2", " 3 4", " 5 6"]
    tokens_to_map_to = ["1 2 3 4", " 5 6"]
    mapping = align_tokens(base_tokens, tokens_to_map_to)
    assert mapping == [[0], [0], [1]]


def test_align_tokens_matching_3():
    base_tokens = ["1 2 3 4", " 5 6"]
    tokens_to_map_to = ["1 2", " 3 4", " 5 6"]
    mapping = align_tokens(base_tokens, tokens_to_map_to)
    assert mapping == [[0, 1], [2]]


def test_align_tokens_matching_4():
    base_tokens = ["1 2", " 3 4", " 5 6"]
    tokens_to_map_to = ["1 2 3", " 4 5", " 6"]
    mapping = align_tokens(base_tokens, tokens_to_map_to)
    assert mapping == [[0], [0, 1], [1, 2]]


def test_align_tokens_matching_extra():
    base_tokens = ["12", "34"]
    tokens_to_map_to = ["0", "12", "34", "56"]
    mapping = align_tokens(base_tokens, tokens_to_map_to)
    assert mapping == [[1], [2]]


if __name__ == "__main__":
    main()
