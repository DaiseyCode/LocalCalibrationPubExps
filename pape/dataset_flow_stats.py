from typing import TypeVar
import dataclasses
import re
from synthegrator.synthdatasets import DatasetName
from localizing.filter_helpers import debug_str_filterables
from localizing.localizing_structs import BaseLocalization, LocalizationList, TokenizedLocalization
from localizing.probe.probe_data_gather import get_or_serialize_localizations_embedded, get_or_serialize_tokenized_localizations
from localizing.probe.agg_models.agg_config import ProbeConfig
from collections import Counter, defaultdict
from lmwrapper.openai_wrapper import OpenAiModelNames
import numpy as np
from itertools import combinations

from pape.configs import BASE_PAPER_CONFIG, BASE_PAPER_CONFIG_QWEN


T_l = TypeVar('T_l', bound=TokenizedLocalization)


def format_counter_compact(counter, max_items=10):
    """Format a Counter compactly, showing most common items"""
    items = list(counter.items())
    if len(items) <= max_items:
        return dict(items)
    
    # Show most common items
    most_common = counter.most_common(max_items)
    remaining = len(items) - max_items
    result = dict(most_common)
    if remaining > 0:
        result[f"... ({remaining} more)"] = "..."
    return result


def format_confusion_matrix(confusion_matrix, max_items=8):
    """Format confusion matrix compactly for display"""
    if not confusion_matrix:
        return {}
    
    # Get all adders and sort them for consistent display
    all_adders = sorted(set(key[0] for key in confusion_matrix.keys()) | 
                       set(key[1] for key in confusion_matrix.keys()))
    
    if len(all_adders) > max_items:
        # Show most active adders (those with highest total overlap)
        adder_totals = defaultdict(int)
        for (adder1, adder2), count in confusion_matrix.items():
            adder_totals[adder1] += count
            adder_totals[adder2] += count
        
        top_adders = sorted(adder_totals.keys(), key=lambda x: adder_totals[x], reverse=True)[:max_items]
        filtered_matrix = {k: v for k, v in confusion_matrix.items() 
                          if k[0] in top_adders and k[1] in top_adders}
        return {"confusion_matrix": filtered_matrix, "note": f"Showing top {max_items} adders by total overlap"}
    
    return {"confusion_matrix": confusion_matrix}


def format_exclusivity_analysis(exclusivity_data, max_items=8):
    """Format exclusivity analysis compactly for display"""
    if not exclusivity_data:
        return {}
    
    # Sort by unique count (descending) for better readability
    sorted_items = sorted(exclusivity_data.items(), key=lambda x: x[1]['unique_count'], reverse=True)
    
    if len(sorted_items) > max_items:
        return {
            "exclusivity_analysis": dict(sorted_items[:max_items]),
            "note": f"Showing top {max_items} adders by unique solutions"
        }
    
    return {"exclusivity_analysis": dict(sorted_items)}


def compute_per_localization_removal_impact(loc, row_stats):
    """
    Compute the impact of removing each adder for a single localization.
    Updates row_stats['impacts_if_removed'] with per-localization data.
    
    Args:
        loc: Single localization to analyze
        row_stats: Dictionary to update with impact data
    """
    # Get all adder distances for this localization
    adder_distances = {}
    for adder, distance in loc.get_tag_values("pick_best_branched_distance"):
        adder_distances[adder] = distance
    
    if not adder_distances:
        return  # Skip if no distance data
    
    # Find the current minimum distance across all adders
    valid_distances = {k: v for k, v in adder_distances.items() if v is not None}
    if not valid_distances:
        return  # Skip if no valid distances
    
    current_min_distance = min(valid_distances.values())
    problem_id = loc.base_solve.problem.problem_id
    
    # For each adder, compute what happens if we remove it
    for adder_to_remove in adder_distances:
        # Initialize if not exists
        if adder_to_remove not in row_stats['impacts_if_removed']:
            row_stats['impacts_if_removed'][adder_to_remove] = {
                'newly_unsolvable': [],
                'distance_changes': []
            }
        
        remaining_distances = {k: v for k, v in adder_distances.items() if k != adder_to_remove}
        valid_remaining = {k: v for k, v in remaining_distances.items() if v is not None}
        
        if not valid_remaining:
            # This localization becomes unsolvable
            row_stats['impacts_if_removed'][adder_to_remove]['newly_unsolvable'].append(problem_id)
        else:
            # Find what the new minimum would be
            new_min_distance = min(valid_remaining.values())
            distance_increase = new_min_distance - current_min_distance
            row_stats['impacts_if_removed'][adder_to_remove]['distance_changes'].append(distance_increase)


def aggregate_per_localization_impacts(impacts_if_removed):
    """
    Convert per-localization impact data into aggregate statistics.
    
    Args:
        impacts_if_removed: Dict with structure {adder: {'newly_unsolvable': [problem_ids], 'distance_changes': [floats]}}
        
    Returns:
        Dict with aggregate statistics
    """
    aggregated_impact = {}
    
    for adder, impact_data in impacts_if_removed.items():
        distance_changes = impact_data['distance_changes']
        newly_unsolvable_count = len(impact_data['newly_unsolvable'])
        
        # Compute detailed stats for distance changes
        distance_stats = compute_list_stats(distance_changes, filter_none=False)
        
        aggregated_impact[adder] = {
            'newly_unsolvable': newly_unsolvable_count,
            'problems_before': len(distance_changes) + newly_unsolvable_count,
            'problems_after': len(distance_changes),
            'distance_increases': distance_changes,
            'distance_stats': distance_stats,
            'mean_distance_increase': distance_stats['mean'] if distance_stats['mean'] is not None else 0.0,
            'median_distance_increase': distance_stats['median'] if distance_stats['median'] is not None else 0.0,
            'max_distance_increase': distance_stats['max'] if distance_stats['max'] is not None else 0.0,
            'problems_with_increase': len([d for d in distance_changes if d > 0]),
            'problems_with_no_change': len([d for d in distance_changes if d == 0]),
        }
    
    return aggregated_impact


def format_impact_analysis(impact_data, max_items=8):
    """Format impact analysis compactly for display"""
    if not impact_data:
        return {}
    
    # Sort by impact (newly_unsolvable + mean_distance_increase)
    def impact_score(item):
        metrics = item[1]
        return metrics['newly_unsolvable'] * 1000 + metrics['mean_distance_increase']  # Weight unsolvable heavily
    
    sorted_items = sorted(impact_data.items(), key=impact_score, reverse=True)
    
    if len(sorted_items) > max_items:
        return {
            "impact_analysis": dict(sorted_items[:max_items]),
            "note": f"Showing top {max_items} adders by removal impact"
        }
    
    return {"impact_analysis": dict(sorted_items)}


def compute_list_stats(values, filter_none=True):
    """
    Compute descriptive statistics from a list of values.
    
    Args:
        values: List of numeric values (may contain None)
        filter_none: If True, filter out None values before computing stats
        
    Returns:
        Dict with keys: count, mean, median, std, min, max (stats are None if no valid values)
    """
    if not values:
        return {'count': 0, 'mean': None, 'median': None, 'std': None, 'min': None, 'max': None}
    
    if filter_none:
        valid_values = [v for v in values if v is not None]
    else:
        valid_values = values
    
    if not valid_values:
        return {'count': 0, 'mean': None, 'median': None, 'std': None, 'min': None, 'max': None}
    
    valid_array = np.array(valid_values)
    return {
        'count': len(valid_array),
        'mean': float(np.mean(valid_array)),
        'median': float(np.median(valid_array)),
        'std': float(np.std(valid_array)),
        'min': float(np.min(valid_array)),
        'max': float(np.max(valid_array)),
        'sum': float(np.sum(valid_array)),
    }


def format_stats_compact(stats_dict, precision=1):
    """
    Format a statistics dictionary compactly using mathematical notation.
    
    Args:
        stats_dict: Dict with keys like count, mean, median, std, min, max
        precision: Number of decimal places for floats
        
    Returns:
        Formatted string
    """
    if stats_dict is None:
        return "[]"
    
    count = stats_dict['count']
    
    # If count is 0 or all stats are None, just show count
    if count == 0 or stats_dict['mean'] is None:
        return f"n={count}"
    
    mean = stats_dict['mean']
    median = stats_dict['median']
    std = stats_dict['std']
    min_val = stats_dict['min']
    max_val = stats_dict['max']
    
    # Format floats vs integers appropriately
    def fmt(val):
        if val == int(val):
            return str(int(val))
        else:
            return f"{val:.{precision}f}"
    
    return f"n={count}, x̄={fmt(mean)}, x̃={fmt(median)}, σ={fmt(std)}, range=[{fmt(min_val)}, {fmt(max_val)}]"


def print_stats_compact(dataset_name: str, stats: dict):
    """Print stats dictionary with compact formatting for a single dataset"""
    print(f"\n{dataset_name}:")
    for key, value in stats.items():
        if isinstance(value, Counter):
            formatted_value = format_counter_compact(value)
            print(f"  {key}: {formatted_value}")
        elif key == 'distances_all_stats':
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict):
                    formatted_stats = format_stats_compact(sub_value, precision=3)
                    print(f"    {sub_key}: {formatted_stats}")
                else:
                    print(f"    {sub_key}: {sub_value}")
        elif key == 'pick_best_confusion_matrix':
            formatted_matrix = format_confusion_matrix(value)
            print(f"  {key}:")
            if 'note' in formatted_matrix:
                print(f"    {formatted_matrix['note']}")
                matrix_data = formatted_matrix['confusion_matrix']
            else:
                matrix_data = formatted_matrix.get('confusion_matrix', value)
            
            for (adder1, adder2), count in sorted(matrix_data.items()):
                print(f"    ({adder1}, {adder2}): {count}")
        elif key == 'adder_exclusivity_analysis':
            formatted_exclusivity = format_exclusivity_analysis(value)
            print(f"  {key}:")
            if 'note' in formatted_exclusivity:
                print(f"    {formatted_exclusivity['note']}")
                analysis_data = formatted_exclusivity['exclusivity_analysis']
            else:
                analysis_data = formatted_exclusivity.get('exclusivity_analysis', value)
            
            for adder, metrics in analysis_data.items():
                unique_count = metrics['unique_count']
                total_count = metrics['total_count']
                unique_pct = (unique_count / total_count * 100) if total_count > 0 else 0
                print(f"    {adder}: {unique_count}/{total_count} unique ({unique_pct:.1f}%)")
                if metrics['unique_examples']:
                    examples = abbreviated(metrics['unique_examples'], list_max_head=3, list_max_tail=1)
                    examples_str = ', '.join(f"prob_{ex}" for ex in examples if ex != '...')
                    if '...' in examples:
                        remaining = len(metrics['unique_examples']) - 4  # 3 head + 1 tail
                        examples_str += f" (+ {remaining} more)"
                    print(f"      Examples: {examples_str}")
        elif key in ['passing_base_char_len', 'passing_num_passing_base_comments', 
                     'passing_fix_char_len', 'passing_fix_num_comments',
                     'passing_base_line_len', 'passing_fix_line_len']:
            # Skip printing raw lists - we'll print the stats versions instead
            continue
        elif key.endswith('_stats') and key.startswith('passing_'):
            formatted_stats = format_stats_compact(value)
            # Clean up the key name for display
            display_key = key.replace('_stats', '')
            print(f"  {display_key}: {formatted_stats}")
        elif key == 'adder_removal_impact':
            formatted_impact = format_impact_analysis(value)
            print(f"  {key}:")
            if 'note' in formatted_impact:
                print(f"    {formatted_impact['note']}")
                analysis_data = formatted_impact['impact_analysis']
            else:
                analysis_data = formatted_impact.get('impact_analysis', value)
            
            for adder, metrics in analysis_data.items():
                unsolvable = metrics['newly_unsolvable']
                mean_increase = metrics['mean_distance_increase']
                median_increase = metrics['median_distance_increase']
                problems_worse = metrics['problems_with_increase']
                problems_same = metrics['problems_with_no_change']
                
                print(f"    {adder}:")
                print(f"      Newly unsolvable: {unsolvable} problems")
                print(f"      Distance impact: mean +{mean_increase:.1f}, median +{median_increase:.1f}")
                print(f"      Problems affected: {problems_worse} worse, {problems_same} unchanged")
                
                # Print compact stats if available, otherwise abbreviated list
                if 'distance_stats' in metrics and metrics['distance_stats']['count'] > 0:
                    compact_stats = format_stats_compact(metrics['distance_stats'])
                    print(f"      Distance stats: {compact_stats}")
                elif 'distance_increases' in metrics and metrics['distance_increases']:
                    print(f"      Distance increases: {abbreviated(metrics['distance_increases'])}")
        else:
            # Use abbreviated for any other values that might contain long lists
            print(f"  {key}: {abbreviated(value)}")


def stratify_by_dataset(
    locs: LocalizationList[T_l]
) -> dict[str, LocalizationList[T_l]]:
    vals = defaultdict(list)
    for loc in locs.iter_all():
        vals[loc.dataset_name].append(loc)
    for k, v in vals.items():
        vals[k] = LocalizationList(v)
    return dict(vals)


def compute_dataset_stats(locs: LocalizationList[T_l]) -> dict:
    """
    Compute comprehensive statistics for a LocalizationList.
    
    Args:
        locs: LocalizationList to analyze
        
    Returns:
        Dictionary containing all computed statistics
    """
    row_stats = {}
    row_stats['total_problems'] = len(locs)
    row_stats['num_patched'] = sum(
        1 for loc in locs.iter_passed_filtered()
        if loc.did_pass_filter("fix_main_metric_is_success")
    )
    
    # Initialize GT vs non-GT tracking
    row_stats['num_patched_gt'] = 0
    row_stats['num_patched_non_gt'] = 0
    
    solves, fails = 0, 0
    row_stats['chosen_best_adder'] = Counter()
    row_stats['pick_best_branched_passed'] = Counter()
    row_stats['pick_best_branched_failed'] = Counter()
    row_stats['distances_all'] = defaultdict(list)
    row_stats['pass_all_filters'] = 0
    row_stats['pass_all_filters_og_failed'] = 0
    row_stats['pass_not_gt'] = 0
    row_stats['passing_base_char_len'] = []
    row_stats['passing_num_passing_base_comments'] = []
    row_stats['passing_fix_char_len'] = []
    row_stats['passing_fix_num_comments'] = []
    row_stats['passing_base_line_len'] = []
    row_stats['passing_fix_line_len'] = []
    row_stats['passing_patched_not_keep_token_rate'] = []
    row_stats['passing_patched_not_keep_token_count'] = []
    row_stats['combined_total_tokens'] = []
    
    # Initialize separate lists for GT vs non-GT token statistics
    row_stats['passing_patched_not_keep_token_rate_gt'] = []
    row_stats['passing_patched_not_keep_token_count_gt'] = []
    row_stats['passing_patched_not_keep_token_rate_non_gt'] = []
    row_stats['passing_patched_not_keep_token_count_non_gt'] = []
    
    # Initialize confusion matrix for pick_best_branched_passed
    row_stats['pick_best_confusion_matrix'] = defaultdict(int)
    
    # Initialize data structures for exclusivity analysis
    adder_to_problems = defaultdict(set)  # adder_name -> set of problem_ids
    problem_to_adders = defaultdict(set)  # problem_id -> set of adder_names
    
    # Initialize per-localization impact tracking
    row_stats['impacts_if_removed'] = {}
    
    for loc in locs.iter_all():
        if loc.base_eval.main_metric is not None and loc.base_eval.main_metric.is_success:
            solves += 1
        elif loc.base_eval.main_metric is not None:
            fails += 1
        if loc.check_passed_all_filters():
            row_stats['pass_all_filters'] += 1
            row_stats['passing_base_char_len'].append(len(loc.get_base_text()))
            row_stats['passing_num_passing_base_comments'].append(count_comments_in_code(loc.get_base_text()))
            row_stats['passing_fix_char_len'].append(len(loc.get_gt_fix_text()))
            row_stats['passing_fix_num_comments'].append(count_comments_in_code(loc.get_gt_fix_text()))
            row_stats['passing_base_line_len'].append(count_non_comment_lines(loc.get_base_text()))
            row_stats['passing_fix_line_len'].append(count_non_comment_lines(loc.get_gt_fix_text()))
            if loc.is_base_success:
                row_stats['pass_all_filters_og_failed'] += 1
            # Track chosen adders and separate GT vs non-GT statistics
            chosen_adders = list(loc.get_tag_values("chosen_best_adder"))
            is_gt_solution = any("ground_truth" in adder for adder in chosen_adders)
            
            for best_adder in chosen_adders:
                row_stats['chosen_best_adder'][best_adder] += 1
                if "ground_truth" not in best_adder:
                    row_stats['pass_not_gt'] += 1
            
            # Count patched solutions by type
            if not loc.is_base_success:  # Only count as patched if original failed
                if is_gt_solution:
                    row_stats['num_patched_gt'] += 1
                else:
                    row_stats['num_patched_non_gt'] += 1
            
            # Track token statistics separately for GT vs non-GT
            not_keeps = (~loc.gt_base_token_keeps).sum()
            if not loc.is_base_success:
                row_stats['passing_patched_not_keep_token_count'].append(not_keeps)
                row_stats['passing_patched_not_keep_token_rate'].append(not_keeps / len(loc.gt_base_token_keeps))
                
                # Separate tracking for GT vs non-GT
                if is_gt_solution:
                    row_stats['passing_patched_not_keep_token_count_gt'].append(not_keeps)
                    row_stats['passing_patched_not_keep_token_rate_gt'].append(not_keeps / len(loc.gt_base_token_keeps))
                else:
                    row_stats['passing_patched_not_keep_token_count_non_gt'].append(not_keeps)
                    row_stats['passing_patched_not_keep_token_rate_non_gt'].append(not_keeps / len(loc.gt_base_token_keeps))
            
            row_stats['combined_total_tokens'].append(len(loc.gt_base_token_keeps))

        
        # Collect passed adders for this localization
        passed_adders = loc.get_tag_values("pick_best_branched_passed")
        problem_id = loc.base_solve.problem.problem_id
        
        for passed_adder in passed_adders:
            row_stats['pick_best_branched_passed'][passed_adder] += 1
            # Track which problems each adder solves
            adder_to_problems[passed_adder].add(problem_id)
            problem_to_adders[problem_id].add(passed_adder)
        
        # Build confusion matrix for passed adders
        for adder1, adder2 in combinations(passed_adders, 2):
            # Sort to ensure consistent ordering (adder1, adder2) vs (adder2, adder1)
            pair = tuple(sorted([adder1, adder2]))
            row_stats['pick_best_confusion_matrix'][pair] += 1
        
        for failed_adder in loc.get_tag_values("pick_best_branched_failed"):
            row_stats['pick_best_branched_failed'][failed_adder] += 1
        for fix_adder, distance in loc.get_tag_values("pick_best_branched_distance"):
            row_stats['distances_all'][fix_adder].append(distance)
        
        # Compute per-localization removal impact
        compute_per_localization_removal_impact(loc, row_stats)
    
    # Compute exclusivity analysis
    row_stats['adder_exclusivity_analysis'] = {}
    for adder, problems_solved in adder_to_problems.items():
        # Find problems that ONLY this adder solved
        unique_problems = set()
        for problem_id in problems_solved:
            if len(problem_to_adders[problem_id]) == 1:  # Only this adder solved it
                unique_problems.add(problem_id)
        
        row_stats['adder_exclusivity_analysis'][adder] = {
            'total_count': len(problems_solved),
            'unique_count': len(unique_problems),
            'unique_examples': sorted(list(unique_problems))
        }
    
    # Convert per-localization impacts to aggregate statistics
    row_stats['adder_removal_impact'] = aggregate_per_localization_impacts(row_stats['impacts_if_removed'])
    
    # Convert defaultdict to regular dict for cleaner output
    row_stats['pick_best_confusion_matrix'] = dict(row_stats['pick_best_confusion_matrix'])
    
    row_stats['og_solves'] = solves
    row_stats['og_not_solves'] = fails
    row_stats['pass@1'] = solves / (solves + fails) if (solves + fails) > 0 else 0.0
    
    # Calculate aggregate statistics for distances_all
    row_stats['distances_all_stats'] = {}
    for fix_adder, distances in row_stats['distances_all'].items():
        stats = compute_list_stats(distances, filter_none=True)
        row_stats['distances_all_stats'][fix_adder] = stats
    
    # Calculate aggregate statistics for character lengths and comments
    row_stats['passing_base_char_len_stats'] = compute_list_stats(row_stats['passing_base_char_len'], filter_none=False)
    row_stats['passing_num_passing_base_comments_stats'] = compute_list_stats(row_stats['passing_num_passing_base_comments'], filter_none=False)
    row_stats['passing_fix_char_len_stats'] = compute_list_stats(row_stats['passing_fix_char_len'], filter_none=False)
    row_stats['passing_fix_num_comments_stats'] = compute_list_stats(row_stats['passing_fix_num_comments'], filter_none=False)
    row_stats['passing_patched_not_keep_token_rate_stats'] = compute_list_stats(row_stats['passing_patched_not_keep_token_rate'], filter_none=False)
    row_stats['passing_patched_not_keep_token_count_stats'] = compute_list_stats(row_stats['passing_patched_not_keep_token_count'], filter_none=False)
    row_stats['combined_total_tokens_stats'] = compute_list_stats(row_stats['combined_total_tokens'], filter_none=False)
    
    # Calculate aggregate statistics for GT vs non-GT token statistics
    row_stats['passing_patched_not_keep_token_rate_gt_stats'] = compute_list_stats(row_stats['passing_patched_not_keep_token_rate_gt'], filter_none=False)
    row_stats['passing_patched_not_keep_token_count_gt_stats'] = compute_list_stats(row_stats['passing_patched_not_keep_token_count_gt'], filter_none=False)
    row_stats['passing_patched_not_keep_token_rate_non_gt_stats'] = compute_list_stats(row_stats['passing_patched_not_keep_token_rate_non_gt'], filter_none=False)
    row_stats['passing_patched_not_keep_token_count_non_gt_stats'] = compute_list_stats(row_stats['passing_patched_not_keep_token_count_non_gt'], filter_none=False)
    # Calculate aggregate statistics for line lengths
    row_stats['passing_base_line_len_stats'] = compute_list_stats(row_stats['passing_base_line_len'], filter_none=False)
    row_stats['passing_fix_line_len_stats'] = compute_list_stats(row_stats['passing_fix_line_len'], filter_none=False)
    
    return row_stats


def count_comments_in_code(code: str) -> int:
    return len(re.findall(r'#.*', code))


def count_non_comment_lines(code: str) -> int:
    """
    Count lines of code excluding comment-only lines and empty lines.
    
    Args:
        code: Source code string
        
    Returns:
        Number of non-comment, non-empty lines
    """
    lines = code.splitlines()
    non_comment_lines = 0
    
    for line in lines:
        # Strip whitespace to check if line is empty or comment-only
        stripped_line = line.strip()
        
        # Skip empty lines
        if not stripped_line:
            continue
            
        # Skip comment-only lines (lines that start with #)
        if stripped_line.startswith('#'):
            continue
            
        # Count this as a non-comment line
        non_comment_lines += 1
    
    return non_comment_lines


def do_with_config(config: ProbeConfig):
    #locs = get_or_serialize_localizations_embedded(config)
    locs = get_or_serialize_tokenized_localizations(config)
    locs_by_dataset = stratify_by_dataset(locs)
    items = {}
    
    # Compute and print stats for each individual dataset
    for k, v in locs_by_dataset.items():
        print(debug_str_filterables(v.iter_all()))
        dataset_stats = compute_dataset_stats(v)
        items[str(k)] = dataset_stats
        print_stats_compact(str(k), dataset_stats)
    
    # Compute and print aggregated analysis using the same function
    print(f"\n{'='*80}")
    print("AGGREGATED ANALYSIS ACROSS ALL DATASETS")
    print(f"{'='*80}")
    
    all_datasets_stats = compute_dataset_stats(locs)
    print_stats_compact("All_Datasets_Combined", all_datasets_stats)
    
    print("\nAll datasets")
    print(debug_str_filterables(locs.iter_all()))
    return locs, all_datasets_stats, items



def abbreviated(obj, list_max_head=6, list_max_tail=2, list_inline_cutoff=16, _depth=0):
    """
    Recursively abbreviate lists in obj.
    - If list is longer than list_max_head + list_max_tail, show head, '...', tail.
    - If flat list shorter than list_inline_cutoff, render as single line.
    """
    # Abbreviate lists/tuples
    if isinstance(obj, (list, tuple)):
        L = len(obj)
        if L > list_max_head + list_max_tail:
            head = [abbreviated(x, list_max_head, list_max_tail, list_inline_cutoff, _depth+1) for x in obj[:list_max_head]]
            tail = [abbreviated(x, list_max_head, list_max_tail, list_inline_cutoff, _depth+1) for x in obj[-list_max_tail:]]
            new = head + ['...'] + tail
        else:
            new = [abbreviated(x, list_max_head, list_max_tail, list_inline_cutoff, _depth+1) for x in obj]
        # For flat lists, try to join into one line if not too deep
        if all(not isinstance(x, (list, tuple, dict)) for x in obj) and L <= list_inline_cutoff and _depth < 4:
            return new  # Let pprint print inline
        if isinstance(obj, tuple):
            return tuple(new)
        return new
    # Recursively process dicts
    if isinstance(obj, dict):
        return {k: abbreviated(v, list_max_head, list_max_tail, list_inline_cutoff, _depth+1) for k, v in obj.items()}
    # Other: just return
    return obj




def main():
    #print("----- GPT-4.1")
    #do_with_config(
    #    ProbeConfig(
    #        datasets=[
    #            DatasetName.humaneval_plus,
    #            DatasetName.livecodebench,
    #            DatasetName.mbpp_plus,
    #        ],
    #        gen_model_name = OpenAiModelNames.gpt_4_1_mini,
    #        max_problems=300,
    #    )
    #)
    print("----- GPT-4o")
    #do_with_config(
    #    ProbeConfig(
    #        datasets=[
    #            DatasetName.humaneval_plus,
    #            DatasetName.livecodebench,
    #            DatasetName.mbpp_plus,
    #        ],
    #        gen_model_name = OpenAiModelNames.gpt_4o,
    #        max_problems=700,
    #    )
    #)
    #print("= Repocod =")
    #do_with_config(
    #    ProbeConfig(
    #        datasets=[
    #            DatasetName.repocod,
    #        ],
    #        gen_model_name = OpenAiModelNames.gpt_4o,
    #        max_problems=200,
    #    )
    #)
    #exit()
    #print("----- GPT-4o-mini")
    #do_with_config(
    #    ProbeConfig(
    #        datasets=[
    #            DatasetName.humaneval_plus,
    #            DatasetName.livecodebench,
    #            DatasetName.mbpp_plus,
    #        ],
    #        gen_model_name = OpenAiModelNames.gpt_4o_mini,
    #        max_problems=300,
    #    )
    #)
    print("---- GPT-4o")
    #do_with_config(
    #    dataclasses.replace(BASE_PAPER_CONFIG, max_problems=20)
    #)
    #locs, all_datasets_stats, items = do_with_config(
    #    ProbeConfig(
    #        datasets=[
    #            DatasetName.humaneval_plus,
    #            #DatasetName.livecodebench,
    #            #DatasetName.mbpp_plus,
    #        ],
    #        max_problems=10,
    #        gen_model_name = OpenAiModelNames.gpt_4o,
    #    )
    #)
    #for loc in locs.iter_passed_filtered():
    #    print("num logprobs")
    #    print(loc.base_solve.lm_prediction[0].prompt.logprobs)
    #    print("Logprobs")
    #    print(loc.base_solve.lm_prediction[0].top_token_logprobs)
    #    exit()

    print("---- Qwen2.5-7B")
    do_with_config(
        BASE_PAPER_CONFIG_QWEN
    )


if __name__ == "__main__":
    print("Hello world")
    main()
