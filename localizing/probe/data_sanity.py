from typing import Dict
from localizing.probe.probe_data_gather import (
    get_or_serialize_localizations_embedded,
    localizations_to_grouped_vec_label_dataset,
)
from localizing.probe.embedding_styles import EmbeddingStyle
from localizing.multi_data_gathering import DatasetName
from lmwrapper.openai_wrapper import OpenAiModelNames
from localizing.probe.probe_models_agg import AggQueriesDataLoader


def print_batch_line_info(batch, max_problems=10, max_lines=1000, show_perfect_problems=False):
    """
    Print code with colored tokens and line indicators showing keep/remove labels.
    
    Args:
        batch: AggregatedBatch object from AggQueriesDataLoader
        max_problems: Maximum number of problems to display
        max_lines: Maximum number of lines to display per problem
        show_perfect_problems: If False, only show problems with errors (perfect problems show as summary)
        
    Returns:
        Dict with aggregate statistics: {'total_tokens', 'total_positive_tokens', 'total_lines', 'total_positive_lines'}
    """
    # ANSI color codes
    GREEN = '\033[92m'  # Green for keep tokens
    RED = '\033[91m'    # Red for remove tokens
    RESET = '\033[0m'   # Reset color
    BOLD = '\033[1m'    # Bold text
    
    print(f"Batch contains {batch.get_num_problems()} problems, {batch.get_num_total_tokens()} total tokens")
    print(f"Embeddings shape: {batch.embeddings.shape}")
    print(f"Labels shape: {batch.labels.shape}")
    
    # Track aggregate statistics across all problems in this batch
    batch_total_tokens = 0
    batch_total_positive_tokens = 0
    batch_total_lines = 0
    batch_total_positive_lines = 0
    per_problem_stats = []  # Track individual problem statistics
    
    # Iterate through problems in the batch
    for problem_idx in range(min(batch.get_num_problems(), max_problems)):
        # Get problem metadata first to check if perfect
        if batch.problem_metadata and problem_idx < len(batch.problem_metadata):
            metadata = batch.problem_metadata[problem_idx]
            base_tokens = metadata.get('base_tokens', [])
        else:
            base_tokens = []
            continue
        
        # Get problem span and line spans to calculate statistics
        if problem_idx >= len(batch.problem_spans) or problem_idx >= len(batch.line_spans):
            continue
            
        prob_start, prob_end = batch.problem_spans[problem_idx]
        problem_lines = batch.line_spans[problem_idx]
        
        # Pre-calculate statistics to determine if problem is perfect
        total_tokens = 0
        total_positive_tokens = 0
        total_lines = 0
        total_positive_lines = 0
        
        for line_idx, (line_start, line_end) in enumerate(problem_lines):
            # Get tokens for this line (relative to problem, not batch)
            relative_start = line_start - prob_start
            relative_end = line_end - prob_start
            
            if not base_tokens or relative_end > len(base_tokens):
                continue
                
            # Get labels for this line
            if line_end > len(batch.labels):
                continue
                
            line_labels = batch.labels[line_start:line_end]
            labels_np = line_labels.cpu().numpy() if hasattr(line_labels, 'cpu') else line_labels.numpy()
            
            # Count token statistics
            positive_tokens = (labels_np == 1).sum()
            total_tokens += len(labels_np)
            total_positive_tokens += positive_tokens
            total_lines += 1
            
            # Determine line-level label (AND operation - all tokens must be good)
            line_label = 1 if (labels_np == 1).all() else 0
            total_positive_lines += line_label
        
        # Calculate error rates
        problem_token_error_rate = (total_tokens - total_positive_tokens) / total_tokens if total_tokens > 0 else 0
        problem_line_error_rate = (total_lines - total_positive_lines) / total_lines if total_lines > 0 else 0
        is_perfect = (problem_token_error_rate == 0.0 and problem_line_error_rate == 0.0)
        
        # Store per-problem statistics
        per_problem_stats.append({
            'problem_id': metadata.get('problem_id', f"problem_{problem_idx}"),
            'total_tokens': total_tokens,
            'positive_tokens': total_positive_tokens,
            'token_error_rate': problem_token_error_rate,
            'total_lines': total_lines,
            'positive_lines': total_positive_lines,
            'line_error_rate': problem_line_error_rate
        })
        
        # Add to batch totals
        batch_total_tokens += total_tokens
        batch_total_positive_tokens += total_positive_tokens
        batch_total_lines += total_lines
        batch_total_positive_lines += total_positive_lines
        
        # Skip detailed display if perfect and show_perfect_problems is False
        if is_perfect and not show_perfect_problems:
            print(f"\n{BOLD}PROBLEM {problem_idx + 1} - {metadata.get('problem_id', 'Unknown')}{RESET}")
            print(f"{GREEN}✓ {total_lines} lines all correct{RESET}")
            continue
        
        # Show detailed problem display
        print(f"\n{BOLD}{'='*80}")
        print(f"PROBLEM {problem_idx + 1}")
        print(f"{'='*80}{RESET}")
        print(f"Problem ID: {metadata.get('problem_id', 'Unknown')}")
        
        lines_displayed = 0
        print(f"\n{BOLD}Code with token-level labels (Green=KEEP, Red=REMOVE):{RESET}")
        print("-" * 80)
        
        # Display each line with colors
        for line_idx, (line_start, line_end) in enumerate(problem_lines):
            if lines_displayed >= max_lines:
                print(f"\n... {len(problem_lines) - lines_displayed} more lines truncated ...")
                break
                
            lines_displayed += 1
            
            # Get tokens for this line (relative to problem, not batch)
            relative_start = line_start - prob_start
            relative_end = line_end - prob_start
            
            if not base_tokens or relative_end > len(base_tokens):
                print(f"? [Unable to extract line {line_idx + 1}]")
                continue
                
            line_tokens = base_tokens[relative_start:relative_end]
            
            # Get labels for this line
            if line_end > len(batch.labels):
                print(f"? [Unable to extract labels for line {line_idx + 1}]")
                continue
                
            line_labels = batch.labels[line_start:line_end]
            labels_np = line_labels.cpu().numpy() if hasattr(line_labels, 'cpu') else line_labels.numpy()
            
            # Determine line-level label (AND operation - all tokens must be good)
            line_label = 1 if (labels_np == 1).all() else 0
            
            # Line indicator
            line_indicator = f"{GREEN}✓{RESET}" if line_label == 1 else f"{RED}✗{RESET}"
            
            # Build colored line text
            colored_line = ""
            for token, label in zip(line_tokens, labels_np):
                if label == 1:
                    colored_line += f"{GREEN}{token}{RESET}"
                else:
                    colored_line += f"{RED}{token}{RESET}"
            
            # Print the line with indicator (strip newlines to avoid double spacing)
            print(f"{line_indicator} {colored_line}", end='')
        
        print()
        # Print summary statistics
        token_percent = (total_positive_tokens / total_tokens * 100) if total_tokens > 0 else 0
        line_percent = (total_positive_lines / total_lines * 100) if total_lines > 0 else 0
        
        print("-" * 80)
        print(f"{BOLD}Statistics:{RESET}")
        print(f"Token-level: {total_positive_tokens}/{total_tokens} positive ({token_percent:.1f}%)")
        print(f"Line-level:  {total_positive_lines}/{total_lines} positive ({line_percent:.1f}%)")
        if lines_displayed < len(problem_lines):
            print(f"Lines displayed: {lines_displayed}/{len(problem_lines)}")
    
    # Show summary if there are more problems
    if batch.get_num_problems() > max_problems:
        print(f"\n{BOLD}... and {batch.get_num_problems() - max_problems} more problems in this batch{RESET}")
    
    # Return the aggregate statistics for this batch
    return {
        'total_tokens': batch_total_tokens,
        'total_positive_tokens': batch_total_positive_tokens,
        'total_lines': batch_total_lines,
        'total_positive_lines': batch_total_positive_lines,
        'per_problem_stats': per_problem_stats
    }


def sanity_lines(
    style: EmbeddingStyle = EmbeddingStyle.LAST_LAYER,
    datasets: list[DatasetName] = [
        #DatasetName.livecodebench, 
        #DatasetName.humaneval_plus, 
        #DatasetName.mbpp_plus,
        DatasetName.dypy_line_completion,
    ],
    gen_model_name: str = OpenAiModelNames.gpt_4o_mini,
    embed_lm_name: str = "Qwen/Qwen2.5-Coder-0.5B",
    fix_reference: str = OpenAiModelNames.o4_mini_2025_04_16,
    token_weight: float = 0.3,
    line_weight: float = 0.7,
    save_plots: bool = True,
    filter_to_original_fails: bool = False,
    total_problems_to_show: int = 1000,
    show_perfect_problems: bool = True,
) -> Dict[str, float]:
    print(f"\n{'='*80}")
    print(f"Training with {style.value} embedding style - Aggregated Line Probe")
    datasets_str = ", ".join(str(ds) for ds in datasets)
    print(f"Using datasets: {datasets_str}")
    print(f"Using generation model: {gen_model_name}")
    print(f"Using embedding model: {embed_lm_name}")
    print(f"Loss weights: token={token_weight}, line={line_weight}")
    print(f"{'='*80}")
    
    # Create ProbeConfig from parameters
    from localizing.probe.agg_models.agg_config import ProbeConfig
    config = ProbeConfig(
        datasets=datasets,
        gen_model_name=gen_model_name,
        embed_lm_name=embed_lm_name,
        fix_reference=fix_reference,
        filter_to_original_fails=filter_to_original_fails,
        max_problems=200,
        embedding_style=style,
    )
    
    # Get localizations dataset
    localizations = get_or_serialize_localizations_embedded(config)
    print(f"Loaded {len(localizations)} localizations")
    print(f"Datasets present: {', '.join(str(ds) for ds in localizations.get_dataset_name_set())}")
    
    # Convert to grouped dataset
    dataset = localizations_to_grouped_vec_label_dataset(localizations, n_folds=5)
    print(dataset)
    
    # Create a dataloader to inspect the line-level data
    print(f"\n{'='*80}")
    print("INSPECTING LINE-LEVEL DATA WITH AggQueriesDataLoader")
    print(f"{'='*80}")
    
    # Use the first fold's training data for inspection
    if dataset.folds_train_test:
        train_data, _ = dataset.folds_train_test[0]
        
        # Create dataloader
        dataloader = AggQueriesDataLoader(
            dataset=train_data,
            batch_size=max(1, len(train_data) // 10),
            shuffle=False,
            include_line_spans=True
        )
        
        # Iterate through batches until we've seen enough problems
        problems_seen = 0
        batch_count = 0
        
        # Track aggregate statistics across all problems
        total_aggregate_tokens = 0
        total_aggregate_positive_tokens = 0
        total_aggregate_lines = 0
        total_aggregate_positive_lines = 0
        all_problem_stats = []  # Collect all individual problem statistics
        
        for batch in dataloader:
            if batch is None:
                continue
                
            batch_count += 1
            batch_problems = batch.get_num_problems()
            
            # Calculate how many problems to show from this batch
            problems_to_show = min(batch_problems, total_problems_to_show - problems_seen)
            
            print(f"\n{'='*80}")
            print(f"BATCH {batch_count}: Showing {problems_to_show}/{batch_problems} problems")
            print(f"{'='*80}")
            
            # Get batch statistics and add to aggregates
            batch_stats = print_batch_line_info(
                batch, 
                max_problems=problems_to_show,
                show_perfect_problems=show_perfect_problems,
            )
            if batch_stats:
                total_aggregate_tokens += batch_stats['total_tokens']
                total_aggregate_positive_tokens += batch_stats['total_positive_tokens']
                total_aggregate_lines += batch_stats['total_lines']
                total_aggregate_positive_lines += batch_stats['total_positive_lines']
                all_problem_stats.extend(batch_stats['per_problem_stats'])
            
            problems_seen += problems_to_show
            
            # Stop if we've seen enough problems
            if problems_seen >= total_problems_to_show:
                break
        
        # Calculate aggregate percentages
        aggregate_token_percent = (total_aggregate_positive_tokens / total_aggregate_tokens * 100) if total_aggregate_tokens > 0 else 0
        aggregate_line_percent = (total_aggregate_positive_lines / total_aggregate_lines * 100) if total_aggregate_lines > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"SUMMARY: Displayed {problems_seen} problems across {batch_count} batches")
        print(f"{'='*80}")
        print(f"AGGREGATE STATISTICS ACROSS ALL {problems_seen} PROBLEMS:")
        print(f"Token-level: {total_aggregate_positive_tokens}/{total_aggregate_tokens} positive ({aggregate_token_percent:.1f}%)")
        print(f"Line-level:  {total_aggregate_positive_lines}/{total_aggregate_lines} positive ({aggregate_line_percent:.1f}%)")
        
        # Detailed per-problem analysis
        if all_problem_stats:
            import numpy as np
            
            # ANSI color codes for formatting
            BOLD = '\033[1m'
            RESET = '\033[0m'
            
            # Extract error rates
            token_error_rates = [p['token_error_rate'] for p in all_problem_stats]
            line_error_rates = [p['line_error_rate'] for p in all_problem_stats]
            
            # Perfect problems (0% error rate)
            perfect_token_problems = sum(1 for rate in token_error_rates if rate == 0.0)
            perfect_line_problems = sum(1 for rate in line_error_rates if rate == 0.0)
            
            print(f"\n{BOLD}PER-PROBLEM ANALYSIS:{RESET}")
            print(f"Perfect problems (0% errors):")
            print(f"  Token-level: {perfect_token_problems}/{len(all_problem_stats)} ({perfect_token_problems/len(all_problem_stats)*100:.1f}%)")
            print(f"  Line-level:  {perfect_line_problems}/{len(all_problem_stats)} ({perfect_line_problems/len(all_problem_stats)*100:.1f}%)")
            
            # Error distribution buckets
            def count_in_bucket(rates, min_rate, max_rate):
                return sum(1 for rate in rates if min_rate <= rate < max_rate)
            
            print(f"\nError rate distribution:")
            print(f"Token-level buckets:")
            print(f"  0%:        {count_in_bucket(token_error_rates, 0.0, 0.0001):3d} problems")
            print(f"  1-10%:     {count_in_bucket(token_error_rates, 0.0001, 0.1):3d} problems")
            print(f"  11-25%:    {count_in_bucket(token_error_rates, 0.1, 0.25):3d} problems")
            print(f"  26-50%:    {count_in_bucket(token_error_rates, 0.25, 0.5):3d} problems")
            print(f"  51%+:      {count_in_bucket(token_error_rates, 0.5, 1.01):3d} problems")
            
            print(f"Line-level buckets:")
            print(f"  0%:        {count_in_bucket(line_error_rates, 0.0, 0.0001):3d} problems")
            print(f"  1-10%:     {count_in_bucket(line_error_rates, 0.0001, 0.1):3d} problems")
            print(f"  11-25%:    {count_in_bucket(line_error_rates, 0.1, 0.25):3d} problems")
            print(f"  26-50%:    {count_in_bucket(line_error_rates, 0.25, 0.5):3d} problems")
            print(f"  51%+:      {count_in_bucket(line_error_rates, 0.5, 1.01):3d} problems")
            
            # Statistical measures
            token_error_array = np.array(token_error_rates)
            line_error_array = np.array(line_error_rates)
            
            print(f"\nError rate statistics:")
            print(f"Token-level: mean={token_error_array.mean():.3f}, median={np.median(token_error_array):.3f}, std={token_error_array.std():.3f}")
            print(f"             75th percentile={np.percentile(token_error_array, 75):.3f}, 90th percentile={np.percentile(token_error_array, 90):.3f}")
            print(f"Line-level:  mean={line_error_array.mean():.3f}, median={np.median(line_error_array):.3f}, std={line_error_array.std():.3f}")
            print(f"             75th percentile={np.percentile(line_error_array, 75):.3f}, 90th percentile={np.percentile(line_error_array, 90):.3f}")
            
            # Concentration analysis - how much error comes from worst problems
            sorted_token_errors = sorted(token_error_rates, reverse=True)
            sorted_line_errors = sorted(line_error_rates, reverse=True)
            
            worst_10_pct = int(0.1 * len(all_problem_stats))
            worst_20_pct = int(0.2 * len(all_problem_stats))
            
            if worst_10_pct > 0:
                token_error_in_worst_10 = sum(sorted_token_errors[:worst_10_pct])
                token_error_in_worst_20 = sum(sorted_token_errors[:worst_20_pct])
                total_token_error = sum(token_error_rates)
                
                line_error_in_worst_10 = sum(sorted_line_errors[:worst_10_pct])
                line_error_in_worst_20 = sum(sorted_line_errors[:worst_20_pct])
                total_line_error = sum(line_error_rates)
                
                print(f"\nError concentration:")
                if total_token_error > 0:
                    print(f"Token errors: worst 10% of problems account for {token_error_in_worst_10/total_token_error*100:.1f}% of total error")
                    print(f"              worst 20% of problems account for {token_error_in_worst_20/total_token_error*100:.1f}% of total error")
                if total_line_error > 0:
                    print(f"Line errors:  worst 10% of problems account for {line_error_in_worst_10/total_line_error*100:.1f}% of total error")
                    print(f"              worst 20% of problems account for {line_error_in_worst_20/total_line_error*100:.1f}% of total error")
            
            # Problem size analysis - how does problem length relate to errors
            print(f"\n{BOLD}PROBLEM SIZE ANALYSIS:{RESET}")
            
            # Define line count buckets
            line_buckets = [
                (1, 1, "1 line"),
                (2, 10, "2-10 lines"),
                (11, 50, "11-50 lines"),
                (51, float('inf'), "51+ lines")
            ]
            
            total_token_errors = sum(p['total_tokens'] - p['positive_tokens'] for p in all_problem_stats)
            total_line_errors = sum(p['total_lines'] - p['positive_lines'] for p in all_problem_stats)
            total_all_lines = sum(p['total_lines'] for p in all_problem_stats)
            total_all_tokens = sum(p['total_tokens'] for p in all_problem_stats)
            
            # Prepare data for tabulate
            from tabulate import tabulate
            
            table_data = []
            headers = [
                "Size Range\n(lines)",
                "Count\n(% all)",
                "Line\nErr Rate",
                "Token\nErr Rate", 
                "% Line\nErrors",
                "% Token\nErrors",
                "% of All\nLines",
                "% of All\nTokens"
            ]
            
            total_problems = len(all_problem_stats)
            
            for min_lines, max_lines, bucket_name in line_buckets:
                # Filter problems in this bucket
                if max_lines == float('inf'):
                    bucket_problems = [p for p in all_problem_stats if p['total_lines'] >= min_lines]
                else:
                    bucket_problems = [p for p in all_problem_stats if min_lines <= p['total_lines'] <= max_lines]
                
                if not bucket_problems:
                    continue
                
                # Calculate error rates for this bucket
                bucket_total_lines = sum(p['total_lines'] for p in bucket_problems)
                bucket_good_lines = sum(p['positive_lines'] for p in bucket_problems)
                bucket_total_tokens = sum(p['total_tokens'] for p in bucket_problems)
                bucket_good_tokens = sum(p['positive_tokens'] for p in bucket_problems)
                
                bucket_line_errors = bucket_total_lines - bucket_good_lines
                bucket_token_errors = bucket_total_tokens - bucket_good_tokens
                
                # Error rates
                bucket_line_error_rate = bucket_line_errors / bucket_total_lines if bucket_total_lines > 0 else 0
                bucket_token_error_rate = bucket_token_errors / bucket_total_tokens if bucket_total_tokens > 0 else 0
                
                # Fraction of total errors
                line_error_fraction = bucket_line_errors / total_line_errors if total_line_errors > 0 else 0
                token_error_fraction = bucket_token_errors / total_token_errors if total_token_errors > 0 else 0
                
                # Fraction of all problems, lines, and tokens
                problem_fraction = len(bucket_problems) / total_problems if total_problems > 0 else 0
                lines_fraction = bucket_total_lines / total_all_lines if total_all_lines > 0 else 0
                tokens_fraction = bucket_total_tokens / total_all_tokens if total_all_tokens > 0 else 0
                
                # Format the data row
                row = [
                    bucket_name,
                    f"{len(bucket_problems)} ({problem_fraction*100:.1f}%)",
                    f"{bucket_line_error_rate*100:.1f}%",
                    f"{bucket_token_error_rate*100:.1f}%",
                    f"{line_error_fraction*100:.1f}%",
                    f"{token_error_fraction*100:.1f}%",
                    f"{lines_fraction*100:.1f}%",
                    f"{tokens_fraction*100:.1f}%"
                ]
                table_data.append(row)
            
            print("Error rates and error distribution by problem size:")
            print(tabulate(table_data, headers=headers, tablefmt="plain"))
            
            # Additional statistics about problem sizes
            problem_sizes = [p['total_lines'] for p in all_problem_stats]
            print(f"\nProblem size statistics:")
            print(f"  Mean lines per problem: {np.mean(problem_sizes):.1f}")
            print(f"  Median lines per problem: {np.median(problem_sizes):.1f}")
            print(f"  Range: {min(problem_sizes)} to {max(problem_sizes)} lines")
            print(f"  75th percentile: {np.percentile(problem_sizes, 75):.1f} lines")
            print(f"  90th percentile: {np.percentile(problem_sizes, 90):.1f} lines")
        
        print(f"{'='*80}")
    else:
        print("No folds available in dataset")


def main():
    sanity_lines()


if __name__ == "__main__":
    main()