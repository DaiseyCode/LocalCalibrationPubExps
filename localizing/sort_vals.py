from localizing.direct_prompt import pull_last_md_block_from_completion
from localizing.probe.probe_data_gather import get_or_serialize_tokenized_localizations, make_basic_serialize_key_args
from pape.configs import BASE_PAPER_CONFIG


def main():
    config = BASE_PAPER_CONFIG
    dev_mode = True
    locs = get_or_serialize_tokenized_localizations(
        config, dev_mode=dev_mode)

    # Collect all examples
    examples = []
    for loc in locs.iter_passed_filtered():
        num_not_keeps = len(loc.gt_base_token_keeps) - sum(loc.gt_base_token_keeps)
        total_toks = len(loc.base_tokens)
        text = loc.base_solve.lm_prediction[0].prompt.text[0].content
        line_labels = [
            min(loc.gt_base_token_keeps[span[0]:span[1]])
            for span in loc.get_line_spans()
        ]
        md_pull = pull_last_md_block_from_completion(text)
        if md_pull is not None:
            text = md_pull
        examples.append((loc, num_not_keeps, total_toks, text, line_labels))

    # Filter for low num_not_keeps (> 0) and low total_toks
    filtered_examples = [
        (loc, not_keeps, total, text, line_labels) for loc, not_keeps, total, text, line_labels in examples
        if 0 < not_keeps <= 30
        and total <= 50  # Low not_keeps but > 0, and low total tokens
        and len(text) < 500
        #and sum(1 if label == 1 else 0 for label in line_labels) < len(line_labels) - 1
    ]

    # Sort by total_toks ascending, then by num_not_keeps ascending
    filtered_examples.sort(key=lambda x: (x[2], x[1]))

    print(f"Found {len(filtered_examples)} examples with 1-3 non-kept tokens and ≤50 total tokens:")
    print("=" * 80)

    for loc, not_keeps, total, text, line_labels in filtered_examples:
        print("---")
        print(loc.base_solve.problem.problem_id)
        print(text)
        print("Base (with diff)")
        for i, (token, keep) in enumerate(zip(loc.base_tokens, loc.gt_base_token_keeps, strict=True)):
            if not keep:
                print(f"\033[91m{token}\033[0m", end="")
            else:
                print(token, end="")
        print()
        print("Fix")
        print(loc.get_gt_fix_text())
        print("line lablejs")
        print(line_labels)

    # Also show some stats
    print("\n" + "=" * 80)
    print("Summary:")
    print(f"Total examples analyzed: {len(examples)}")
    print(f"Filtered examples (1-3 non-kept, ≤50 total): {len(filtered_examples)}")

    # Show distribution of num_not_keeps in filtered set
    not_keeps_counts = {}
    for _, not_keeps, _, _, _ in filtered_examples:
        not_keeps_counts[not_keeps] = not_keeps_counts.get(not_keeps, 0) + 1

    print(f"Distribution of non-kept tokens in filtered set: {not_keeps_counts}")


if __name__ == "__main__":
    main()