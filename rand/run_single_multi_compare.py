#!/usr/bin/env python3

import os
import argparse
import pandas as pd
from pathlib import Path
import sys

# Import the multi_explore function and related classes from your module
from multi_compare import multi_explore, DatasetName, cur_path

def main():
    parser = argparse.ArgumentParser(description="Run multi_explore with specified parameters.")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., humaneval, mbpp, dypy_line_completion)')
    parser.add_argument('--multi_samples', type=int, required=True, help='Number of multi samples (e.g., 10, 50)')
    parser.add_argument('--temperature', type=float, required=True, help='Temperature value (e.g., 1.0, 0.7, 0.1)')
    args = parser.parse_args()

    # Read CUDA_VISIBLE_DEVICES from the environment (set by the dispatcher script)
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"Using GPU(s): {cuda_visible_devices}")

    # Import modules that might initialize CUDA after setting the environment variable
    import torch
    import transformers

    # Map the dataset string back to the DatasetName enum
    dataset = DatasetName[args.dataset]

    # Run the multi_explore function
    (
        res, fix_rate, orig_solve_rate,
        old_junk_rate, multi_bust_rate, final_multi_used
    ) = multi_explore(
        use_eval_plus=False,
        dataset=dataset,
        multi_temperature=args.temperature,
        fix_reference="gpt4" if dataset in (DatasetName.humaneval, DatasetName.mbpp) else "gt",
        max_problems=None if dataset == DatasetName.humaneval else 500,
        multi_samples=args.multi_samples,
    )

    res_scaled = res.to_platt_scaled()
    data = {
        "dataset": args.dataset,
        "fix_rate": fix_rate,
        "orig_solve_rate": orig_solve_rate,
        "old_junk_rate": old_junk_rate,
        "multi_bust_rate": multi_bust_rate,
        "final_multi_used": final_multi_used,
        "temperature": args.temperature,
        "multi_samples": args.multi_samples,
        "ece": res.ece,
        "brier": res.brier_score,
        "base_rate": res.base_rate,
        "ece_scaled": res_scaled.ece,
        "brier_scaled": res_scaled.brier_score,
        "ss": res.skill_score,
    }

    # Save the results to a CSV file
    unix_time = int(pd.Timestamp.now().timestamp())
    filename = cur_path / f"temperature_sweep_{args.dataset}_{args.multi_samples}_{args.temperature}_{unix_time}.csv"
    pd.DataFrame([data]).to_csv(filename, index=False)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    main()
