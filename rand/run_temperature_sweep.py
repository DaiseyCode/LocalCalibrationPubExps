#!/usr/bin/env python3

import os
import subprocess
import time
from multi_compare import DatasetName, cur_path
from itertools import product
import sys
import pandas as pd
from pathlib import Path

def temperature_sweep():
    datasets = [
        DatasetName.humaneval,
        DatasetName.mbpp,
        DatasetName.dypy_line_completion
    ]
    multi_samples_list = [10, 50]
    temperatures = [1.0, 0.7, 0.1]

    # Generate combinations in the same order as your original loop
    combinations = []
    for dataset in datasets:
        for multi_samples in multi_samples_list:
            for temperature in temperatures:
                combinations.append((dataset, multi_samples, temperature))

    gpu_ids = [i for i in range(7)]

    processes = []
    gpu_queue = gpu_ids.copy()

    # Start processes in the order of combinations
    for idx, (dataset, multi_samples, temperature) in enumerate(combinations):
        # Wait until a GPU is available
        while not gpu_queue:
            # Wait for any process to finish
            for p_info in processes:
                p = p_info['process']
                if p.poll() is not None:
                    # Process has finished
                    gpu_id = p_info['gpu_id']
                    gpu_queue.append(gpu_id)
                    processes.remove(p_info)
                    break
            else:
                # No process has finished yet
                time.sleep(1)
                continue

        # Assign the next available GPU
        gpu_id = gpu_queue.pop(0)

        # Set the environment variable for CUDA
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        cmd = [
            sys.executable, "run_single_multi_compare.py",
            "--dataset", dataset.name,
            "--multi_samples", str(multi_samples),
            "--temperature", str(temperature)
        ]

        # Start the subprocess
        print(f"Starting process for dataset={dataset.name}, multi_samples={multi_samples}, temperature={temperature}, on GPU {gpu_id}")
        p = subprocess.Popen(cmd, env=env)
        processes.append({'process': p, 'gpu_id': gpu_id})

    # Wait for all remaining processes to finish
    while processes:
        for p_info in processes:
            p = p_info['process']
            if p.poll() is not None:
                gpu_id = p_info['gpu_id']
                gpu_queue.append(gpu_id)
                processes.remove(p_info)
                break
        else:
            time.sleep(1)
            continue

    # Aggregate results after all processes have completed
    aggregate_results()

def aggregate_results():
    csv_files = list(Path(cur_path).glob("temperature_sweep_*.csv"))
    df_list = [pd.read_csv(csv_file) for csv_file in csv_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    output_file = cur_path / "aggregated_temperature_sweep_results.csv"
    combined_df.to_csv(output_file, index=False)
    print(f"Aggregated results saved to {output_file}")

if __name__ == "__main__":
    temperature_sweep()
