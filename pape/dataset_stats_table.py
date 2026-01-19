from pprint import pprint
import dataclasses
from pathlib import Path
from synthegrator.synthdatasets import DatasetName
from synthegrator.synthdatasets import yield_problems_from_name
from typing import TypeVar
import re
from synthegrator.synthdatasets import DatasetName
from localizing.filter_helpers import debug_str_filterables
from localizing.localizing_structs import BaseLocalization, LocalizationList
from localizing.probe.embedding_styles import EmbeddingStyle
from localizing.probe.probe_data_gather import get_or_serialize_localizations_embedded
from localizing.probe.agg_models.agg_config import ProbeConfig
from collections import Counter, defaultdict
from lmwrapper.openai_wrapper import OpenAiModelNames
import numpy as np
from itertools import combinations
from localizing.probe.agg_models.agg_config import ProbeConfig
from pape.dataset_flow_stats import do_with_config
from pape.configs import BASE_PAPER_CONFIG, BASE_PAPER_CONFIG_QWEN
from pape.texrendering import generate_latex_table, TABLE_DIR 


def make_dataset_stats_table(config: ProbeConfig, table_name: str):
    #locs, all_datasets_stats, strat_stats = do_with_config(
    #    ProbeConfig(
    #        datasets=[
    #            DatasetName.humaneval_plus,
    #            DatasetName.livecodebench,
    #            DatasetName.mbpp_plus,
    #        ],
    #        gen_model_name = OpenAiModelNames.gpt_4_1_mini,
    #        max_problems=50,
    #        embedding_style=EmbeddingStyle.LAST_LAYER,
    #    )
    #)
    locs, all_datasets_stats, strat_stats = do_with_config(
        config
        #ProbeConfig(
        #    datasets=[
        #        DatasetName.humaneval_plus,
        #        DatasetName.livecodebench,
        #        DatasetName.mbpp_plus,
        #        DatasetName.repocod,
        #    ],
        #    gen_model_name = "Qwen/Qwen2.5-Coder-7B-Instruct",
        #    max_problems=50,
        #)
    )
    print_dict_structure(all_datasets_stats)
    #exit()
    rows = []
    def make_row(name, stat):
        print(name)
        pass_percent = stat['og_solves'] / (stat['og_solves'] + stat['og_not_solves']) * 100
        name = {
            "humaneval_plus": "HumanEval+",
            "livecodebench": "LiveCodeBench",
            "mbpp_plus": "MBPP+",
            "repocod": "RepoCod-s",
        }.get(name, name)
        model_name = {
            "Qwen/Qwen2.5-Coder-7B-Instruct": "Qwen2.5Coder",
        }.get(config.gen_model_name, config.gen_model_name)
        rows.append({
            "Model Name": model_name,
            "Dataset": name,
            "Solve Step": {
                "Available": str(stat['total_problems']),
                "Pass $\\checkmark$": f"{stat['og_solves']} ({pass_percent:.0f}%)",
                "Fail $\\times$": f"{stat['og_not_solves']}",
            },
            "Patch Step": {
                "\\makecell{Count\\\\w/ Model}": str(stat.get('num_patched_non_gt', 0)),
                "\\makecell{Avg Toks\\\\w/ Model}": f"{stat['passing_patched_not_keep_token_count_non_gt_stats']['mean']:.1f} ({stat['passing_patched_not_keep_token_rate_non_gt_stats']['mean'] * 100:.0f}%)" if stat['passing_patched_not_keep_token_count_non_gt_stats']['mean'] is not None else "N/A",
                "\\makecell{Count\\\\w/ Ref}": str(stat.get('num_patched_gt', 0)),
                "\\makecell{Avg Toks\\\\w/ Ref}": f"{stat['passing_patched_not_keep_token_count_gt_stats']['mean']:.1f} ({stat['passing_patched_not_keep_token_rate_gt_stats']['mean'] * 100:.0f}%)" if stat['passing_patched_not_keep_token_count_gt_stats']['mean'] is not None else "N/A",
                #"Count": str(stat['num_patched']),
                #"Avg Tokens (All)": f"{stat['passing_patched_not_keep_token_count_stats']['mean']:.1f} ({stat['passing_patched_not_keep_token_rate_stats']['mean'] * 100:.0f}%)" if stat['passing_patched_not_keep_token_count_stats']['mean'] is not None else "N/A",
            },
            "Final Used Data": {
                "Problems": str(stat['pass_all_filters']),
                "\\makecell{Total\\\\Tokens}": f"{stat['combined_total_tokens_stats']['sum'] / 1000:.0f}k",
            },
        })

    for name, stat in strat_stats.items():
        make_row(name, stat)
    rows.append("\\midrule")
    make_row("Total", all_datasets_stats)

    pprint(rows)

    # Generate and print LaTeX table
    print("\n" + "="*50)
    print("LaTeX Table:")
    print("="*50)
    if table_name == "gpt_4o":
        caption_content=(
            "Descriptive statistics on the dataset contents split by source dataset for GPT-4o. "
            "The 'Solve Step Pass $\\checkmark$' represents a Pass@1. "
            "The 'Patch Step Count' shows the number of failing solutions successfully patched. It split by whether the patch was generated with a model, or having to fallback on a reference solution."
            "The 'Avg Tokens' shows the number tokens not kept in the patch along with what percent of the "
            "tokens in the solution were not kept. The model generated patches are typically more more minimal. Final data shows used problems and token totals. "
        )
    else:
        caption_content=(
            "Descriptive statistics on the dataset contents split by source dataset for Qwen 2.5 Coder 7B Instruct. See above for more details on columns."
        )


    latex_table = generate_latex_table(
        rows,
        full_width=True,
        caption_content=caption_content,
        label=f"tab:dataset_stats_{table_name}",
        caption_at_top=False,
        resize_to_fit=True,
    )
    print(latex_table)
    
    # Write to file
    output_file = TABLE_DIR / f"dataset_stats_table_{table_name}.tex"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(latex_table)
    print(f"\nLaTeX table written to: {output_file}")



def main():
    make_dataset_stats_table(BASE_PAPER_CONFIG, "gpt_4o")
    make_dataset_stats_table(BASE_PAPER_CONFIG_QWEN, "qwen7b")


def print_dict_structure(d: dict, indent: int = 0):
    """Print dictionary structure with proper indentation for readability."""
    indent_str = "  " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{indent_str}{k}:")
            print_dict_structure(v, indent + 1)
        else:
            v_str = str(v)[:10]
            if len(str(v)) > 10:
                v_str += "..."
            print(f"{indent_str}{k}: {v_str}")


if __name__ == "__main__":
    main()