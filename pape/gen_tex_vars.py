from datetime import datetime
from typing import Literal
from localizing.predictions_repr import flatten_dict
from pape.configs import BASE_PAPER_CONFIG, BASE_PAPER_CONFIG_QWEN, DEFAULT_LINE_AGGREGATOR_INTRINSIC
from pathlib import Path
from pape.dataset_flow_stats import do_with_config
from pape.main_metrics_table import make_main_metrics_table_rows
from pape.texrendering import dict_to_latex_vars_cmds, get_git_info
import sys
import os
import re
from contextlib import redirect_stdout, contextmanager

cur_path = Path(__file__).parent


@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout output."""
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull):
            yield


def make_probe_tex_vars() -> dict[str, str]:
    print("Making probe tex vars...")
    vals = {
        "probe_hidden_dim": str(BASE_PAPER_CONFIG.hidden_dim),
        "probe_dropout_rate": str(BASE_PAPER_CONFIG.dropout_rate),
        "probe_learning_rate": str(BASE_PAPER_CONFIG.learning_rate),
        "probe_weight_decay": str(BASE_PAPER_CONFIG.weight_decay),
        "probe_num_epochs": str(BASE_PAPER_CONFIG.num_epochs),
        "probe_batch_size": str(BASE_PAPER_CONFIG.batch_size),
    }
    return vals


def make_dataset_stats_tex_vars() -> dict[str, str]:
    vals = {}
    combo_fails = 0
    combo_total_patches = 0
    combo_model_patches = 0
    for config in [
        BASE_PAPER_CONFIG,
        BASE_PAPER_CONFIG_QWEN,
    ]:
        print(f"Making dataset stats tex vars for {config.tidy_gen_name_code}...")
        with suppress_stdout():
            locs, all_datasets_stats, strat_stats = do_with_config(
                config
            )
        pass_percent = all_datasets_stats['og_solves'] / (all_datasets_stats['og_solves'] + all_datasets_stats['og_not_solves']) * 100
        
        # Calculate patch statistics
        total_patches = all_datasets_stats['num_patched_non_gt'] + all_datasets_stats['num_patched_gt']
        model_patches = all_datasets_stats['num_patched_non_gt']
        fails = all_datasets_stats['og_not_solves']
        combo_fails += fails
        combo_total_patches += total_patches
        combo_model_patches += model_patches

        # Calculate model patch fraction
        model_patch_fraction_pct = (model_patches / total_patches * 100)
        patch_frac = (total_patches) / fails
        patch_pct = patch_frac * 100
        
        vals.update({
            f"{config.tidy_gen_name_code}_solve_pct": f"{pass_percent:.0f}%",
            f"{config.tidy_gen_name_code}_total_patches": str(total_patches),
            f"{config.tidy_gen_name_code}_model_patch_pct": f"{model_patch_fraction_pct:.0f}%",
            f"{config.tidy_gen_name_code}_patch_pct": f"{patch_pct:.0f}%",
        })
    combo_patch_frac = combo_total_patches / combo_fails
    combo_patch_pct = combo_patch_frac * 100
    vals.update({
        "combo_patch_pct": f"{combo_patch_pct:.0f}%",
        "combo_model_patch_pct": f"{combo_model_patches / combo_total_patches * 100:.0f}%",
    })

    return vals


def make_intrinsic_stats_tex_vars() -> dict[str, str]:
    def agg_to_friendly(agg: Literal["mean", "gmean", "min"]) -> str:
        return {
            "mean": "mean",
            "gmean": "geometric mean",
            "min": "minimum",
        }[agg]
    return {
        "line_agg_intrinsic": agg_to_friendly(DEFAULT_LINE_AGGREGATOR_INTRINSIC),
    }


def make_results_vars() -> dict[str, str]:
    vals = {}
    for config in [
        BASE_PAPER_CONFIG,
        BASE_PAPER_CONFIG_QWEN,
    ]:
        print("Making main results vars for ", config.tidy_gen_name_code)
        with suppress_stdout():
            rows = make_main_metrics_table_rows(config)
        rows = [flatten_dict(r) for r in rows]
        for row in rows:
            technique = row['Technique']
            technique = technique.replace(" ", "_")
            technique = technique.replace(".", "_")
            technique = technique.replace("-", "")
            for k, v in row.items():
                if k == "Technique":
                    continue
                k = re.sub(r'\$[^$]*\$', '', k) # strip formulas
                k = k.replace("\\", "")
                k = k.replace("-", "")
                k = k.replace(" ", "")
                k = k.replace("(", "")
                k = k.replace(")", "")
                k = k.replace(".", "")
                k = k.replace(":", "")
                k = k.replace("=", "")
                k = k.replace("+", "")
                k = k.replace("*", "")
                k = k.replace("/", "")
                if "(" not in v:
                    vals[f"{config.tidy_gen_name_code}_{technique}_{k}"] = v
                else:
                    main_v = v.split("(")[0].strip()
                    # Get the value in the parentheses
                    sub_v = v.split("(")[1].split(")")[0].strip()
                    vals[f"{config.tidy_gen_name_code}_{technique}_{k}"] = main_v
                    vals[f"{config.tidy_gen_name_code}_{technique}_{k}_paren"] = sub_v
    return vals

        
def main():
    vars = {
        **make_probe_tex_vars(),
        **make_dataset_stats_tex_vars(),
        **make_intrinsic_stats_tex_vars(),
        **make_results_vars(),
    }
    cmds = dict_to_latex_vars_cmds(vars)
    header = f"""
% This file is auto-generated by gen_tex_vars.py.
% Commit {get_git_info()} on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    cmds = header + "\n" + cmds
    (cur_path / "gen/auto_vars.tex").write_text(cmds)
    print(cmds)
    print(f"Wrote {cur_path / 'gen/auto_vars.tex'}")


if __name__ == "__main__":
    main()