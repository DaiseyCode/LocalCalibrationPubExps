from dataclasses import dataclass
import dataclasses

from regex import B
from localizing.direct_prompt import get_direct_prompt_results
from localizing.intrinsic import get_or_serialize_logprobs_fold_results
from localizing.localizing_structs import MultiSampleMode, MultiSamplingConfig
from localizing.pape_multis import make_fold_preds_for_multis
from localizing.predictions_repr import calc_exp_results_per_fold, make_metrics_from_fold_results_multilevel
from localizing.probe.agg_models.agg_config import ProbeConfig
from localizing.probe.run_all_probe_exps import get_fold_results_preds_probe
from pape.configs import BASE_PAPER_CONFIG, BASE_PAPER_CONFIG_QWEN, DEFAULT_LINE_AGGREGATOR_INTRINSIC, DEFAULT_LINE_AGGREGATOR_MULTIS, gen_model_name_qwen
from pape.dataset_stats_table import TABLE_DIR, generate_latex_table

BASE_MULTIS_CONFIG = MultiSamplingConfig(
    mode=MultiSampleMode.from_prompt,
    multi_temperature=0.8,
    target_num_samples=5,
)


def make_main_metrics_table_rows(
    config: ProbeConfig,
    dev_mode = False,
) -> list[dict[str, str]]:
    rows = []

    results = get_or_serialize_logprobs_fold_results(
        config,
        dev_mode=dev_mode,
        line_aggregator=DEFAULT_LINE_AGGREGATOR_INTRINSIC,
        problem_aggregator=DEFAULT_LINE_AGGREGATOR_INTRINSIC,
    )
    rows.append(
        {
            "Technique": "Token Prob",
            **make_metrics_from_fold_results_multilevel(results),
        }
    )

    results = make_fold_preds_for_multis(
        config,
        BASE_MULTIS_CONFIG,
        line_aggregator=DEFAULT_LINE_AGGREGATOR_MULTIS,
        dev_mode=dev_mode,
    )
    rows.append(
        {
            "Technique": "Multisampling",
            **make_metrics_from_fold_results_multilevel(results),
        }
    )
    

    results = get_direct_prompt_results(
        config,
        dev_mode=dev_mode,
    )
    rows.append(
        {
            "Technique": "Line Verbalized",
            **make_metrics_from_fold_results_multilevel(results),
        }
    )

    results = get_fold_results_preds_probe(
        dataclasses.replace(
            config, 
            embed_lm_name="Qwen/Qwen2.5-Coder-0.5B",
        ), 
        dev_mode=dev_mode
    )[0]
    rows.append(
        {
            "Technique": "Probe-0.5B",
            **make_metrics_from_fold_results_multilevel(results),
        }
    )

    results = get_fold_results_preds_probe(
        dataclasses.replace(
            config, 
            embed_lm_name=gen_model_name_qwen,
        ), 
        dev_mode=dev_mode
    )[0]
    rows.append(
        {
            "Technique": "Probe-7B",
            **make_metrics_from_fold_results_multilevel(results),
        }
    )
    return rows


def make_and_save_main_metrics_table(
    config: ProbeConfig,
    dev_mode = False,
) -> None:
    model_name_friendly = {
        "gpt-4o": "GPT-4o",
        gen_model_name_qwen: "Qwen2.5-Coder",
    }[config.gen_model_name]
    model_name_longer = {
        "gpt-4o": "GPT-4o",
        gen_model_name_qwen: "Qwen2.5-Coder-7B-Instruct",
    }[config.gen_model_name]

    rows = make_main_metrics_table_rows(config, dev_mode)

    if config.gen_model_name == "gpt-4o":
        caption_content = f"Calibration Results for {model_name_longer} with various techniques. We show both unscaled and Platt scaled results and localizing at the line and token level."
    else:
        caption_content = f"Calibration Results for {model_name_longer} with various techniques."

    table = generate_latex_table(
        rows,
        caption_content=caption_content,
        label=f"tab:main_metrics_{model_name_friendly.replace('-', '_')}",
        caption_at_top=False,
        resize_to_fit=True,
        full_width=True,
    )
    (TABLE_DIR / f"main_metrics_table_{model_name_friendly}.tex").write_text(table)


def main():
    make_and_save_main_metrics_table(BASE_PAPER_CONFIG)
    make_and_save_main_metrics_table(BASE_PAPER_CONFIG_QWEN)


if __name__ == "__main__":
    main()