from dataclasses import dataclass
import pprint
import dataclasses
from localizing.direct_prompt import get_direct_prompt_results
from localizing.intrinsic import get_or_serialize_logprobs_fold_results
from localizing.localizing_structs import MultiSampleMode, MultiSamplingConfig
from localizing.pape_multis import make_fold_preds_for_multis
from localizing.predictions_repr import calc_exp_results_per_fold, make_metrics_from_fold_results_for_level_not_averaged, make_metrics_from_fold_results_multilevel
from localizing.probe.agg_models.agg_config import FoldMode, ProbeConfig
from localizing.probe.run_all_probe_exps import get_fold_results_preds_probe
from pape.configs import BASE_PAPER_CONFIG, BASE_PAPER_CONFIG_QWEN, DEFAULT_LINE_AGGREGATOR_INTRINSIC, DEFAULT_LINE_AGGREGATOR_MULTIS, gen_model_name_qwen
from pape.dataset_stats_table import TABLE_DIR, generate_latex_table
from lmwrapper.openai_wrapper import OpenAiModelNames

from pape.main_metrics_table import BASE_MULTIS_CONFIG


def main():
    config = BASE_PAPER_CONFIG
    dev_mode = False
    config = dataclasses.replace(
        config,
        fold_mode=FoldMode.dataset_fold,
    )

    all_results = []


    for gen_model_name in [
        OpenAiModelNames.gpt_4o,
        gen_model_name_qwen,
    ]:
        g_config = dataclasses.replace(
            config,
            gen_model_name=gen_model_name,
        )
        def append_results(results, technique):
            all_results.append({
                "results": results,
                "\\mutt": gen_model_name,
                "technique": technique,
            })
        #print("-- {gen_model_name} Getting Probe-0.5B results")
        #results = get_fold_results_preds_probe(
        #    dataclasses.replace(
        #        g_config, 
        #        embed_lm_name="Qwen/Qwen2.5-Coder-0.5B",
        #    ), 
        #    dev_mode=dev_mode
        #)[0]
        #append_results(results, "Probe-0.5B")

        #print("-- {gen_model_name} Getting Probe-7B results")
        #results = get_fold_results_preds_probe(
        #    dataclasses.replace(
        #        g_config, 
        #        embed_lm_name=gen_model_name_qwen,
        #    ), 
        #    dev_mode=dev_mode
        #)[0]
        #append_results(results, "Probe-7B")

        #print("-- {gen_model_name} Getting Multisampling results")
        #results = make_fold_preds_for_multis(
        #    g_config,
        #    BASE_MULTIS_CONFIG,
        #    line_aggregator=DEFAULT_LINE_AGGREGATOR_MULTIS,
        #    dev_mode=dev_mode,
        #)
        #append_results(results, "Multisampling")
        #exit()

        print("-- {gen_model_name} Getting Line Verbalized results")
        results = get_direct_prompt_results(
            g_config,
            dev_mode=dev_mode,
        )
        append_results(results, "Line Verbalized")

        print("-- {gen_model_name} Getting Token Prob results")
        results = get_or_serialize_logprobs_fold_results(
            g_config,
            dev_mode=dev_mode,
            line_aggregator=DEFAULT_LINE_AGGREGATOR_INTRINSIC,
            problem_aggregator=DEFAULT_LINE_AGGREGATOR_INTRINSIC,
        )
        append_results(results, "Token Prob")

    #print(results)
    #print(results.fold_names)
    #for train, test, fold_name in zip(
    #    results.train_preds_each_fold, 
    #    results.test_preds_each_fold, 
    #    results.fold_names
    #):
    #    print("-" * 100)
    #    print(fold_name)
    #    print(len(train))
    #    print(len(test))

    print("Assemble rows")
    rows = []
    for vals in all_results:
        results = vals["results"]
        line_metrics = make_metrics_from_fold_results_for_level_not_averaged(results, level="line")
        token_metrics = make_metrics_from_fold_results_for_level_not_averaged(results, level="token")
        for line_metric, token_metric in zip(line_metrics, token_metrics):
            line_metric = line_metric.copy()
            token_metric = token_metric.copy()
            assert line_metric["Fold"] == token_metric["Fold"]
            fold = line_metric["Fold"]
            fold = {
                "mbpp_plus": "MBPP+",
                "humaneval_plus": "HumanEval+",
            }.get(fold, fold)
            del line_metric["Fold"]
            del token_metric["Fold"]
            rows.append({
                "\\mutt": vals["\\mutt"],
                "Technique": vals["technique"],
                "Eval Dataset": fold,
                "Line-Level": line_metric,
                "Token-Level": token_metric,
            })

    rows.sort(key=lambda x: (x["\\mutt"], x["Technique"], x["Eval Dataset"]))
    table = generate_latex_table(
        rows,
        caption_content="Cross-SE Robustness with with a leave-one-out of each problem source.",
        full_width=True,
        label="tab:cross_se_robustness",
        resize_to_fit=True,
    )
    (TABLE_DIR / "cross_se_robustness.tex").write_text(table)



###########

def make_append_results(all_results, gen_model_name):
    def append_results(results, technique):
        all_results.append({
            "results": results,
            "\\mutt": gen_model_name,
            "technique": technique,
        })
    return append_results

def make_probe_cross_tables(
    config: ProbeConfig,
    dev_mode: bool,
):
    all_results = []
    for gen_model_name in [
        OpenAiModelNames.gpt_4o,
        gen_model_name_qwen,
    ]:
        g_config = dataclasses.replace(
            config,
            gen_model_name=gen_model_name,
        )
        append_results = make_append_results(all_results, gen_model_name)
        print(f"-- {gen_model_name} Getting Probe-0.5B results")
        results = get_fold_results_preds_probe(
            dataclasses.replace(
                g_config, 
                embed_lm_name="Qwen/Qwen2.5-Coder-0.5B",
            ), 
            dev_mode=dev_mode
        )[0]
        append_results(results, "Probe-0.5B")

        print(f"-- {gen_model_name} Getting Probe-7B results")
        results = get_fold_results_preds_probe(
            dataclasses.replace(
                g_config, 
                embed_lm_name=gen_model_name_qwen,
            ), 
            dev_mode=dev_mode
        )[0]
        append_results(results, "Probe-7B")

    print("Assemble rows")
    rows = prep_ros(all_results, include_technique=True)
    table = generate_latex_table(
        rows,
        caption_content="\\textbf{Probing} with a leave-one-out of each dataset for each generating Model Under Test (\\mutt).",
        full_width=True,
        label="tab:cross_se_robustness_probe",
        resize_to_fit=True,
    )
    (TABLE_DIR / "cross_se_robustness_probe.tex").write_text(table)


def make_multisampling_cross_tables(
    config: ProbeConfig,
    dev_mode: bool,
):
    all_results = []
    for gen_model_name in [
        OpenAiModelNames.gpt_4o,
        gen_model_name_qwen,
    ]:
        g_config = dataclasses.replace(
            config,
            gen_model_name=gen_model_name,
        )
        append_results = make_append_results(all_results, gen_model_name)
        print(f"-- {gen_model_name} Getting Multisampling results")
        results = make_fold_preds_for_multis(
            g_config,
            BASE_MULTIS_CONFIG,
            line_aggregator=DEFAULT_LINE_AGGREGATOR_MULTIS,
            dev_mode=dev_mode,
        )
        append_results(results, "Multisampling")

    print("Assemble rows")
    rows = prep_ros(all_results, include_technique=False)
    table = generate_latex_table(
        rows,
        caption_content="\\textbf{Multisampling} split by dataset for each generating Model Under Test (\\mutt).",
        full_width=True,
        label="tab:cross_se_robustness_multisampling",
        resize_to_fit=True,
    )
    (TABLE_DIR / "cross_se_robustness_multisampling.tex").write_text(table)


def make_line_verbalized_cross_tables(
    config: ProbeConfig,
    dev_mode: bool,
):
    all_results = []
    for gen_model_name in [
        OpenAiModelNames.gpt_4o,
        gen_model_name_qwen,
    ]:
        g_config = dataclasses.replace(
            config,
            gen_model_name=gen_model_name,
        )
        append_results = make_append_results(all_results, gen_model_name)
        print(f"-- {gen_model_name} Getting Line Verbalized results")
        results = get_direct_prompt_results(
            g_config,
            dev_mode=dev_mode,
        )
        append_results(results, "Line Verbalized")

    print("Assemble rows")
    rows = prep_ros(all_results, include_technique=False)
    table = generate_latex_table(
        rows,
        caption_content="\\textbf{Line Verbalized} split by dataset for each generating Model Under Test (\\mutt).",
        full_width=True,
        label="tab:cross_se_robustness_line_verbalized",
        resize_to_fit=True,
    )
    (TABLE_DIR / "cross_se_robustness_line_verbalized.tex").write_text(table)


def make_line_token_prob_cross_tables(
    config: ProbeConfig,
    dev_mode: bool,
):
    all_results = []
    for gen_model_name in [
        OpenAiModelNames.gpt_4o,
        gen_model_name_qwen,
    ]:
        g_config = dataclasses.replace(
            config,
            gen_model_name=gen_model_name,
        )
        append_results = make_append_results(all_results, gen_model_name)
        print("-- {gen_model_name} Getting Token Prob results")
        results = get_or_serialize_logprobs_fold_results(
            g_config,
            dev_mode=dev_mode,
        )
        append_results(results, "Token Prob")

    print("Assemble rows")
    rows = prep_ros(all_results, include_technique=False)
    table = generate_latex_table(
        rows,
        caption_content="\\textbf{Token Prob} estimates split by dataset for each generating Model Under Test (\\mutt).",
        full_width=True,
        label="tab:cross_se_robustness_line_token_prob",
        resize_to_fit=True,
    )
    (TABLE_DIR / "cross_se_robustness_line_token_prob.tex").write_text(table)


def prep_ros(all_results, include_technique: bool):
    rows = []
    # Define custom order for Eval Dataset
    eval_dataset_order = ["HumanEval+", "MBPP+", "LiveCodeBench", "RepoCod-s"]

    for vals in all_results:
        results = vals["results"]
        line_metrics = make_metrics_from_fold_results_for_level_not_averaged(results, level="line")
        token_metrics = make_metrics_from_fold_results_for_level_not_averaged(results, level="token")
        for line_metric, token_metric in zip(line_metrics, token_metrics):
            line_metric = line_metric.copy()
            token_metric = token_metric.copy()
            assert line_metric["Fold"] == token_metric["Fold"]
            fold = line_metric["Fold"]
            fold = {
                "mbpp_plus": "MBPP+",
                "humaneval_plus": "HumanEval+",
                "livecodebench": "LiveCodeBench",
                "repocod": "RepoCod-s",
            }.get(fold, fold)
            mutt = vals["\\mutt"]
            mutt = {
                "gpt-4o": "GPT-4o",
                gen_model_name_qwen: "Qwen2.5Coder",
            }.get(mutt, mutt)
            del line_metric["Fold"]
            del token_metric["Fold"]
            row = {
                "\\mutt": mutt,
            }
            if include_technique:
                row["Technique"] = vals["technique"]
            row.update({
                "Eval Dataset": fold,
                "Line-Level": line_metric,
                "Token-Level": token_metric,
            })
            rows.append(row)
    def eval_dataset_key(ds):
        return str(eval_dataset_order.index(ds) if ds in eval_dataset_order else ds)
    if include_technique:
        rows.sort(key=lambda x: (x["\\mutt"], x["Technique"], eval_dataset_key(x["Eval Dataset"])))
    else:
        rows.sort(key=lambda x: (x["\\mutt"], eval_dataset_key(x["Eval Dataset"])))
    return rows


def make_indiv_cross_tables():
    config = BASE_PAPER_CONFIG
    dev_mode = False
    config = dataclasses.replace(
        config,
        fold_mode=FoldMode.dataset_fold,
    )
    make_probe_cross_tables(config, dev_mode)
    make_multisampling_cross_tables(config, dev_mode)
    make_line_verbalized_cross_tables(config, dev_mode)
    make_line_token_prob_cross_tables(config, dev_mode)


if __name__ == "__main__":
    #main()
    make_indiv_cross_tables()