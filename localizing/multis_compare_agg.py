from localizing.pape_multis import make_fold_preds_for_multis, multis_to_fold_pred, pape_multis_equals_from_scratch
from localizing.predictions_repr import make_metrics_from_fold_results_multilevel
from pape.configs import BASE_MULTIS_CONFIG, BASE_PAPER_CONFIG
from pape.texrendering import TABLE_DIR, generate_latex_table, render_latex_table_to_file


def main():
    rows = []
    config = BASE_PAPER_CONFIG
    multi_config = BASE_MULTIS_CONFIG
    dev_mode = False
    locs = pape_multis_equals_from_scratch(
        probe_config=config,
        multi_config=multi_config,
        dev_mode=dev_mode,
    )
    for line_agg in [
        "mean", 
        "gmean", 
        "min", 
        "pre_min", 
        "pre_mean", 
    ]:
        print(f"Running {line_agg}")
        results = multis_to_fold_pred(
            locs, 
            n_folds=config.n_folds, 
            probe_config=config, 
            multi_config=multi_config,
            line_aggregator=line_agg,
        )
        rows.append({
            "line\\_agg": line_agg.replace("_", "\\_"),
            **make_metrics_from_fold_results_multilevel(results)['Line-Level'],
        })
    print(rows)
    table = generate_latex_table(
        rows,
        caption_content="Multisampling line-level results for different line aggregators",
    )
    (TABLE_DIR / "multis_compare_agg.tex").write_text(table)


if __name__ == "__main__":
    main()